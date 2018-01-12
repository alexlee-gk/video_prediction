import itertools
from collections import OrderedDict

import numpy as np
import six
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.training import device_setter
from tensorflow.python.util import nest


def local_device_setter(num_devices=1,
                        ps_device_type='cpu',
                        worker_device='/cpu:0',
                        ps_ops=None,
                        ps_strategy=None):
    if ps_ops == None:
        ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

    if ps_strategy is None:
        ps_strategy = device_setter._RoundRobinStrategy(num_devices)
    if not six.callable(ps_strategy):
        raise TypeError("ps_strategy must be callable")

    def _local_device_chooser(op):
        current_device = pydev.DeviceSpec.from_string(op.device or "")

        node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
        if node_def.op in ps_ops:
            ps_device_spec = pydev.DeviceSpec.from_string(
                '/{}:{}'.format(ps_device_type, ps_strategy(op)))

            ps_device_spec.merge_from(current_device)
            return ps_device_spec.to_string()
        else:
            worker_device_spec = pydev.DeviceSpec.from_string(worker_device or "")
            worker_device_spec.merge_from(current_device)
            return worker_device_spec.to_string()

    return _local_device_chooser


def replace_read_ops(loss_or_losses, var_list):
    """
    Replaces read ops of each variable in `vars` with new read ops obtained
    from `read_value()`, thus forcing to read the most up-to-date values of
    the variables (which might incur copies across devices).
    The graph is seeded from the tensor(s) `loss_or_losses`.
    """
    # ops between var ops and the loss
    ops = set(ge.get_walks_intersection_ops([var.op for var in var_list], loss_or_losses))
    if not ops:  # loss_or_losses doesn't depend on any var in var_list, so there is nothiing to replace
        return

    # assume that for each variable, the only op required to compute the loss
    # is a read op, and there is exactly one per variable
    read_ops = []
    for var in var_list:
        output, = var.op.outputs
        read_op, = set(output.consumers()) & ops
        read_ops.append(read_op)

    for var, read_op in zip(var_list, read_ops):
        with tf.name_scope('/'.join(read_op.name.split('/')[:-1])):
            with tf.device(read_op.device):
                read_t, = read_op.outputs
                consumer_ops = set(read_t.consumers()) & ops
                ge.connect(ge.sgv(var.read_value().op), ge.sgv(consumer_ops))


def print_loss_info(losses, inputs, outputs, targets):
    def get_descendants(tensor, tensors):
        descendants = []
        for child in tensor.op.inputs:
            if child in tensors:
                descendants.append(child)
            else:
                descendants.extend(get_descendants(child, tensors))
        return descendants

    name_to_tensors = list(inputs.items()) + list(outputs.items()) + [('targets', targets)]
    tensor_to_names = OrderedDict([(v, k) for k, v in name_to_tensors])

    print(tf.get_default_graph().get_name_scope())
    for name, (loss, weight) in losses.items():
        print('  %s (%r)' % (name, weight))
        for descendant in get_descendants(loss, tensor_to_names.keys()):
            print('    %s' % tensor_to_names[descendant])


def tensor_to_clip(tensor):
    if tensor.shape.ndims == 6:
        # concatenate last dimension vertically
        tensor = tf.concat(tf.unstack(tensor, axis=-1), axis=-3)
    if tensor.shape.ndims == 5:
        # concatenate batch dimension horizontally
        tensor = tf.concat(tf.unstack(tensor, axis=0), axis=2)
    if tensor.shape.ndims == 4:
        tensor = tf.image.convert_image_dtype(tensor, dtype=tf.uint8, saturate=True)
    else:
        raise NotImplementedError
    return tensor


def tensor_to_image_batch(tensor):
    if tensor.shape.ndims == 6:
        # concatenate last dimension vertically
        tensor= tf.concat(tf.unstack(tensor, axis=-1), axis=-3)
    if tensor.shape.ndims == 5:
        # concatenate time dimension horizontally
        tensor = tf.concat(tf.unstack(tensor, axis=1), axis=2)
    if tensor.shape.ndims == 4:
        tensor = tf.image.convert_image_dtype(tensor, dtype=tf.uint8, saturate=True)
    else:
        raise NotImplementedError
    return tensor


def add_image_summaries(outputs):
    for name, output in outputs.items():
        with tf.name_scope("%s_summary" % name):
            tf.summary.image(name, tensor_to_image_batch(output))


def add_scalar_summaries(losses_or_metrics):
    for name, loss_or_metric in losses_or_metrics.items():
        if isinstance(loss_or_metric, tuple):
            loss_or_metric, _ = loss_or_metric
        with tf.name_scope("%s_summary" % name):
            tf.summary.scalar(name, loss_or_metric)


def compute_averaged_gradients(opt, tower_loss, **kwargs):
    tower_gradvars = []
    for loss in tower_loss:
        with tf.device(loss.device):
            gradvars = opt.compute_gradients(loss, **kwargs)
            tower_gradvars.append(gradvars)

    # Now compute global loss and gradients.
    gradvars = []
    with tf.name_scope('gradient_averaging'):
        all_grads = {}
        for grad, var in itertools.chain(*tower_gradvars):
            if grad is not None:
                all_grads.setdefault(var, []).append(grad)
        for var, grads in all_grads.items():
            # Average gradients on the same device as the variables
            # to which they apply.
            with tf.device(var.device):
                if len(grads) == 1:
                    avg_grad = grads[0]
                else:
                    avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
            gradvars.append((avg_grad, var))
    return gradvars


def _reduce_entries(*entries):
    num_gpus = len(entries)
    if entries[0] is None:
        assert all(entry is None for entry in entries[1:])
        reduced_entry = None
    elif isinstance(entries[0], tf.Tensor):
        if entries[0].shape.ndims == 0:
            reduced_entry = tf.add_n(entries) / tf.to_float(num_gpus)
        else:
            reduced_entry = tf.concat(entries, axis=0)
    elif np.isscalar(entries[0]) or isinstance(entries[0], np.ndarray):
        if np.isscalar(entries[0]) or entries[0].ndim == 0:
            reduced_entry = sum(entries) / float(num_gpus)
        else:
            reduced_entry = np.concatenate(entries, axis=0)
    elif isinstance(entries[0], tuple) and len(entries[0]) == 2:
        losses, weights = zip(*entries)
        loss = tf.add_n(losses) / tf.to_float(num_gpus)
        if isinstance(weights[0], tf.Tensor):
            with tf.control_dependencies([tf.assert_equal(weight, weights[0]) for weight in weights[1:]]):
                weight = tf.identity(weights[0])
        else:
            assert all(weight == weights[0] for weight in weights[1:])
            weight = weights[0]
        reduced_entry = (loss, weight)
    else:
        raise NotImplementedError
    return reduced_entry


def reduce_tensors(structures, shallow=False):
    if len(structures) == 1:
        reduced_structure = structures[0]
    else:
        if shallow:
            if isinstance(structures[0], dict):
                shallow_tree = type(structures[0])([(k, None) for k in structures[0]])
            else:
                shallow_tree = type(structures[0])([None for _ in structures[0]])
            reduced_structure = nest.map_structure_up_to(shallow_tree, _reduce_entries, *structures)
        else:
            reduced_structure = nest.map_structure(_reduce_entries, *structures)
    return reduced_structure
