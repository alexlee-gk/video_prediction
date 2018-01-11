import itertools
from collections import OrderedDict

import six
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.training import device_setter


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


def replace_read_ops(loss, var_list):
    """
    Replaces read ops of each variable in `vars` with new read ops obtained
    from `read_value()`, thus forcing to read the most up-to-date values of
    the variables (which might incur copies across devices).
    The graph is seeded from the tensor `loss`.
    """
    dg_ops = set(ge.get_walks_intersection_ops([var.op for var in var_list], loss))
    read_ops = []
    enter_ops = []
    for var in var_list:
        output, = var.op.outputs
        read_op, = set(output.consumers()) & dg_ops
        output, = read_op.outputs
        enter_op, = set(output.consumers()) & dg_ops
        read_ops.append(read_op)
        enter_ops.append(enter_op)

    for var, read_op, enter_op in zip(var_list, read_ops, enter_ops):
        with tf.name_scope('/'.join(read_op.name.split('/')[:-1])):
            with tf.device(read_op.device):
                ge.connect(ge.sgv(var.read_value().op), ge.sgv(enter_op))


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


def add_image_summaries(outputs):
    for name, output in outputs.items():
        with tf.name_scope("%s_summary" % name):
            if output.shape.ndims == 6:
                # concatenate last dimension vertically
                output = tf.concat(tf.unstack(output, axis=-1), axis=-3)
            if output.shape.ndims == 5:
                # concatenate time dimension horizontally
                output = tf.concat(tf.unstack(output, axis=1), axis=2)
            if output.shape.ndims == 4:
                output = tf.image.convert_image_dtype(output, dtype=tf.uint8, saturate=True)
                tf.summary.image(name, output)
            else:
                raise NotImplementedError


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


def reduce_tensors(all_names_and_values):
    if any(isinstance(names_and_values, tf.Tensor) or names_and_values for names_and_values in all_names_and_values):
        num_gpus = len(all_names_and_values)
        if isinstance(all_names_and_values[0], tf.Tensor):
            all_value = all_names_and_values
            if all_value[0].shape.ndims == 0:
                reduced_value = tf.add_n(all_value) / tf.to_float(num_gpus)
            else:
                reduced_value = tf.concat(all_value, axis=0)
            return reduced_value
        else:
            if isinstance(all_names_and_values[0], (dict, OrderedDict)):
                dtype = type(all_names_and_values[0])
                assert all(isinstance(names_and_values, dtype) for names_and_values in all_names_and_values[1:])
                all_names_and_values = [names_and_values.items() for names_and_values in all_names_and_values]
            else:
                dtype = None
            reduced_names_and_values = []
            for all_name_and_value in zip(*all_names_and_values):
                all_name, all_value = zip(*all_name_and_value)
                assert all(name == all_name[0] for name in all_name[1:])
                name = all_name[0]
                if isinstance(all_value[0], tuple):
                    if len(all_value[0]) == 2:
                        all_loss, all_weight = zip(*all_value)
                        loss = tf.add_n(all_loss) / tf.to_float(num_gpus)
                        if isinstance(all_weight[0], tf.Tensor):
                            with tf.control_dependencies([tf.assert_equal(weight, all_weight[0]) for weight in all_weight[1:]]):
                                weight = tf.identity(all_weight[0])
                        else:
                            assert all(weight == all_weight[0] for weight in all_weight[1:])
                            weight = all_weight[0]
                        value = (loss, weight)
                    else:
                        raise NotImplementedError
                elif all_value[0].shape.ndims == 0:
                    value = tf.add_n(all_value) / tf.to_float(num_gpus)
                else:
                    value = tf.concat(all_value, axis=0)
                reduced_names_and_values.append((name, value))
            if dtype is not None:
                reduced_names_and_values = dtype(reduced_names_and_values)
            return reduced_names_and_values
    else:
        return all_names_and_values[0]
