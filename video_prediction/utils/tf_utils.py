import itertools
import os
from collections import OrderedDict

import numpy as np
import six
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.training import device_setter
from tensorflow.python.util import nest

from video_prediction.utils import ffmpeg_gif
from video_prediction.utils import gif_summary

IMAGE_SUMMARIES = "image_summaries"
EVAL_SUMMARIES = "eval_summaries"


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

    # filter out variables that are not involved in computing the loss
    var_list = [var for var in var_list if var.op in ops]

    for var in var_list:
        output, = var.op.outputs
        read_ops = set(output.consumers()) & ops
        for read_op in read_ops:
            with tf.name_scope('/'.join(read_op.name.split('/')[:-1])):
                with tf.device(read_op.device):
                    read_t, = read_op.outputs
                    consumer_ops = set(read_t.consumers()) & ops
                    # consumer_sgv might have multiple inputs, but we only care
                    # about replacing the input that is read_t
                    consumer_sgv = ge.sgv(consumer_ops)
                    consumer_sgv = consumer_sgv.remap_inputs([list(consumer_sgv.inputs).index(read_t)])
                    ge.connect(ge.sgv(var.read_value().op), consumer_sgv)


def print_loss_info(losses, *tensors):
    def get_descendants(tensor, tensors):
        descendants = []
        for child in tensor.op.inputs:
            if child in tensors:
                descendants.append(child)
            else:
                descendants.extend(get_descendants(child, tensors))
        return descendants

    name_to_tensors = itertools.chain(*[tensor.items() for tensor in tensors])
    tensor_to_names = OrderedDict([(v, k) for k, v in name_to_tensors])

    print(tf.get_default_graph().get_name_scope())
    for name, (loss, weight) in losses.items():
        print('  %s (%r)' % (name, weight))
        descendant_names = []
        for descendant in set(get_descendants(loss, tensor_to_names.keys())):
            descendant_names.append(tensor_to_names[descendant])
        for descendant_name in sorted(descendant_names):
            print('    %s' % descendant_name)


def with_flat_batch(flat_batch_fn, ndims=4):
    def fn(x, *args, **kwargs):
        shape = tf.shape(x)
        flat_batch_shape = tf.concat([[-1], shape[-(ndims-1):]], axis=0)
        flat_batch_shape.set_shape([ndims])
        flat_batch_x = tf.reshape(x, flat_batch_shape)
        flat_batch_r = flat_batch_fn(flat_batch_x, *args, **kwargs)
        r = nest.map_structure(lambda x: tf.reshape(x, tf.concat([shape[:-(ndims-1)], tf.shape(x)[1:]], axis=0)),
                               flat_batch_r)
        return r
    return fn


def transpose_batch_time(x):
    if isinstance(x, tf.Tensor) and x.shape.ndims >= 2:
        return tf.transpose(x, [1, 0] + list(range(2, x.shape.ndims)))
    else:
        return x


def dimension(inputs, axis=0):
    shapes = [input_.shape for input_ in nest.flatten(inputs)]
    s = tf.TensorShape([None])
    for shape in shapes:
        s = s.merge_with(shape[axis:axis + 1])
    dim = s[0].value
    return dim


def unroll_rnn(cell, inputs, scope=None, use_dynamic_rnn=True):
    """Chooses between dynamic_rnn and static_rnn if the leading time dimension is dynamic or not."""
    dim = dimension(inputs, axis=0)
    if use_dynamic_rnn or dim is None:
        return tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32,
                                 swap_memory=False, time_major=True, scope=scope)
    else:
        return static_rnn(cell, inputs, scope=scope)


def static_rnn(cell, inputs, scope=None):
    """Simple version of static_rnn."""
    with tf.variable_scope(scope or "rnn") as varscope:
        batch_size = dimension(inputs, axis=1)
        state = cell.zero_state(batch_size, tf.float32)
        flat_inputs = nest.flatten(inputs)
        flat_inputs = list(zip(*[tf.unstack(flat_input, axis=0) for flat_input in flat_inputs]))
        flat_outputs = []
        for time, flat_input in enumerate(flat_inputs):
            if time > 0:
                varscope.reuse_variables()
            input_ = nest.pack_sequence_as(inputs, flat_input)
            output, state = cell(input_, state)
            flat_output = nest.flatten(output)
            flat_outputs.append(flat_output)
        flat_outputs = [tf.stack(flat_output, axis=0) for flat_output in zip(*flat_outputs)]
        outputs = nest.pack_sequence_as(output, flat_outputs)
        return outputs, state


def maybe_pad_or_slice(tensor, desired_length):
    length = tensor.shape.as_list()[0]
    if length < desired_length:
        paddings = [[0, desired_length - length]] + [[0, 0]] * (tensor.shape.ndims - 1)
        tensor = tf.pad(tensor, paddings)
    elif length > desired_length:
        tensor = tensor[:desired_length]
    assert tensor.shape.as_list()[0] == desired_length
    return tensor


def tensor_to_clip(tensor):
    if tensor.shape.ndims == 6:
        # concatenate last dimension vertically
        tensor = tf.concat(tf.unstack(tensor, axis=-1), axis=-3)
    if tensor.shape.ndims == 5:
        # concatenate batch dimension horizontally
        tensor = tf.concat(tf.unstack(tensor, axis=0), axis=2)
    if tensor.shape.ndims == 4:
        # keep up to the first 3 channels
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
        # keep up to the first 3 channels
        tensor = tf.image.convert_image_dtype(tensor, dtype=tf.uint8, saturate=True)
    else:
        raise NotImplementedError
    return tensor


def _as_name_scope_map(values):
    name_scope_to_values = {}
    for name, value in values.items():
        name_scope = name.split('/')[0]
        name_scope_to_values.setdefault(name_scope, {})
        name_scope_to_values[name_scope][name] = value
    return name_scope_to_values


def add_image_summaries(outputs, max_outputs=8, collections=None):
    if collections is None:
        collections = [tf.GraphKeys.SUMMARIES, IMAGE_SUMMARIES]
    for name_scope, outputs in _as_name_scope_map(outputs).items():
        with tf.name_scope(name_scope):
            for name, output in outputs.items():
                if max_outputs:
                    output = output[:max_outputs]
                output = tensor_to_image_batch(output)
                if output.shape[-1] not in (1, 3):
                    # these are feature maps, so just skip them
                    continue
                tf.summary.image(name, output, collections=collections)


def add_gif_summaries(outputs, max_outputs=8, collections=None):
    if collections is None:
        collections = [tf.GraphKeys.SUMMARIES, IMAGE_SUMMARIES]
    for name_scope, outputs in _as_name_scope_map(outputs).items():
        with tf.name_scope(name_scope):
            for name, output in outputs.items():
                if max_outputs:
                    output = output[:max_outputs]
                output = tensor_to_clip(output)
                if output.shape[-1] not in (1, 3):
                    # these are feature maps, so just skip them
                    continue
                gif_summary.gif_summary(name, output[None], fps=4, collections=collections)


def add_scalar_summaries(losses_or_metrics, collections=None):
    for name_scope, losses_or_metrics in _as_name_scope_map(losses_or_metrics).items():
        with tf.name_scope(name_scope):
            for name, loss_or_metric in losses_or_metrics.items():
                if isinstance(loss_or_metric, tuple):
                    loss_or_metric, _ = loss_or_metric
                tf.summary.scalar(name, loss_or_metric, collections=collections)


def add_summaries(outputs, collections=None):
    scalar_outputs = OrderedDict()
    image_outputs = OrderedDict()
    gif_outputs = OrderedDict()
    for name, output in outputs.items():
        if not isinstance(output, tf.Tensor):
            continue
        if output.shape.ndims == 0:
            scalar_outputs[name] = output
        elif output.shape.ndims == 4:
            image_outputs[name] = output
        elif output.shape.ndims > 4 and output.shape[4].value in (1, 3):
            gif_outputs[name] = output
    add_scalar_summaries(scalar_outputs, collections=collections)
    add_image_summaries(image_outputs, collections=collections)
    add_gif_summaries(gif_outputs, collections=collections)


def plot_buf(y):
    def _plot_buf(y):
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        from matplotlib.figure import Figure
        import io
        fig = Figure(figsize=(3, 3))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.plot(y)
        ax.grid(axis='y')
        fig.tight_layout(pad=0)

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        return buf.getvalue()

    s = tf.py_func(_plot_buf, [y], tf.string)
    return s


def add_plot_image_summaries(metrics, collections=None):
    if collections is None:
        collections = [IMAGE_SUMMARIES]
    for name_scope, metrics in _as_name_scope_map(metrics).items():
        with tf.name_scope(name_scope):
            for name, metric in metrics.items():
                try:
                    buf = plot_buf(metric)
                except:
                    continue
                image = tf.image.decode_png(buf, channels=4)
                image = tf.expand_dims(image, axis=0)
                tf.summary.image(name, image, max_outputs=1, collections=collections)


def plot_summary(name, x, y, display_name=None, description=None, collections=None):
    """
    Hack that uses pr_curve summaries for 2D plots.

    Args:
        x: 1-D tensor with values in increasing order.
        y: 1-D tensor with static shape.

    Note: tensorboard needs to be modified and compiled from source to disable
    default axis range [-0.05, 1.05].
    """
    from tensorboard import summary as summary_lib
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    with tf.control_dependencies([
        tf.assert_equal(tf.shape(x), tf.shape(y)),
        tf.assert_equal(y.shape.ndims, 1),
    ]):
        y = tf.identity(y)
    num_thresholds = y.shape[0].value
    if num_thresholds is None:
        raise ValueError('Size of y needs to be statically defined for num_thresholds argument')
    summary = summary_lib.pr_curve_raw_data_op(
        name,
        true_positive_counts=tf.ones(num_thresholds),
        false_positive_counts=tf.ones(num_thresholds),
        true_negative_counts=tf.ones(num_thresholds),
        false_negative_counts=tf.ones(num_thresholds),
        precision=y[::-1],
        recall=x[::-1],
        num_thresholds=num_thresholds,
        display_name=display_name,
        description=description,
        collections=collections)
    return summary


def add_plot_summaries(metrics, x_offset=0, collections=None):
    for name_scope, metrics in _as_name_scope_map(metrics).items():
        with tf.name_scope(name_scope):
            for name, metric in metrics.items():
                plot_summary(name, x_offset + tf.range(tf.shape(metric)[0]), metric, collections=collections)


def add_plot_and_scalar_summaries(metrics, x_offset=0, collections=None):
    for name_scope, metrics in _as_name_scope_map(metrics).items():
        with tf.name_scope(name_scope):
            for name, metric in metrics.items():
                tf.summary.scalar(name, tf.reduce_mean(metric), collections=collections)
                plot_summary(name, x_offset + tf.range(tf.shape(metric)[0]), metric, collections=collections)


def convert_tensor_to_gif_summary(summ):
    if isinstance(summ, bytes):
        summary_proto = tf.Summary()
        summary_proto.ParseFromString(summ)
        summ = summary_proto

    summary = tf.Summary()
    for value in summ.value:
        tag = value.tag
        try:
            images_arr = tf.make_ndarray(value.tensor)
        except TypeError:
            summary.value.add(tag=tag, image=value.image)
            continue

        if len(images_arr.shape) == 5:
            images_arr = np.concatenate(list(images_arr), axis=-2)
        if len(images_arr.shape) != 4:
            raise ValueError('Tensors must be 4-D or 5-D for gif summary.')
        channels = images_arr.shape[-1]
        if channels < 1 or channels > 4:
            raise ValueError('Tensors must have 1, 2, 3, or 4 color channels for gif summary.')

        encoded_image_string = ffmpeg_gif.encode_gif(images_arr, fps=4)

        image = tf.Summary.Image()
        image.height = images_arr.shape[-3]
        image.width = images_arr.shape[-2]
        image.colorspace = channels  # 1: grayscale, 2: grayscale + alpha, 3: RGB, 4: RGBA
        image.encoded_image_string = encoded_image_string
        summary.value.add(tag=tag, image=image)
    return summary


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


# the next 3 function are from tensorpack:
# https://github.com/tensorpack/tensorpack/blob/master/tensorpack/graph_builder/utils.py
def split_grad_list(grad_list):
    """
    Args:
        grad_list: K x N x 2

    Returns:
        K x N: gradients
        K x N: variables
    """
    g = []
    v = []
    for tower in grad_list:
        g.append([x[0] for x in tower])
        v.append([x[1] for x in tower])
    return g, v


def merge_grad_list(all_grads, all_vars):
    """
    Args:
        all_grads (K x N): gradients
        all_vars(K x N): variables

    Return:
        K x N x 2: list of list of (grad, var) pairs
    """
    return [list(zip(gs, vs)) for gs, vs in zip(all_grads, all_vars)]


def allreduce_grads(all_grads, average):
    """
    All-reduce average the gradients among K devices. Results are broadcasted to all devices.

    Args:
        all_grads (K x N): List of list of gradients. N is the number of variables.
        average (bool): average gradients or not.

    Returns:
        K x N: same as input, but each grad is replaced by the average over K devices.
    """
    from tensorflow.contrib import nccl
    nr_tower = len(all_grads)
    if nr_tower == 1:
        return all_grads
    new_all_grads = []  # N x K
    for grads in zip(*all_grads):
        summed = nccl.all_sum(grads)

        grads_for_devices = []  # K
        for g in summed:
            with tf.device(g.device):
                # tensorflow/benchmarks didn't average gradients
                if average:
                    g = tf.multiply(g, 1.0 / nr_tower)
            grads_for_devices.append(g)
        new_all_grads.append(grads_for_devices)

    # transpose to K x N
    ret = list(zip(*new_all_grads))
    return ret


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


def get_checkpoint_restore_saver(checkpoint, var_list=None, skip_global_step=False, restore_to_checkpoint_mapping=None):
    if os.path.isdir(checkpoint):
        # latest_checkpoint doesn't work when the path has special characters
        checkpoint = tf.train.latest_checkpoint(checkpoint)
    checkpoint_reader = tf.pywrap_tensorflow.NewCheckpointReader(checkpoint)
    checkpoint_var_names = checkpoint_reader.get_variable_to_shape_map().keys()
    restore_to_checkpoint_mapping = restore_to_checkpoint_mapping or (lambda name, _: name.split(':')[0])
    if not var_list:
        var_list = tf.global_variables()
    restore_vars = {restore_to_checkpoint_mapping(var.name, checkpoint_var_names): var for var in var_list}
    if skip_global_step and 'global_step' in restore_vars:
        del restore_vars['global_step']
    # restore variables that are both in the global graph and in the checkpoint
    restore_and_checkpoint_vars = {name: var for name, var in restore_vars.items() if name in checkpoint_var_names}
    restore_saver = tf.train.Saver(max_to_keep=1, var_list=restore_and_checkpoint_vars, filename=checkpoint)
    # print out information regarding variables that were not restored or used for restoring
    restore_not_in_checkpoint_vars = {name: var for name, var in restore_vars.items() if
                                      name not in checkpoint_var_names}
    checkpoint_not_in_restore_var_names = [name for name in checkpoint_var_names if name not in restore_vars]
    if skip_global_step and 'global_step' in checkpoint_not_in_restore_var_names:
        checkpoint_not_in_restore_var_names.remove('global_step')
    if restore_not_in_checkpoint_vars:
        print("global variables that were not restored because they are "
              "not in the checkpoint:")
        for name, _ in sorted(restore_not_in_checkpoint_vars.items()):
            print("    ", name)
    if checkpoint_not_in_restore_var_names:
        print("checkpoint variables that were not used for restoring "
              "because they are not in the graph:")
        for name in sorted(checkpoint_not_in_restore_var_names):
            print("    ", name)
    return restore_saver, checkpoint


def pixel_distribution(pos, height, width):
    batch_size = pos.get_shape().as_list()[0]
    y, x = tf.unstack(pos, 2, axis=1)

    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    Ia = tf.reshape(tf.one_hot(y0 * width + x0, height * width), [batch_size, height, width])
    Ib = tf.reshape(tf.one_hot(y1 * width + x0, height * width), [batch_size, height, width])
    Ic = tf.reshape(tf.one_hot(y0 * width + x1, height * width), [batch_size, height, width])
    Id = tf.reshape(tf.one_hot(y1 * width + x1, height * width), [batch_size, height, width])

    x0_f = tf.cast(x0, 'float32')
    x1_f = tf.cast(x1, 'float32')
    y0_f = tf.cast(y0, 'float32')
    y1_f = tf.cast(y1, 'float32')
    wa = ((x1_f - x) * (y1_f - y))[:, None, None]
    wb = ((x1_f - x) * (y - y0_f))[:, None, None]
    wc = ((x - x0_f) * (y1_f - y))[:, None, None]
    wd = ((x - x0_f) * (y - y0_f))[:, None, None]

    return tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])


def flow_to_rgb(flows):
    """The last axis should have dimension 2, for x and y values."""

    def cartesian_to_polar(x, y):
        magnitude = tf.sqrt(tf.square(x) + tf.square(y))
        angle = tf.atan2(y, x)
        return magnitude, angle

    mag, ang = cartesian_to_polar(*tf.unstack(flows, axis=-1))
    ang_normalized = (ang + np.pi) / (2 * np.pi)
    mag_min = tf.reduce_min(mag)
    mag_max = tf.reduce_max(mag)
    mag_normalized = (mag - mag_min) / (mag_max - mag_min)
    hsv = tf.stack([ang_normalized, tf.ones_like(ang), mag_normalized], axis=-1)
    rgb = tf.image.hsv_to_rgb(hsv)
    return rgb
