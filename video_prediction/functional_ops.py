from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest


def foldl(fn, elems, initializer=None, parallel_iterations=10, back_prop=True,
          swap_memory=False, name=None):
  """
  Same as tf.foldl but with support for a possibly nested sequence of tensors.
  """
  if not callable(fn):
    raise TypeError("fn must be callable.")

  input_is_sequence = nest.is_sequence(elems)
  input_flatten = lambda x: nest.flatten(x) if input_is_sequence else [x]
  def input_pack(x):
    return nest.pack_sequence_as(elems, x) if input_is_sequence else x[0]

  if initializer is None:
    output_is_sequence = input_is_sequence
    output_flatten = input_flatten
    output_pack = input_pack
  else:
    output_is_sequence = nest.is_sequence(initializer)
    output_flatten = lambda x: nest.flatten(x) if output_is_sequence else [x]
    def output_pack(x):
      return (nest.pack_sequence_as(initializer, x)
              if output_is_sequence else x[0])

  elems_flat = input_flatten(elems)

  in_graph_mode = context.in_graph_mode()
  with ops.name_scope(name, "foldl", [elems]):
    # TODO(akshayka): Remove the in_graph_mode check once caching devices are
    # supported in Eager
    if in_graph_mode:
      # Any get_variable calls in fn will cache the first call locally
      # and not issue repeated network I/O requests for each iteration.
      varscope = vs.get_variable_scope()
      varscope_caching_device_was_none = False
      if varscope.caching_device is None:
        # TODO(ebrevdo): Change to using colocate_with here and in other
        # methods.
        varscope.set_caching_device(lambda op: op.device)
        varscope_caching_device_was_none = True

    # Convert elems to tensor array.
    elems_flat = [
        ops.convert_to_tensor(elem, name="elem") for elem in elems_flat]

    n = array_ops.shape(elems_flat[0])[0]

    # TensorArrays are always flat
    elems_ta = [
        tensor_array_ops.TensorArray(dtype=elem.dtype, size=n,
                                     dynamic_size=False,
                                     infer_shape=True)
        for elem in elems_flat]
    # Unpack elements
    elems_ta = [
        elem_ta.unstack(elem) for elem_ta, elem in zip(elems_ta, elems_flat)]

    if initializer is None:
      a_flat = [elem.read(0) for elem in elems_ta]
      i = constant_op.constant(1)
    else:
      initializer_flat = output_flatten(initializer)
      a_flat = [ops.convert_to_tensor(init) for init in initializer_flat]
      i = constant_op.constant(0)

    def compute(i, a_flat):
      packed_elems = input_pack([elem_ta.read(i) for elem_ta in elems_ta])
      packed_a = output_pack(a_flat)
      a_out = fn(packed_a, packed_elems)
      nest.assert_same_structure(
          elems if initializer is None else initializer, a_out)
      flat_a_out = output_flatten(a_out)
      return (i + 1, flat_a_out)

    _, r_a = control_flow_ops.while_loop(
        lambda i, a: i < n, compute, (i, a_flat),
        parallel_iterations=parallel_iterations,
        back_prop=back_prop,
        swap_memory=swap_memory)

    # TODO(akshayka): Remove the in_graph_mode check once caching devices are
    # supported in Eager
    if in_graph_mode and varscope_caching_device_was_none:
      varscope.set_caching_device(None)

    return output_pack(r_a)
