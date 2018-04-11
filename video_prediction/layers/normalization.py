# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Contains the normalization layer classes and their functional aliases."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope


DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'


def fused_instance_norm(inputs,
                        center=True,
                        scale=True,
                        epsilon=1e-6,
                        activation_fn=None,
                        param_initializers=None,
                        reuse=None,
                        variables_collections=None,
                        outputs_collections=None,
                        trainable=True,
                        data_format=DATA_FORMAT_NHWC,
                        scope=None):
  """Functional interface for the instance normalization layer.

  Reference: https://arxiv.org/abs/1607.08022.

    "Instance Normalization: The Missing Ingredient for Fast Stylization"
    Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky

  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
      `batch_size`. The normalization is over all but the last dimension if
      `data_format` is `NHWC` and the second dimension if `data_format` is
      `NCHW`.
    center: If True, add offset of `beta` to normalized tensor. If False, `beta`
      is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: Small float added to variance to avoid dividing by zero.
    activation_fn: Activation function, default set to None to skip it and
      maintain a linear activation.
    param_initializers: Optional initializers for beta, gamma, moving mean and
      moving variance.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional collections for the variables.
    outputs_collections: Collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    data_format: A string. `NHWC` (default) and `NCHW` are supported.
    scope: Optional scope for `variable_scope`.

  Returns:
    A `Tensor` representing the output of the operation.

  Raises:
    ValueError: If `data_format` is neither `NHWC` nor `NCHW`.
    ValueError: If the rank of `inputs` is undefined.
    ValueError: If rank or channels dimension of `inputs` is undefined.
  """
  inputs = ops.convert_to_tensor(inputs)
  inputs_shape = inputs.shape
  inputs_rank = inputs.shape.ndims

  if inputs_rank is None:
    raise ValueError('Inputs %s has undefined rank.' % inputs.name)
  if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
    raise ValueError('data_format has to be either NCHW or NHWC.')

  with variable_scope.variable_scope(
      scope, 'InstanceNorm', [inputs], reuse=reuse) as sc:
    if data_format == DATA_FORMAT_NCHW:
      reduction_axis = 1
      # For NCHW format, rather than relying on implicit broadcasting, we
      # explicitly reshape the params to params_shape_broadcast when computing
      # the moments and the batch normalization.
      params_shape_broadcast = list(
          [1, inputs_shape[1].value] + [1 for _ in range(2, inputs_rank)])
    else:
      reduction_axis = inputs_rank - 1
      params_shape_broadcast = None
    moments_axes = list(range(inputs_rank))
    del moments_axes[reduction_axis]
    del moments_axes[0]
    params_shape = inputs_shape[reduction_axis:reduction_axis + 1]
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined channels dimension %s.' % (
          inputs.name, params_shape))

    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    dtype = inputs.dtype.base_dtype
    if param_initializers is None:
      param_initializers = {}
    if center:
      beta_collections = utils.get_variable_collections(
          variables_collections, 'beta')
      beta_initializer = param_initializers.get(
          'beta', init_ops.zeros_initializer())
      beta = variables.model_variable('beta',
                                      shape=params_shape,
                                      dtype=dtype,
                                      initializer=beta_initializer,
                                      collections=beta_collections,
                                      trainable=trainable)
      if params_shape_broadcast:
        beta = array_ops.reshape(beta, params_shape_broadcast)
    if scale:
      gamma_collections = utils.get_variable_collections(
          variables_collections, 'gamma')
      gamma_initializer = param_initializers.get(
          'gamma', init_ops.ones_initializer())
      gamma = variables.model_variable('gamma',
                                       shape=params_shape,
                                       dtype=dtype,
                                       initializer=gamma_initializer,
                                       collections=gamma_collections,
                                       trainable=trainable)
      if params_shape_broadcast:
        gamma = array_ops.reshape(gamma, params_shape_broadcast)

    if data_format == DATA_FORMAT_NHWC:
      inputs = array_ops.transpose(inputs, list(range(1, reduction_axis)) + [0, reduction_axis])
    if data_format == DATA_FORMAT_NCHW:
      inputs = array_ops.transpose(inputs, list(range(2, inputs_rank)) + [0, reduction_axis])
    hw, n, c = inputs.shape.as_list()[:-2], inputs.shape[-2].value, inputs.shape[-1].value
    inputs = array_ops.reshape(inputs, [1] + hw + [n * c])
    if inputs.shape.ndims != 4:
        # combine all the spatial dimensions into only two, e.g. [D, H, W] -> [DH, W]
        if inputs.shape.ndims > 4:
            inputs_ndims4_shape = [1, hw[0], -1, n * c]
        else:
            inputs_ndims4_shape = [1, 1, -1, n * c]
        inputs = array_ops.reshape(inputs, inputs_ndims4_shape)
    beta = array_ops.reshape(array_ops.tile(beta[None, :], [n, 1]), [-1])
    gamma = array_ops.reshape(array_ops.tile(gamma[None, :], [n, 1]), [-1])

    outputs, _, _ = nn.fused_batch_norm(
        inputs, gamma, beta, epsilon=epsilon,
        data_format=DATA_FORMAT_NHWC, name='instancenorm')

    outputs = array_ops.reshape(outputs, hw + [n, c])
    if data_format == DATA_FORMAT_NHWC:
      outputs = array_ops.transpose(outputs, [inputs_rank - 2] + list(range(inputs_rank - 2)) + [inputs_rank - 1])
    if data_format == DATA_FORMAT_NCHW:
      outputs = array_ops.transpose(outputs, [inputs_rank - 2, inputs_rank - 1] + list(range(inputs_rank - 2)))

    # if data_format == DATA_FORMAT_NHWC:
    #   inputs = array_ops.transpose(inputs, [0, reduction_axis] + list(range(1, reduction_axis)))
    # inputs_nchw_shape = inputs.shape
    # inputs = array_ops.reshape(inputs, [1, -1] + inputs_nchw_shape.as_list()[2:])
    # if inputs.shape.ndims != 4:
    #     # combine all the spatial dimensions into only two, e.g. [D, H, W] -> [DH, W]
    #     if inputs.shape.ndims > 4:
    #         inputs_ndims4_shape = inputs.shape.as_list()[:2] + [-1, inputs_nchw_shape.as_list()[-1]]
    #     else:
    #         inputs_ndims4_shape = inputs.shape.as_list()[:2] + [1, -1]
    #     inputs = array_ops.reshape(inputs, inputs_ndims4_shape)
    # beta = array_ops.reshape(array_ops.tile(beta[None, :], [inputs_nchw_shape[0].value, 1]), [-1])
    # gamma = array_ops.reshape(array_ops.tile(gamma[None, :], [inputs_nchw_shape[0].value, 1]), [-1])
    #
    # outputs, _, _ = nn.fused_batch_norm(
    #     inputs, gamma, beta, epsilon=epsilon,
    #     data_format=DATA_FORMAT_NCHW, name='instancenorm')
    #
    # outputs = array_ops.reshape(outputs, inputs_nchw_shape)
    # if data_format == DATA_FORMAT_NHWC:
    #   outputs = array_ops.transpose(outputs, [0] + list(range(2, inputs_rank)) + [1])

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)
