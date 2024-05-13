# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A Predictor wrapping a one-step Predictor to make autoregressive predictions.预测器包装一个一步预测器
"""

from typing import Optional, cast

from absl import logging
from graphcast import predictor_base
from graphcast import xarray_jax
from graphcast import xarray_tree
import haiku as hk
import jax
import xarray


def _unflatten_and_expand_time(flat_variables, tree_def, time_coords):
  variables = jax.tree_util.tree_unflatten(tree_def, flat_variables)
  return variables.expand_dims(time=time_coords, axis=0)


def _get_flat_arrays_and_single_timestep_treedef(variables):
  flat_arrays = jax.tree_util.tree_leaves(variables.transpose('time', ...))
  _, treedef = jax.tree_util.tree_flatten(variables.isel(time=0, drop=True))
  return flat_arrays, treedef


class Predictor(predictor_base.Predictor):
  """Wraps a one-step Predictor to make multi-step predictions autoregressively.

  The wrapped Predictor will be used to predict a single timestep conditional
  on the inputs passed to the outer Predictor. Its predictions are then
  passed back in as inputs at the next timestep, for as many timesteps as are
  requested in the targets_template. (When multiple timesteps of input are
  used, a rolling window of inputs is maintained with new predictions
  concatenated onto the end).

  You may ask for additional variables to be predicted as targets which aren't
  used as inputs. These will be predicted as output variables only and not fed
  back in autoregressively. All target variables must be time-dependent however.

  You may also specify static (non-time-dependent) inputs which will be passed
  in at each timestep but are not predicted.

  At present, any time-dependent inputs must also be present as targets so they
  can be passed in autoregressively.

  The loss of the wrapped one-step Predictor is averaged over all timesteps to
  give a loss for the autoregressive Predictor.
  
包装一个一步预测器，以自回归方式进行多步预测。

包装的预测器将用于预测单个时间步条件
在传递到外部预测器的输入上。它的预测是
在下一个时间步长作为输入传回，时间步长尽可能多
在targets_template中请求。（当输入的多个时间步长
使用时，使用新的预测来保持输入的滚动窗口
连接到末端）。

您可以要求将其他变量作为目标进行预测，而这些变量不是
用作输入。这些将仅作为输出变量进行预测，而不是馈送
自回归。然而，所有目标变量都必须与时间相关。

您还可以指定将传递的静态（非时间相关）输入
在每个时间步长中，但是不被预测。

目前，任何与时间相关的输入也必须作为目标存在，因此
可以自回归传递。

包裹的一步预测器的损失在所有时间步长上平均为
给出自回归预测器的损失。
  """

  def __init__(
      self,
      predictor: predictor_base.Predictor,
      noise_level: Optional[float] = None,
      gradient_checkpointing: bool = False,
      ):
    """Initializes an autoregressive predictor wrapper.

    Args:
      predictor: A predictor to wrap in an auto-regressive way.
      noise_level: Optional value that multiplies the standard normal noise
        added to the time-dependent variables of the predictor inputs. In
        particular, no noise is added to the predictions that are fed back
        auto-regressively. Defaults to not adding noise.
      gradient_checkpointing: If True, gradient checkpointing will be
        used at each step of the computation to save on memory. Roughtly this
        should make the backwards pass two times more expensive, and the time
        per step counting the forward pass, should only increase by about 50%.
        Note this parameter will be ignored with a warning if the scan sequence
        length is 1.
        初始化自回归预测器包装。

Args：
预测器：一个以自回归方式包装的预测器。
noise_level：乘以标准法线噪波的可选值
添加到预测器输入的时间相关变量。在里面
特别地，没有噪声被添加到反馈的预测中
自动回归。默认为不添加噪波。
gradient_checkpointing：如果为True，则梯度检查点将
用于计算的每个步骤以节省内存。大致如此
应该会使向后传球的费用增加两倍，而且时间
计算向前传球的每一步应仅增加约50%。
注意，如果扫描顺序为
长度为1。
    """
    self._predictor = predictor
    self._noise_level = noise_level
    self._gradient_checkpointing = gradient_checkpointing

  def _get_and_validate_constant_inputs(self, inputs, targets, forcings):
    constant_inputs = inputs.drop_vars(targets.keys(), errors='ignore')
    constant_inputs = constant_inputs.drop_vars(
        forcings.keys(), errors='ignore')
    for name, var in constant_inputs.items():
      if 'time' in var.dims:
        raise ValueError(
            f'Time-dependent input variable {name} must either be a forcing '
            'variable, or a target variable to allow for auto-regressive '
            'feedback.')
    return constant_inputs

  def _validate_targets_and_forcings(self, targets, forcings):
    for name, var in targets.items():
      if 'time' not in var.dims:
        raise ValueError(f'Target variable {name} must be time-dependent.')

    for name, var in forcings.items():
      if 'time' not in var.dims:
        raise ValueError(f'Forcing variable {name} must be time-dependent.')

    overlap = forcings.keys() & targets.keys()
    if overlap:
      raise ValueError('The following were specified as both targets and '
                       f'forcings, which isn\'t allowed: {overlap}')

  def _update_inputs(self, inputs, next_frame):
    num_inputs = inputs.dims['time']

    predicted_or_forced_inputs = next_frame[list(inputs.keys())]

    # Combining datasets with inputs and target time stamps aligns them.
    # Only keep the num_inputs trailing frames for use as next inputs.
    # 将数据集与输入和目标时间戳相结合，使它们对齐。
    #只保留num_inputs后面的帧作为下一个输入。
    return (xarray.concat([inputs, predicted_or_forced_inputs], dim='time')
            .tail(time=num_inputs)
            # Update the time coordinate to reset the lead times for
            # next AR iteration.
            .assign_coords(time=inputs.coords['time']))

  def __call__(self,
               inputs: xarray.Dataset,
               targets_template: xarray.Dataset,
               forcings: xarray.Dataset,
               **kwargs) -> xarray.Dataset:
    """Calls the Predictor.

    Args:
      inputs: input variable used to make predictions. Inputs can include both
        time-dependent and time independent variables. Any time-dependent
        input variables must also be present in the targets_template or the
        forcings.
      targets_template: A target template containing informations about which
        variables should be predicted and the time alignment of the predictions.
        All target variables must be time-dependent.
        The number of time frames is used to set the number of unroll of the AR
        predictor (e.g. multiple unroll of the inner predictor for one time step
        in the targets is not supported yet).
      forcings: Variables that will be fed to the model. The variables
        should not overlap with the target ones. The time coordinates of the
        forcing variables should match the target ones.
        Forcing variables which are also present in the inputs, will be used to
        supply ground-truth values for those inputs when they are passed to the
        underlying predictor at timesteps beyond the first timestep.
      **kwargs: Additional arguments passed along to the inner Predictor.

    Returns:
      predictions: the model predictions matching the target template.

    Raise:
      ValueError: if the time coordinates of the inputs and targets are not
        different by a constant time step.
    """

    constant_inputs = self._get_and_validate_constant_inputs(
        inputs, targets_template, forcings)
    self._validate_targets_and_forcings(targets_template, forcings)

    # After the above checks, the remaining inputs must be time-dependent:在上述检查之后，其余输入必须与时间相关：
    inputs = inputs.drop_vars(constant_inputs.keys())

    # A predictions template only including the next time to predict.
    target_template = targets_template.isel(time=[0])

    flat_forcings, forcings_treedef = (
        _get_flat_arrays_and_single_timestep_treedef(forcings))
    scan_variables = flat_forcings

    def one_step_prediction(inputs, scan_variables):

      flat_forcings = scan_variables
      forcings = _unflatten_and_expand_time(flat_forcings, forcings_treedef,
                                            target_template.coords['time'])

      # Add constant inputs:
      all_inputs = xarray.merge([constant_inputs, inputs])
      predictions: xarray.Dataset = self._predictor(
          all_inputs, target_template,
          forcings=forcings,
          **kwargs)

      next_frame = xarray.merge([predictions, forcings])
      next_inputs = self._update_inputs(inputs, next_frame)

      # Drop the length-1 time dimension, since scan will concat all the outputs
      # for different times along a new leading time dimension:
      predictions = predictions.squeeze('time', drop=True)
      # We return the prediction flattened into plain jax arrays, because the
      # extra leading dimension added by scan prevents the tree_util
      # registrations in xarray_jax from unflattening them back into an
      # xarray.Dataset automatically:
      flat_pred = jax.tree_util.tree_leaves(predictions)
      return next_inputs, flat_pred

    if self._gradient_checkpointing:
      scan_length = targets_template.dims['time']
      if scan_length <= 1:
        logging.warning(
            'Skipping gradient checkpointing for sequence length of 1')
      else:
        # Just in case we take gradients (e.g. for control), although
        # in most cases this will just be for a forward pass.
        one_step_prediction = hk.remat(one_step_prediction)

    # Loop (without unroll) with hk states in cell (jax.lax.scan won't do).
    _, flat_preds = hk.scan(one_step_prediction, inputs, scan_variables)

    # The result of scan will have an extra leading axis on all arrays,
    # corresponding to the target times in this case. We need to be prepared for
    # it when unflattening the arrays back into a Dataset:
    scan_result_template = (
        target_template.squeeze('time', drop=True)
        .expand_dims(time=targets_template.coords['time'], axis=0))
    _, scan_result_treedef = jax.tree_util.tree_flatten(scan_result_template)
    predictions = jax.tree_util.tree_unflatten(scan_result_treedef, flat_preds)
    return predictions

  def loss(self,
           inputs: xarray.Dataset,
           targets: xarray.Dataset,
           forcings: xarray.Dataset,
           **kwargs
           ) -> predictor_base.LossAndDiagnostics:
    """The mean of the per-timestep losses of the underlying predictor."""
    if targets.sizes['time'] == 1:
      # If there is only a single target timestep then we don't need any
      # autoregressive feedback and can delegate the loss directly to the
      # underlying single-step predictor. This means the underlying predictor
      # doesn't need to implement .loss_and_predictions.
      return self._predictor.loss(inputs, targets, forcings, **kwargs)

    constant_inputs = self._get_and_validate_constant_inputs(
        inputs, targets, forcings)
    self._validate_targets_and_forcings(targets, forcings)
    # After the above checks, the remaining inputs must be time-dependent:
    inputs = inputs.drop_vars(constant_inputs.keys())

    if self._noise_level:
      def add_noise(x):
        return x + self._noise_level * jax.random.normal(
            hk.next_rng_key(), shape=x.shape)
      # Add noise to time-dependent variables of the inputs.
      inputs = jax.tree_map(add_noise, inputs)

    # The per-timestep targets passed by scan to one_step_loss below will have
    # no leading time axis. We need a treedef without the time axis to use
    # inside one_step_loss to unflatten it back into a dataset:
    flat_targets, target_treedef = _get_flat_arrays_and_single_timestep_treedef(
        targets)
    scan_variables = flat_targets

    flat_forcings, forcings_treedef = (
        _get_flat_arrays_and_single_timestep_treedef(forcings))
    scan_variables = (flat_targets, flat_forcings)

    def one_step_loss(inputs, scan_variables):
      flat_target, flat_forcings = scan_variables
      forcings = _unflatten_and_expand_time(flat_forcings, forcings_treedef,
                                            targets.coords['time'][:1])

      target = _unflatten_and_expand_time(flat_target, target_treedef,
                                          targets.coords['time'][:1])

      # Add constant inputs:
      all_inputs = xarray.merge([constant_inputs, inputs])

      (loss, diagnostics), predictions = self._predictor.loss_and_predictions(
          all_inputs,
          target,
          forcings=forcings,
          **kwargs)

      # Unwrap to jax arrays shape (batch,):
      loss, diagnostics = xarray_tree.map_structure(
          xarray_jax.unwrap_data, (loss, diagnostics))

      predictions = cast(xarray.Dataset, predictions)  # Keeps pytype happy.
      next_frame = xarray.merge([predictions, forcings])
      next_inputs = self._update_inputs(inputs, next_frame)

      return next_inputs, (loss, diagnostics)

    if self._gradient_checkpointing:
      scan_length = targets.dims['time']
      if scan_length <= 1:
        logging.warning(
            'Skipping gradient checkpointing for sequence length of 1')
      else:
        one_step_loss = hk.remat(one_step_loss)

    # We can pass inputs (the initial state of the loop) in directly as a
    # Dataset because the shape we pass in to scan is the same as the shape scan
    # passes to the inner function. But, for scan_variables, we must flatten the
    # targets (and unflatten them inside the inner function) because they are
    # passed to the inner function per-timestep without the original time axis.
    # The same apply to the optional forcing.
    _, (per_timestep_losses, per_timestep_diagnostics) = hk.scan(
        one_step_loss, inputs, scan_variables)

    # Re-wrap loss and diagnostics as DataArray and average them over time:
    (loss, diagnostics) = jax.tree_util.tree_map(
        lambda x: xarray_jax.DataArray(x, dims=('time', 'batch')).mean(  # pylint: disable=g-long-lambda
            'time', skipna=False),
        (per_timestep_losses, per_timestep_diagnostics))

    return loss, diagnostics
