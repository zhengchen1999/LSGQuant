"""Forward hooks used by the calibration-data collection.

* :class:`SaveActivationVarHook` records the per-channel activation variance of
  every calibration sample; ``gen_calib_data.py --calib_mode var`` turns it into
  the scalar layer sensitivity consumed by VOLTS.
* :class:`SaveActivationMaxHook` records the per-channel activation absolute
  maximum used by the SmoothQuant / ViDiT-Q baselines
  (``gen_calib_data.py --calib_mode act_max``).
"""

from typing import List, Optional, Tuple

import torch
from torch import nn


class BaseHook:
    """Base class of the forward hooks."""

    def __init__(self):
        self.hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("hook must implement __call__")

    def clear(self):
        raise NotImplementedError("hook must implement clear")


def add_hook_to_module_(module: nn.Module, hook_cls: type[BaseHook]) -> BaseHook:
    hook = hook_cls()
    hook.hook_handle = module.register_forward_hook(hook)
    return hook


class SaveActivationVarHook(BaseHook):
    """Record the per-channel activation variance of one forward pass.

    Every forward pass (one calibration video) appends one ``[C]`` vector: the
    variance over the flattened batch and token dimensions of each channel.
    ``gen_calib_data.py --calib_mode var`` stacks those vectors into
    ``[N_samples, C]`` and reduces them to the scalar layer sensitivity used by
    VOLTS.
    """

    def __init__(self):
        super().__init__()
        self.outputs: List[torch.Tensor] = []

    def __call__(
        self,
        module: nn.Module,
        module_input: Tuple[torch.Tensor],
        module_output: torch.Tensor
    ) -> None:
        input_tensor = module_input[0]  # (batch_size, ..., channels)
        channels = input_tensor.shape[-1]

        # flatten the batch and token dimensions, keep the channels
        flattened_input = input_tensor.reshape(-1, channels)  # (total_elements, channels)
        channel_var = flattened_input.var(dim=0)  # [channels]

        self.outputs.append(channel_var)

    def clear(self) -> None:
        """Drop the recorded activations."""
        self.outputs.clear()


class SaveActivationMaxHook(BaseHook):
    """Record the per-channel activation absolute maximum."""

    def __init__(self):
        super().__init__()
        self.outputs: List[torch.Tensor] = []

    def __call__(
        self,
        module: nn.Module,
        module_input: Tuple[torch.Tensor],
        module_output: torch.Tensor
    ) -> None:
        input_tensor = module_input[0]
        channels = input_tensor.shape[-1]

        # flatten the batch dimension and take the maximum of every channel
        reshaped_input = input_tensor.reshape(-1, channels)  # (total_elements, channels)
        channel_max = reshaped_input.abs().amax(dim=0)  # [channels]

        self.outputs.append(channel_max)

    def clear(self) -> None:
        """Drop the recorded activations."""
        self.outputs.clear()
