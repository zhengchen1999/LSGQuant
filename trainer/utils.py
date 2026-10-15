import imageio, torch
from accelerate import Accelerator
import numpy as np
import math
import imageio.v3 as iio


def linear_warmup_decay_with_min_lr(
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,  # e.g. 0.1 keeps at least 10% of base_lr
):
    """
    Build the ``lr_lambda`` of a ``LambdaLR``.

    - 0 -> warmup_steps: linear warm-up, 0 -> 1
    - warmup_steps -> total_steps: linear decay, 1 -> min_lr_ratio

    Parameters
    ----------
    warmup_steps : int
        Number of warm-up steps (>0).
    total_steps : int
        Total number of steps (>warmup_steps).
    base_lr : float
        Learning rate set in the optimizer; the output of this function
        multiplies it.
    min_lr_ratio : float
        Ratio between the final and the initial learning rate
        (0 < min_lr_ratio <= 1).

    Returns
    -------
    Callable[[int], float]
        The ``lr_lambda`` passed to ``LambdaLR``.
    """
    assert 0 < min_lr_ratio <= 1, "`min_lr_ratio` must be in (0, 1]"
    assert total_steps > warmup_steps > 0, "`total_steps` must be larger than `warmup_steps`"

    def lr_lambda(current_step: int) -> float:
        # warm-up
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        # decay
        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))

    return lr_lambda


def cosine_schedule_with_min_lr(
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,  # e.g. 0.1 keeps at least 10% of base_lr
):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (1 - min_lr_ratio) * cosine + min_lr_ratio
    return lr_lambda


class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self

    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules

    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names

    def export_trainable_state_dict(self, state_dict, remove_prefix, trainable_models):
        trainable_param_names = self.trainable_param_names()
        filtered_state = {name: param for name, param in state_dict.items() if name in trainable_param_names}

        model_state_dict = {model: {} for model in trainable_models}
        for name, param in filtered_state.items():
            if name.startswith(remove_prefix):
                name = name[len(remove_prefix):]

            for model in trainable_models:
                prefix = model + "."
                if name.startswith(prefix):
                    clean_name = name[len(prefix):]
                    model_state_dict[model][clean_name] = param
                    break  # every parameter belongs to a single module

        return model_state_dict


def save_video_with_imageio(frame_list, output_path, fps=8, format='yuv444p'):
    """
    Save a list of PIL.Image frames to .mp4 using imageio.v3.imwrite with ffmpeg backend.

    Args:
        frame_list (List[PIL.Image]): list of RGB images (same size)
        output_path (str): where to save the .mp4 file
        fps (int): frames per second
        format (str): 'yuv444p' or 'yuv420p'
    """
    # Convert all frames to uint8 NumPy arrays
    frames = [np.array(frame.convert("RGB")) for frame in frame_list]  # each is (H, W, 3)
    frames = np.stack(frames, axis=0)  # shape: (F, H, W, 3)

    # Save video using imageio
    if format == 'yuv444p':
        iio.imwrite(
            output_path,
            frames,
            fps=fps,
            codec='libx264',
            pixelformat='yuv444p',
            macro_block_size=None,
            ffmpeg_params=['-crf', '0'],
        )
    else:
        iio.imwrite(
            output_path,
            frames,
            fps=fps,
            codec='libx264',
            pixelformat='yuv420p',
            macro_block_size=None,
            ffmpeg_params=['-crf', '10'],
        )
