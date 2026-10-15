import json
import os

import torch
from safetensors.torch import load_file

from trainer.utils import DiffusionTrainingModule
from wan_sr import ModelConfig, WanVideoPipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        args,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
    ):
        super().__init__()
        # Load models
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1]) for i in model_id_with_origin_paths]
        self.pipe = WanVideoPipeline.from_pretrained(args=args, torch_dtype=torch.bfloat16, device="cuda", model_configs=model_configs)

        # Reset scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)

        if args.ckpt_path is not None:
            if args.ckpt_path.endswith(".safetensors"):
                state_dict = load_file(args.ckpt_path)
                self.pipe.dit.load_state_dict(state_dict)
            else:
                def recursive_getattr(obj, attr_path):
                    for attr in attr_path.split('.'):
                        obj = getattr(obj, attr)
                    return obj

                state_dict = torch.load(args.ckpt_path)
                for name in state_dict.keys():
                    module = recursive_getattr(self.pipe, name)
                    module.load_state_dict(state_dict[name])

        # Freeze untrainable models
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.upscale = args.upscale
        self.tile_size = args.tile_size
        self.tile_stride = args.tile_stride
        self.args = args

    def forward_preprocess(self, data):
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "lq_video": data["lq_video"],
            "hq_video": data["hq_video"],
            # None for validation and online training.
            "hq_latents": data["hq_latents"] if "hq_latents" in data else None,
            "context": data["prompt_embed"],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "tiled": False,
            "tile_size": self.tile_size,
            "tile_stride": self.tile_stride,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "upscale": self.upscale,
        }
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared = self.pipe.unit_runner(unit, self.pipe, inputs_shared)
        return {**inputs_shared}

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.forward_preprocess(data)
        if self.args.video_loss:
            loss = self.pipe.training_loss_video(**{'dit': getattr(self.pipe, 'dit')}, **inputs)
        else:
            loss = self.pipe.training_loss(**{'dit': getattr(self.pipe, 'dit')}, **inputs)
        return loss

    def inference(self, data, inputs=None):
        self.pipe.scheduler.training = False
        if inputs is None:
            inputs = self.forward_preprocess(data)
        video = self.pipe.validate(**{'dit': getattr(self.pipe, 'dit')}, **inputs)
        self.pipe.scheduler.training = True
        return video
