import torch, os, json
from safetensors.torch import load_file
from wan_sr import WanVideoPipeline, ModelConfig
from datasets.datasets import RealSRValDataset
from trainer.trainer import launch_validation_task_sr
from trainer.parser import wan_parser

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class WanValidationModule(torch.nn.Module):
    def __init__(
        self,
        args,
        model_paths=None, model_id_with_origin_paths=None,
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

        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=False)
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

        self.pipe.eval()
        self.args = args

        self.extra_inputs = []

    def forward_preprocess(self, data):
        h, w, n = data["lq_video"][0][0].size[1], data["lq_video"][0][0].size[0], len(data["lq_video"][0])
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "lq_video": data["lq_video"],
            "hq_video": data["hq_video"],
            "num_frames": n,
            "context": data["prompt_embed"],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": self.args.tiled,
            "tile_size": self.args.tile_size,
            "tile_stride": self.args.tile_stride,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": True,
            "use_gradient_checkpointing_offload": False,
            "upscale": self.args.upscale
        }
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared = self.pipe.unit_runner(unit, self.pipe, inputs_shared)
        return {**inputs_shared}

    def forward(self, data, inputs=None):
        if inputs is None: 
            inputs = self.forward_preprocess(data)
        video = self.pipe.validate(**{'dit': getattr(self.pipe, 'dit')}, **inputs)[:inputs["num_frames"]]
        return video

def compute_model_params(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    val_dataset = RealSRValDataset(
        lq_root=os.path.join(args.val_path, 'LQ-Video'),
        hq_root=os.path.join(args.val_path, 'GT-Video'),
        prompt_cache=args.prompt_cache,
    )

    model = WanValidationModule(
        args=args,
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
    )

    launch_validation_task_sr(model, val_dataset, output_path=args.output_path, args=args)
