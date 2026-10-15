import torch, warnings, types
import torch.nn.functional as F
from einops import repeat, reduce
from typing import Union
from dataclasses import dataclass
from modelscope import snapshot_download
import numpy as np
from PIL import Image
from typing import Optional
from torch import nn

from models.model_manager import ModelManager
from models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from models.scheduler import FlowMatchScheduler
from models.vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear, WanAutoCastLayerNorm
# import pyiqa

from trainer.temp_loss.losses import warp_loss
from trainer.temp_loss.raft import RAFT_SR


class BasePipeline(torch.nn.Module):

    def __init__(
            self,
            device="cuda", torch_dtype=torch.float16,
            time_division_factor=None, time_division_remainder=None,
    ):
        super().__init__()
        # The device and torch_dtype is used for the storage of intermediate variables, not models.
        self.device = device
        self.torch_dtype = torch_dtype
        # The following parameters are used for shape check.
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.vram_management_enabled = False

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self

    def preprocess_image(self, image, torch_dtype=None, device=None, pattern="B C H W", min_value=-1, max_value=1):
        # Transform a PIL.Image to torch.Tensor
        image = torch.Tensor(np.array(image, dtype=np.float32))
        image = image.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        image = image * ((max_value - min_value) / 255) + min_value
        image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))
        return image

    def preprocess_video(self, video, torch_dtype=None, device=None, pattern="B C T H W", min_value=-1, max_value=1):
        # Transform a list of PIL.Image to torch.Tensor
        video = [self.preprocess_image(image, torch_dtype=torch_dtype, device=device, min_value=min_value,
                                       max_value=max_value) for image in video]
        video = torch.stack(video, dim=pattern.index("T") // 2)
        return video

    def vae_output_to_image(self, vae_output, pattern="B C H W", min_value=-1, max_value=1):
        # Transform a torch.Tensor to PIL.Image
        if pattern != "H W C":
            vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
        image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(0, 255)
        image = image.to(device="cpu", dtype=torch.uint8)
        image = Image.fromarray(image.numpy())
        return image

    def vae_output_to_video(self, vae_output, pattern="B C T H W", min_value=-1, max_value=1):
        # Transform a torch.Tensor to list of PIL.Image
        if pattern != "T H W C":
            vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
        video = [self.vae_output_to_image(image, pattern="H W C", min_value=min_value, max_value=max_value) for image in
                 vae_output]
        return video

    def load_models_to_device(self, model_names=[]):
        if self.vram_management_enabled:
            # offload models
            for name, model in self.named_children():
                if name not in model_names:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                module.offload()
                    else:
                        model.cpu()
            torch.cuda.empty_cache()
            # onload models
            for name, model in self.named_children():
                if name in model_names:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        for module in model.modules():
                            if hasattr(module, "onload"):
                                module.onload()
                    else:
                        model.to(self.device)

    def enable_cpu_offload(self):
        warnings.warn("`enable_cpu_offload` will be deprecated. Please use `enable_vram_management`.")
        self.vram_management_enabled = True

    def get_free_vram(self):
        total_memory = torch.cuda.get_device_properties(self.device).total_memory
        allocated_memory = torch.cuda.device_memory_used(self.device)
        return (total_memory - allocated_memory) / (1024 ** 3)

    def freeze_except(self, model_names):
        for module in self.children():
            module.eval()
        for module in model_names:
            recursive_getattr(self, module).train()
        for name, p in self.named_parameters():
            if any(keyword in name for keyword in model_names):
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)


def recursive_getattr(obj, attr_path):
    for attr in attr_path.split('.'):
        obj = getattr(obj, attr)
    return obj


@dataclass
class ModelConfig:
    path: Union[str, list[str]] = None
    model_id: str = None
    origin_file_pattern: Union[str, list[str]] = None
    download_resource: str = "ModelScope"
    offload_device: Optional[Union[str, torch.device]] = None
    offload_dtype: Optional[torch.dtype] = None


class Projector(torch.nn.Module):
    def __init__(self, first_dim, other_dim):
        super().__init__()
        self.conv1 = nn.Conv3d(
                        in_channels=first_dim,
                        out_channels=16,
                        kernel_size=1,        # keeps the spatio-temporal resolution
                        stride=1,
                        padding=0
                    )
        self.conv2 = nn.Conv3d(
            in_channels=other_dim,
            out_channels=16,
            kernel_size=1,  # keeps the spatio-temporal resolution
            stride=1,
            padding=0
        )

    def encoder_first_frame(self, x):
        return self.conv1(x)

    def encoder_other_frames(self, x):
        return self.conv2(x)

    def forward(self, first_frame, other_frames):
        first_frame = self.encoder_first_frame(first_frame)  # B, 16, 1, H//8, W//8
        other_frames = self.encoder_other_frames(other_frames)  # B, 16, (T-1)//4, H//8, W//8
        lq_latents = torch.cat([first_frame, other_frames], dim=2)
        return lq_latents


class WanVideoPipeline(BasePipeline):

    def __init__(self, args, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.args = args
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.flownet = None
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_InputVideoEmbedder(),
            # WanVideoUnit_PromptEmbedder(),
        ]
        self.projector = Projector(12, 48).to(dtype=torch_dtype, device=device)
        self.model_fn = model_fn_wan_video
        # if args.video_loss:
        #     self.loss_fn = pyiqa.create_metric(args.aux_loss, device=device, as_loss=True)
        if args.temp_loss == "warp_loss":
            self.flownet = RAFT_SR(load_path=args.flow_path)
            self.flownet.eval()

    def training_loss_video(self, **inputs):
        # the single denoising step of the one-step VSR model
        timestep_id = torch.full(
            (self.args.batch_size,),
            fill_value=self.args.sr_step,
            dtype=torch.long,
        )
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)

        # targets
        training_target = inputs["hq_video"]        # b,c,t,h,w -1,1
        latents_target = inputs["hq_latents"]
        keys = ['dit', 'lq_latents', 'timestep', 'context']
        tiled = inputs["tiled"]
        inputs = {k: v for k, v in inputs.items() if k in keys}
        inputs['context'] = inputs['context'].to(dtype=self.torch_dtype, device=self.device)
        u_pred = self.model_fn(**inputs, timestep=timestep)
        print("u_pred requires_grad:", u_pred.requires_grad)
        while timestep.ndim < u_pred.ndim:
            timestep = timestep.unsqueeze(-1)
        latents_pred = self.scheduler.step(u_pred, timestep, inputs["lq_latents"], to_final=True)

        loss_dict = {}

        if self.args.latent_loss:
            # latent_loss = torch.nn.functional.mse_loss(latents_pred.float(), latents_target.float())
            latent_loss = torch.nn.functional.mse_loss(latents_pred, latents_target)   
            loss_dict['latent_loss'] = latent_loss # .item()
        else:
            latent_loss = 0

        vae_output = self.vae.decode(latents_pred, device=self.device, tiled=tiled, tile_size=None, tile_stride=None)  # -1, 1

        min_value, max_value = -1, 1
        video_pred = ((vae_output - min_value)  / (max_value - min_value)).clip(0, 1)       # b c t h w
        training_target = ((training_target - min_value)  / (max_value - min_value)).clip(0, 1)     # b c t h w

        if self.args.temp_loss is not None:
            if self.args.temp_loss == "frame_diff_loss":
                # video_generate, hq_videos: [B, C, F, H, W]
                diff_gen = video_pred[:, :, 1:, :, :] - video_pred[:, :, :-1, :, :]  # [B, C, F-1, H, W]
                diff_gt  = training_target[:, :, 1:, :, :] - training_target[:, :, :-1, :, :]       # [B, C, F-1, H, W]
                temp_loss = torch.nn.functional.l1_loss(diff_gen.float(), diff_gt.float(), reduction="mean") * self.args.temp_weight
            elif self.args.temp_loss == "warp_loss":
                temp_loss = warp_loss((video_pred * 2 - 1).permute(0, 2, 1, 3, 4).flatten(0, 1), #.float(),
                                      (training_target * 2 - 1).permute(0, 2, 1, 3, 4).flatten(0, 1), #.float(),
                                      t=video_pred.shape[2],
                                      flownet=self.flownet) * self.args.temp_weight
            loss_dict[self.args.temp_loss] = temp_loss# .item()
            print("temp_loss requires_grad:", temp_loss.requires_grad)
        else:
            temp_loss = 0

        video_pred = video_pred.permute(0, 2, 1, 3, 4).flatten(0, 1)
        training_target = training_target.permute(0, 2, 1, 3, 4).flatten(0, 1)
        # mse_loss = torch.nn.functional.mse_loss(video_pred.float(), training_target.float(), reduction="mean")
        # aux_loss = self.loss_fn(video_pred.float(), training_target.float())
        mse_loss = torch.nn.functional.mse_loss(video_pred, training_target, reduction="mean")
        # aux_loss = self.loss_fn(video_pred, training_target)
        print("mse_loss requires_grad:", mse_loss.requires_grad)
        # print("aux_loss requires_grad:", aux_loss.requires_grad)

        loss = mse_loss # + aux_loss # + temp_loss + latent_loss
        # loss_dict.update({'mse_loss': mse_loss.item(), self.args.aux_loss + '_loss': aux_loss.item()})
        # loss_dict.update({'mse_loss': mse_loss, self.args.aux_loss + '_loss': aux_loss})

        return loss, loss_dict

    def validate(self, **inputs):
        with torch.no_grad():
            timestep_id = torch.full(
                (1,),
                fill_value=self.args.sr_step,
                dtype=torch.long,
            )
            timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)
            keys = ['dit', 'lq_latents', 'timestep', 'context']
            tiled = inputs["tiled"]     # False
            tile_size = inputs["tile_size"]
            tile_stride = inputs["tile_stride"]
            inputs["context"] = inputs["context"].to(dtype = self.torch_dtype, device = self.device)
            inputs = {k: v for k, v in inputs.items() if k in keys}
            u_pred = self.model_fn(**inputs, timestep=timestep)
            while timestep.ndim < u_pred.ndim:
                timestep = timestep.unsqueeze(-1)
            latents_pred = self.scheduler.step(u_pred, timestep, inputs["lq_latents"], to_final=True)
            video_pred = self.vae.decode(latents_pred, device=self.device, tiled=tiled,
                                         tile_size=tile_size, tile_stride=tile_stride)   # here might change frames
            video_pred = self.vae_output_to_video(video_pred)

        return video_pred

    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_free_vram()
            vram_limit = vram_limit - vram_buffer
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )

    def initialize_usp(self):
        import torch.distributed as dist
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        dist.init_process_group(backend="nccl", init_method="env://")
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        torch.cuda.set_device(dist.get_rank())

    def enable_usp(self):
        from xfuser.core.distributed import get_sequence_parallel_world_size
        from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True

    @staticmethod
    def from_pretrained(
            args,
            torch_dtype: torch.dtype = torch.bfloat16,
            device: Union[str, torch.device] = "cuda",
            model_configs: list[ModelConfig] = [],
            use_usp=False,
    ):
        model_manager = ModelManager()
        for model_config in model_configs:
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )

        # Initialize pipeline
        pipe = WanVideoPipeline(args=args, device=device, torch_dtype=torch_dtype)
        if use_usp: pipe.initialize_usp()
        pipe.dit = model_manager.fetch_model("wan_video_dit")
        pipe.vae = model_manager.fetch_model("wan_video_vae")

        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()
        return pipe


class PipelineUnit:
    def __init__(
            self,
            seperate_cfg: bool = False,
            take_over: bool = False,
            input_params: tuple[str] = None,
            input_params_posi: dict[str, str] = None,
            input_params_nega: dict[str, str] = None,
            onload_model_names: tuple[str] = None
    ):
        self.seperate_cfg = seperate_cfg
        self.take_over = take_over
        self.input_params = input_params
        self.input_params_posi = input_params_posi
        self.input_params_nega = input_params_nega
        self.onload_model_names = onload_model_names

    def process(self, pipe: WanVideoPipeline, inputs: dict, positive=True, **kwargs) -> dict:
        raise NotImplementedError("`process` is not implemented.")


class PipelineUnitRunner:
    def __init__(self):
        pass

    def __call__(self, unit: PipelineUnit, pipe: WanVideoPipeline, inputs_shared: dict):
        # print(f"inputs shared keys before processor_inputs: {inputs_shared.keys()}")
        # print(f"unit.input_params is: {unit.input_params}")
        processor_inputs = {name: inputs_shared.get(name) for name in unit.input_params}
        processor_outputs = unit.process(pipe, **processor_inputs)
        inputs_shared.update(processor_outputs)
        return inputs_shared


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            # input_params=("tiled", "tile_size", "tile_stride", "lq_video", "hq_video", "hq_latents", "upscale"),
            input_params=("tiled", "tile_size", "tile_stride", "lq_video", "hq_video", "upscale"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, tiled, tile_size, tile_stride, lq_video, hq_video=None, upscale=4):
        if pipe.scheduler.training:
            assert hq_video is not None
        pipe.load_models_to_device(["vae"])
        lq_frames = []
        for video in lq_video:
            lq_frames.append(pipe.preprocess_video(video))
        # b c t h w
        lq_frames = torch.cat(lq_frames, dim=0)
        b, c, t, h, w = lq_frames.shape
        lq_frames = lq_frames.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        lq_frames = F.interpolate(lq_frames, size=(h * upscale, w * upscale), mode='bilinear', align_corners=False)
        lq_frames = lq_frames.reshape(b, t, c, h * upscale, w * upscale).permute(0, 2, 1, 3, 4)

        _, _, t, h, w = lq_frames.shape
        h_ = h // 16 * 16
        w_ = w // 16 * 16
        lq_frames = lq_frames[:, :, :, :h_, :w_]
        t_new = ((max(t - 1, 0) + 7) // 8) * 8 + 1
        if t_new > t:
            pad = lq_frames[:, :, -1:].expand(-1, -1, t_new - t, -1, -1)
            lq_frames = torch.cat([lq_frames, pad], dim=2)

        lq_latents = pipe.vae.encode(lq_frames, device=pipe.device, tiled=tiled, tile_size=tile_size,
                                     tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        if pipe.scheduler.training:
            hq_frames = []
            for video in hq_video:
                hq_frames.append(pipe.preprocess_video(video))
            hq_frames = torch.cat(hq_frames, dim=0)
            hq_latents = pipe.vae.encode(hq_frames, device=pipe.device, tiled=tiled, tile_size=tile_size,
                                         tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            return {"lq_latents": lq_latents, "hq_latents": hq_latents, "hq_video": hq_frames}
        else:
            return {"lq_latents": lq_latents}


def model_fn_wan_video(
        dit: WanModel,
        lq_latents: torch.Tensor = None,
        timestep: torch.Tensor = None,
        context: torch.Tensor = None,
):
    # x will change shape if size does not divide 16 or frames do not divide 8
    batch_size = len(lq_latents)
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = lq_latents
    assert context.shape[0] == batch_size
    assert timestep.shape[0] == context.shape[0]
    # Add camera control
    # x, (f, h, w), original_size, pad_info = dit.padding_patchify(x)
    x, (f, h, w) = dit.patchify(x)
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    # freqs = freqs.repeat(1, batch_size, 1)

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)

        return custom_forward

    for block in dit.blocks:
        x = torch.utils.checkpoint.checkpoint(
            create_custom_forward(block),
            x, context, t_mod, freqs,
            use_reentrant=False,
        )

    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    return x


def video_unshuffle(x, spatial_factor=2, temporal_factor=4):
    """
    x: Tensor of shape [B, C, 8n+1, H, W]
    """
    B, C, T, H, W = x.shape
    assert T % 8 == 1, "T must be 8N+1"
    assert H % spatial_factor == 0 and W % spatial_factor == 0, "H/W must be divisible by 16"

    def temporal_spatial_unshuffle(x, spatial_factor, temporal_factor):
        """
        Rearranges [B, C, T, H, W] into [B, C * temporal_factor * spatial_factor^2, T//temporal_factor, H//spatial_factor, W//spatial_factor]
        """
        B, C, T, H, W = x.shape
        assert T % temporal_factor == 0
        assert H % spatial_factor == 0 and W % spatial_factor == 0

        x = x.view(B, C, T // temporal_factor, temporal_factor, H // spatial_factor, spatial_factor,
                   W // spatial_factor, spatial_factor)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)  # B, C, t_fact, sH, sW, T//t, H//s, W//s
        x = x.reshape(B, C * temporal_factor * spatial_factor ** 2, T // temporal_factor, H // spatial_factor,
                      W // spatial_factor)
        return x

    # extract first frame
    first_frame = x[:, :, [0]]  # shape: [B, C, 1, H, W]
    first_frame = temporal_spatial_unshuffle(first_frame, spatial_factor, 1)

    # extract other frames
    other_frames = x[:, :, 1:]  # shape: [B, C, 8n+1, H, W]
    other_frames = temporal_spatial_unshuffle(other_frames, spatial_factor, temporal_factor)

    # [B, C*temporal_factor*spatio_factor**2, 1, H//spatio_factor, W//spatio_factor], [B, C*temporal_factor*spatio_factor**2, (T-1)//4, H//spatio_factor, W//spatio_factor]
    return first_frame, other_frames
