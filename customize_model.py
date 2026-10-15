from torch import nn
from typing import Tuple
import torch
from typing import Optional

from models.wan_video_dit import WanModel, sinusoidal_embedding_1d

from qdiff.utils import apply_func_to_submodules
from qdiff.base.quant_model import (
    save_quant_param_dict_, 
    load_quant_param_dict_,
    quant_layer_refactor_, 
    set_init_done_, 
    BaseQuantizer,
)

class CustomizeWanModel(WanModel):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
    ):
        super().__init__(
            dim=dim, 
            in_dim=in_dim, 
            ffn_dim=ffn_dim, 
            out_dim=out_dim, 
            text_dim=text_dim, 
            freq_dim=freq_dim, 
            eps=eps, 
            patch_size=patch_size, 
            num_heads=num_heads, 
            num_layers=num_layers, 
            has_image_input=has_image_input, 
            has_image_pos_emb=has_image_pos_emb, 
            has_ref_conv=has_ref_conv
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.has_image_input = has_image_input
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.eps = eps

    # viditq + smoothquant + quarot 
    def convert_quant(self, quant_config, device = "cuda"):
        self.quant_config = quant_config
        self.quant_param_dict = {}
        self.quant_layer_refactor(device=device)

    def quant_layer_refactor(self, device):
        print("start transformer quant layer refactor")
        apply_func_to_submodules(
            self, 
            class_type=nn.Linear,
            function=quant_layer_refactor_,
            name=None,
            parent_module=None,
            quant_config=self.quant_config,
            full_name=None,
            remain_fp_regex=self.quant_config.remain_fp_regex,
            device=device
        )
        print("complete transformer quant layer refactor")

    def set_init_done(self):
        apply_func_to_submodules(
            self,
            class_type=BaseQuantizer,
            function=set_init_done_,
        )

    def save_quant_param_dict(self):
        apply_func_to_submodules(
            self,
            class_type=BaseQuantizer,
            function=save_quant_param_dict_,
            full_name=None,
            parent_module=None,
            model=self
        )

    def load_quant_param_dict(self, quant_param_dict, device="cuda"):
        apply_func_to_submodules(
            self,
            class_type=BaseQuantizer,
            function=load_quant_param_dict_,
            full_name=None,
            parent_module=None,
            quant_param_dict=quant_param_dict,
            model=self,
            device=device
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ):
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        x, (f, h, w) = self.patchify(x)

        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs, 
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x
