import torch
import torch.nn.functional as F
from tqdm import tqdm

from qdiff.base.base_quantizer import DynamicQuantizer
from qdiff.base.draq_quantizer import DRAQQuantizer
from qdiff.base.quant_layer import QuantizedLinear
from qdiff.quarot.quarot_utils import hadamard_from_sign, random_hadamard_sign


class LSGQuantizedLinear(QuantizedLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        device: None,
        quant_config: dict,
        fp_module: torch.nn.Linear,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, quant_config, fp_module)

        self.rotation_matrix = None   # init so it can be filled by quant_params
        self.had_sign = None          # compact description of the random Hadamard matrix
        self.L1 = None
        self.L2 = None
        self.device = device

        # Activation quantizer:
        #   'draq'    -> DRAQ; the released setting.
        #   'dynamic' -> per-token symmetric quantizer.
        lsg_config = quant_config.get("lsgquant", None)
        act_quant = "draq" if lsg_config is None else str(lsg_config.get("act_quant", "draq")).lower()
        if quant_config.get("act", None) is not None:
            if act_quant == "draq":
                self.a_quantizer = DRAQQuantizer(quant_config["act"])
            elif act_quant == "dynamic":
                self.a_quantizer = DynamicQuantizer(quant_config["act"])
            else:
                raise ValueError(
                    f"unknown activation quantizer '{act_quant}', expected 'draq' or 'dynamic'"
                )

    def get_rotation_matrix(self, device="cuda"):
        """Sample the random Hadamard rotation ``H``.

        ``H`` is fully described by its +/-1 sign vector, so only ``had_sign`` is
        stored in the quantization parameters (a dense ``H`` costs up to 160 MB
        per layer). ``qdiff.base.quant_model.load_quant_param_dict_`` rebuilds
        ``H`` from it bit-exactly at load time.
        """
        self.had_sign = random_hadamard_sign(self.in_features)
        self.rotation_matrix = hadamard_from_sign(self.had_sign).to(device)
        print(self.rotation_matrix.shape, self.rotation_matrix.dtype)

    def reload_R_quantized(self):
        W_hat = torch.matmul(self.fp_module.weight.data, self.rotation_matrix)
        R = W_hat - self.L1 @ self.L2
        self.weight.data = self.w_quantizer(R)

    def update_quantized_weight_rotated_and_svd(self, rank=16, max_iter=10):
        """QAO: Quantization-Aware Alternating Optimization.

        Returns:
            (best_cnt, best_error): iteration index and residual of the
            retained pair.
        """
        self.w_quantizer.init_done = False   # unset the init done to overwrite quant_params

        W = self.fp_module.weight.data
        block_dtype = W.dtype
        W_hat = torch.matmul(W, self.rotation_matrix).float() # (C_out, C_in)

        # best (L1, L2) of the QAO iterations so far
        best_error = float('inf')
        best_L1, best_L2 = None, None
        best_cnt, duplicate_cnt = 0, 0

        # initial SVD of the rotated weight
        U, S, Vh = torch.linalg.svd(W_hat, full_matrices=False)
        L1 = U[:, :rank] @ torch.diag(S[:rank])
        L2 = Vh[:rank, :]

        # residual R = W_hat - L1 L2, quantized by the weight quantizer
        R = W_hat - L1 @ L2
        R_quantized = self.w_quantizer(R)
        self.w_quantizer.init_done = True

        progress_bar = tqdm(range(max_iter))

        for cnt in progress_bar:

            # approximation error of the current pair
            current_error = torch.norm((R - R_quantized))

            # keep the pair if it improves the residual
            tol = best_error - current_error
            if tol > 0:
                best_error = current_error
                best_L1, best_L2 = L1, L2
                best_cnt = cnt
                duplicate_cnt = 0
            else:
                duplicate_cnt += 1

            progress_bar.set_description(
                f"Err: {current_error:.4f} | Best: {best_error:.4f} | Dup: {duplicate_cnt}"
            )
            if duplicate_cnt > 10:
                break

            # QAO: re-fit the low-rank branch on the residual left by the
            # quantized branch, W_hat - R_quantized
            target_matrix = W_hat - R_quantized
            U, S, Vh = torch.linalg.svd(target_matrix, full_matrices=False)

            # low-rank branch of the next iteration
            L1 = U[:, :rank] @ torch.diag(S[:rank])
            L2 = Vh[:rank, :]

            R = W_hat - L1 @ L2
            R_quantized = self.w_quantizer(R)

        self.L1 = best_L1.clone().to(block_dtype)
        self.L2 = best_L2.clone().to(block_dtype)
        self.w_quantizer.delta = self.w_quantizer.delta.to(block_dtype)
        self.w_quantizer.zero_point = self.w_quantizer.zero_point.to(block_dtype)
        return best_cnt, best_error

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        input shape: [B,N_token,C]
        """
        if not self.quant_mode:  # use the FP
            return self.fp_module(x, *args, **kwargs)
        else:
            B, N_token, C = x.shape
            # rotation, then the full-precision low-rank branch
            x = torch.matmul(x, self.rotation_matrix.to(self.device)).to(dtype=x.dtype)
            y = F.linear(x, self.L1 @ self.L2)

            # low-bit branch: DRAQ performs the channel-wise range adaptation itself
            x = self.a_quantizer(x.reshape([B * N_token, -1]))

            y += F.linear(x.view([B, N_token, C]), self.weight, self.bias, *args, **kwargs)
            return y


if __name__ == '__main__':
    pass
