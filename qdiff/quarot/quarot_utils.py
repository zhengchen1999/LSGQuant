import os

import torch
from torch import Tensor

_HADAMARD_MAT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'hadamard_utils', 'hadamard_mat.pth'
)
_HADAMARD_CACHE = {}


def _load_hadamard_table(key: str) -> Tensor:
    path = os.environ.get('HADAMARD_MAT_PATH', _HADAMARD_MAT_PATH)
    if path not in _HADAMARD_CACHE:
        _HADAMARD_CACHE[path] = torch.load(path, weights_only=True)
    table = _HADAMARD_CACHE[path]
    if key not in table:
        raise KeyError(
            f"'{key}' not found in {path}; "
            "regenerate it with qdiff/quarot/hadamard_utils/convert_hardamard_mat.py"
        )
    return table[key]


def get_had12() -> Tensor:
    return _load_hadamard_table('had.12')


def get_had140() -> Tensor:
    return _load_hadamard_table('had.140')


def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)


def get_hadK(n, transpose=False):
    """Return the Hadamard block ``hadK`` and its size ``K`` for dimension ``n``."""
    hadK, K = None, None
    if n % 140 == 0 and is_pow2(n // 140):
        K = 140
        hadK = get_had140().T if transpose else get_had140()
    elif n % 12 == 0 and is_pow2(n // 12):
        K = 12
        hadK = get_had12().T if transpose else get_had12()
    elif is_pow2(n):
        K = 1
    else:
        raise ValueError(
            f"no Hadamard decomposition for size {n}: supported sizes are the "
            "WAN2.1 DiT linear dimensions 12 * 2^m (1536 = 12 * 128) and "
            "140 * 2^m (8960 = 140 * 64), plus powers of two"
        )
    return hadK, K


def matmul_hadU(X, transpose=False):
    n = X.shape[-1]
    hadK, K = get_hadK(n, transpose)
    input = X.clone().view(-1, n, 1)
    output = input.clone()
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)
    del output

    if K > 1:
        # Do not explicitly repeat - OOM
        # input = torch.bmm(
        #     hadK.repeat(len(input), 1, 1).to(input.device).to(input.dtype), input)
        # Use bcast instead
        input = hadK.view(1, K, K).to(input) @ input

    return input.view(X.shape) / torch.tensor(n).sqrt()


def matmul_hadUt(X):
    return matmul_hadU(X, transpose=True)


def random_hadamard_sign(size, dtype=torch.bfloat16):
    """Random +/-1 sign vector of the randomized Hadamard rotation."""
    return torch.randint(low=0, high=2, size=(size,)).to(dtype) * 2 - 1


def hadamard_from_sign(sign):
    """Rebuild ``H = matmul_hadU(diag(sign))`` from its +/-1 sign vector."""
    return matmul_hadU(torch.diag(sign))


def random_hadamard_matrix(size, device):
    # See https://cornell-relaxml.github.io/quip-sharp/,
    # Section "Randomized Hadamard Transformation"
    return hadamard_from_sign(random_hadamard_sign(size)).to(device)
