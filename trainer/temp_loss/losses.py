from einops import rearrange
import kornia as K
import torch
import torch.nn.functional as F

from .utils import forward_backward_consistency_check, compute_flow, feature_warp


def swc_loss(hr_frms, gt_frms, t, flownet, w=3):
    bt, _, _, _ = hr_frms.shape
    # compute weight
    edge_maps = K.filters.sobel(gt_frms).detach()
    weight_maps = 1 + w * edge_maps
    # (b t) c h w -> b t c h w
    hr_frms = rearrange(hr_frms, '(b t) c h w -> b t c h w', t=t)
    gt_frms = rearrange(gt_frms, '(b t) c h w -> b t c h w', t=t)
    weight_maps = rearrange(weight_maps, '(b t) c h w -> b t c h w', t=t)
    # compute flows
    # fwd_flows: forward flow, for backward propagation
    # bwd_flows: backward flow, for forward propagation
    fwd_flows, bwd_flows = compute_flow(gt_frms, flownet)       # b t-1 2 h w

    fwd_occ_list, bwd_occ_list = list(), list()
    for i in range(t - 1):
        fwd_flow, bwd_flow = fwd_flows[:, i, :, :, :], bwd_flows[:, i, :, :, :]
        fwd_occ, bwd_occ = forward_backward_consistency_check(fwd_flow, bwd_flow, alpha=0.01, beta=0.5)
        fwd_occ_list.append(fwd_occ.unsqueeze_(1))
        bwd_occ_list.append(bwd_occ.unsqueeze_(1))
    fwd_occs = torch.stack(fwd_occ_list, dim=1)
    bwd_occs = torch.stack(bwd_occ_list, dim=1)

    loss_b = 0
    curr_warp = torch.zeros_like(hr_frms[:, -1, :, :, :])
    # backward propagation
    for i in range(t - 1, -1, -1):
        curr = hr_frms[:, i, :, :, :]
        if i < t - 1:  # no warping required for the last timestep
            flow = fwd_flows[:, i, :, :, :]
            curr_warp = feature_warp(curr, flow.permute(0, 2, 3, 1), interp_mode='bilinear')
            loss_b += F.l1_loss(
                weight_maps[:, i, :, :, :] * (1 - fwd_occs[:, i, :, :, :]) * prev,
                weight_maps[:, i, :, :, :] * (1 - fwd_occs[:, i, :, :, :]) * curr
            )
        prev = curr_warp
    loss_f = 0
    curr_warp = torch.zeros_like(hr_frms[:, 0, :, :, :])
    # forward propagation
    for i in range(0, t):
        curr = hr_frms[:, i, :, :, :]
        if i > 0:  # no warping required for the first timestep
            flow = bwd_flows[:, i - 1, :, :, :]
            curr_warp = feature_warp(curr, flow.permute(0, 2, 3, 1), interp_mode='bilinear')
            loss_f += F.l1_loss(
                weight_maps[:, i, :, :, :] * (1 - bwd_occs[:, i - 1, :, :, :]) * prev,
                weight_maps[:, i, :, :, :] * (1 - bwd_occs[:, i - 1, :, :, :]) * curr
            )
        prev = curr_warp
    loss = loss_b + loss_f
    return loss


def warp_loss(hr_frms, gt_frms, t, flownet):
    # shape: (b t) c h w
    # value: [-1, 1]

    bt, _, _, _ = hr_frms.shape

    # (b t) c h w -> b t c h w
    hr_frms = rearrange(hr_frms, '(b t) c h w -> b t c h w', t=t)
    gt_frms = rearrange(gt_frms, '(b t) c h w -> b t c h w', t=t)

    # compute flows
    # fwd_flows: forward flow, for backward propagation
    # bwd_flows: backward flow, for forward propagation
    with torch.no_grad():
        fwd_flows, bwd_flows = compute_flow(gt_frms, flownet)       # b t-1 2 h w

        fwd_occ_list, bwd_occ_list = list(), list()
        for i in range(t - 1):
            fwd_flow, bwd_flow = fwd_flows[:, i, :, :, :], bwd_flows[:, i, :, :, :]
            fwd_occ, bwd_occ = forward_backward_consistency_check(fwd_flow, bwd_flow, alpha=0.01, beta=0.5)
            fwd_occ_list.append(fwd_occ.unsqueeze_(1))
            bwd_occ_list.append(bwd_occ.unsqueeze_(1))
        fwd_occs = torch.stack(fwd_occ_list, dim=1)
        bwd_occs = torch.stack(bwd_occ_list, dim=1)

    loss_b = 0
    curr_warp = torch.zeros_like(hr_frms[:, -1, :, :, :])
    # backward propagation
    for i in range(t - 1, -1, -1):
        curr = hr_frms[:, i, :, :, :]
        if i < t - 1:  # no warping required for the last timestep
            flow = fwd_flows[:, i, :, :, :]
            curr_warp = feature_warp(curr, flow.permute(0, 2, 3, 1), interp_mode='bilinear')
            loss_b += F.l1_loss(
                (1 - fwd_occs[:, i, :, :, :]) * prev,
                (1 - fwd_occs[:, i, :, :, :]) * curr
            )
        prev = curr_warp

    loss_f = 0
    curr_warp = torch.zeros_like(hr_frms[:, 0, :, :, :])
    # forward propagation
    for i in range(0, t):
        curr = hr_frms[:, i, :, :, :]
        if i > 0:  # no warping required for the first timestep
            flow = bwd_flows[:, i - 1, :, :, :]
            curr_warp = feature_warp(curr, flow.permute(0, 2, 3, 1), interp_mode='bilinear')
            loss_f += F.l1_loss(
                (1 - bwd_occs[:, i - 1, :, :, :]) * prev,
                (1 - bwd_occs[:, i - 1, :, :, :]) * curr
            )
        prev = curr_warp
    loss = (loss_b + loss_f) / t
    return loss
