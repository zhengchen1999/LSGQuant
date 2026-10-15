import torch
import torch.nn.functional as F


def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid


def bilinear_sample(img,
                    sample_coords,
                    mode='bilinear',
                    padding_mode='zeros',
                    return_mask=False):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img,
                        grid,
                        mode=mode,
                        padding_mode=padding_mode,
                        align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (
            y_grid <= 1)  # [B, H, W]

        return img, mask

    return img


def feature_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True, return_mask=False):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    if not return_mask:
        return output

    mask = torch.ones_like(x)
    mask = F.grid_sample(mask, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)
    mask[mask < 0.9999] = 0
    mask[mask > 0.0000] = 1
    return output, mask


def flow_warp(feature,
              flow,
              mask=False,
              mode='bilinear',
              padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]

    return bilinear_sample(feature,
                           grid,
                           mode=mode,
                           padding_mode=padding_mode,
                           return_mask=mask)


def forward_backward_consistency_check(fwd_flow,
                                       bwd_flow,
                                       alpha=0.01,
                                       beta=0.5):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    # alpha and beta values are following UnFlow
    # (https://arxiv.org/abs/1711.07837)
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)  # [B, H, W]

    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow)  # [B, 2, H, W]
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow)  # [B, 2, H, W]

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)  # [B, H, W]
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd > threshold).float()  # [B, H, W]
    bwd_occ = (diff_bwd > threshold).float()

    return fwd_occ, bwd_occ


def compute_flow(imgs, flownet):
    """Compute optical flow using flow estimation network for warping.

    Args:
        imgs (tensor): Input images with shape (n, t, c, h, w)

    Return:
        tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
            flows used for forward-time propagation (current to previous).
            'flows_backward' corresponds to the flows used for
            backward-time propagation (current to next).
    """

    n, t, c, h, w = imgs.size()
    imgs_1 = imgs[:, :-1, :, :, :].reshape(-1, c, h, w)  # former
    imgs_2 = imgs[:, 1:, :, :, :].reshape(-1, c, h, w)  # latter

    # forward flow, for backward propagation
    fwd_flows = flownet(imgs_1, imgs_2).view(n, t - 1, 2, h, w)
    # backward flow, for forward propagation
    bwd_flows = flownet(imgs_2, imgs_1).view(n, t - 1, 2, h, w)

    return fwd_flows, bwd_flows
