"""
A set of geometry tools for PyTorch tensors and sometimes NumPy arrays.
"""

import torch
import numpy as np


def to_homogeneous(points):
    """Convert N-dimensional points to homogeneous coordinates.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N).
    Returns:
        A torch.Tensor or numpy.ndarray with size (..., N+1).
    """
    if isinstance(points, torch.Tensor):
        pad = points.new_ones(points.shape[:-1]+(1,))
        return torch.cat([points, pad], dim=-1)
    elif isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1]+(1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    else:
        raise ValueError


def from_homogeneous(points):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / points[..., -1:]

def radial_function(t):
    # calculates arctan(t)/t
    return torch.arctan(t) / torch.clamp(t, min=1.0e-8)

@torch.jit.script
def undistort_points(pts, dist):
    '''Undistort normalized 2D coordinates
       and check for validity of the distortion model.
    '''
    dist = dist.unsqueeze(-2)  # add point dimension
    ndist = dist.shape[-1]
    undist = pts
    valid = torch.ones(pts.shape[:-1], device=pts.device, dtype=torch.bool)
    if ndist > 0:
        k1, k2 = dist[..., :2].split(1, -1)
        r2 = torch.sum(pts**2, -1, keepdim=True)
        if not k1.isnan().any():  # TODO: do we want this check to check only one k1 or all? or individually?
            radial = k1*r2 + k2*r2**2
            undist = undist + pts * radial

            # The distortion model is supposedly only valid within the image
            # boundaries. Because of the negative radial distortion, points that
            # are far outside of the boundaries might actually be mapped back
            # within the image. To account for this, we discard points that are
            # beyond the inflection point of the distortion model,
            # e.g. such that d(r + k_1 r^3 + k2 r^5)/dr = 0
            limited = ((k2 > 0) & ((9*k1**2-20*k2) > 0)) | ((k2 <= 0) & (k1 > 0))
            limit = torch.abs(torch.where(
                k2 > 0, (torch.sqrt(9*k1**2-20*k2)-3*k1)/(10*k2), 1/(3*k1)))
            valid = valid & torch.squeeze(~limited | (r2 < limit), -1)
        else:  # if we choose k1=NaN, arctan model is implemented
            r = torch.sqrt(r2)
            # instead of multiplying with arctan(r)/r, set s=arctan(r), and calculate s/tan(s)=cos(s)/sinc(s)

            undist = undist*radial_function(r)

            # I do not think we need to take care of valid here, since distortion model is valid on all of R^2?


        if ndist > 2:  # TODO-G: How is this handled when using the NaN-thing above
            p12 = dist[..., 2:]
            p21 = p12.flip(-1)
            uv = torch.prod(pts, -1, keepdim=True)
            undist = undist + 2*p12*uv + p21*(r2 + 2*pts**2)
            # TODO: handle tangential boundaries

    return undist, valid

@torch.jit.script
def J_undistort_points(pts, dist):
    dist = dist.unsqueeze(-2)  # add point dimension
    ndist = dist.shape[-1]

    J_diag = torch.ones_like(pts)
    J_cross = torch.zeros_like(pts)
    if ndist > 0:
        k1, k2 = dist[..., :2].split(1, -1)
        if not k1.isnan().any():
            r2 = torch.sum(pts**2, -1, keepdim=True)
            uv = torch.prod(pts, -1, keepdim=True)
            radial = k1*r2 + k2*r2**2
            d_radial = (2*k1 + 4*k2*r2)
            J_diag += radial + (pts**2)*d_radial
            J_cross += uv*d_radial
        else:
            r = torch.sqrt(torch.sum(pts**2, -1, keepdim=True))

            f = radial_function(r)  # calculate value of varphi(|x|)

            # formula for jacobian: varphi(|x|)*id + varphi'(|x|)/|x|*xx^T
            # and varphi'(|x|) = 1/|x|*(1/(1+|x|^2)-varphi(|x|))

            g = 1/(1+r**2)-f

            norm_pts = torch.nn.functional.normalize(pts, dim=1, eps=1e-12)

            J_diag = f*J_diag + g * norm_pts**2
            J_cross = g*torch.prod(pts, -1, keepdim=True)

        if ndist > 2:
            p12 = dist[..., 2:]
            p21 = p12.flip(-1)
            J_diag += 2*p12*pts.flip(-1) + 6*p21*pts
            J_cross += 2*p12*pts + 2*p21*pts.flip(-1)

    J = torch.diag_embed(J_diag) + torch.diag_embed(J_cross).flip(-1)
    return J
