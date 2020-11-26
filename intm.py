import torch
from torch import nn
import math

def geometric_transform(pose_tensor,
                        similarity=False,
                        nonlinear=True,
                        as_matrix=False):
    """Converts pose tensor into an affine or similarity transform.

    Args:
      pose_tensor: [..., 6] tensor.
      similarity (bool):
      nonlinear (bool): Applies non-linearities to pose params if True.
      as_matrix (bool): Converts the transform to a matrix if True.

    Returns:
      [..., 3, 3] tensor if `as_matrix` else [..., 6] tensor.
    """

    scale_x, scale_y, theta, shear, trans_x, trans_y = torch.split(pose_tensor,
                                                                   1,
                                                                   dim=-1)

    if nonlinear:
        scale_x, scale_y = (torch.sigmoid(t) + 1e-2 for t in (scale_x, scale_y))

        trans_x, trans_y, shear = (torch.tanh(t * 5.)
                                   for t in (trans_x, trans_y, shear))
        theta *= 2. * math.pi
    else:
        scale_x, scale_y = (abs(t) + 1e-2 for t in (scale_x, scale_y))

    c, s = torch.cos(theta), torch.sin(theta)

    if similarity:
        scale = scale_x
        pose = [scale * c, -scale * s, trans_x,
                scale * s, scale * c, trans_y]
    else:
        pose = [
            scale_x * c + shear * scale_y * s,
            -scale_x * s + shear * scale_y * c,
            trans_x,
            scale_y * s,
            scale_y * c,
            trans_y
        ]

    pose = torch.cat(pose, -1)

    # convert to a matrix
    if as_matrix:
        shape = list(pose.shape[:-1])
        shape += [2, 3]
        pose = pose.view(*shape)
        zeros = torch.zeros_like(pose[..., :1, 0])
        last = torch.stack([zeros, zeros, zeros + 1], -1)
        pose = torch.cat([pose, last], -2)

    return pose