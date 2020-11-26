import torch
from torch import nn
import torch.nn.functional as F
from intm import geometric_transform

from monty.collections import AttrDict

class ACR(nn.Module):
    def __init__(self,
                 n_templates,
                 template_size,
                 image_size
                 ):
        super(ACR, self).__init__()

        self.image_size = image_size
        self.template_size = template_size
        # those templates are easily death according to the different activation function, such as relu
        # relu6 and (tanh + log(1+exp(x*10))/10) are the best choice.
        self.template = nn.Parameter(torch.zeros(1, n_templates, template_size, template_size))
        nn.init.orthogonal_(self.template.view(n_templates, -1), )

    def forward(self, pose, intensity):
        B, C, _ = pose.size()
        template_ = F.tanh(self.template)
        template_ = torch.log(1 + torch.exp(template_ * 10)) / 10.
        pose = pose.view(-1, pose.size(-1))
        pose = geometric_transform(pose, as_matrix=True)[:, :2]
        grid = F.affine_grid(pose, (B*C, 1, self.image_size, self.image_size), align_corners=False)
        templates = template_.repeat(B, 1, 1, 1)
        templates = templates.view(B*C, 1, self.template_size, self.template_size)
        transformed_templates = F.grid_sample(templates, grid, align_corners=False)
        transformed_templates = transformed_templates.view(B, C, self.image_size, self.image_size) * intensity.unsqueeze(dim=-1)
        return AttrDict(templates=template_,
                        transformed_templates=transformed_templates)

