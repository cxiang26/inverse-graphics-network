import torch

def om(ac_contributions):
    temp = ac_contributions * 70
    ac_contributions = torch.exp(temp)
    return torch.log(torch.sum(ac_contributions, dim=1, keepdim=True)) / 70. - 0.01