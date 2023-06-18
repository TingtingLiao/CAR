import torch


class Sampler:

    def __init__(self, global_sigma=1.0, local_sigma=0.01):
        self.global_sigma = global_sigma
        self.local_sigma = local_sigma

    def get_points(self, pc_input):
        """
        :param pc_input: [batch, N, dim]
        :return sample [batch, N + N + N // 8, dim]
        """
        batch, N, dim = pc_input.shape

        sample_local = pc_input + torch.randn_like(pc_input) * self.local_sigma

        sample_global = (torch.rand(batch, N // 8, dim, device=pc_input.device) * 2 - 1) * self.global_sigma

        sample = torch.cat([pc_input, sample_local, sample_global], dim=1)

        return sample
