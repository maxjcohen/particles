import torch

from ..distributions import ProbDist


class TorchModifiedDist(ProbDist):
    def __init__(self, torch_dist):
        super().__init__()
        self.torch_dist = torch_dist

    def shape(self, size):
        if size is None:
            return []
        try:
            return torch.Size(size)
        except TypeError:
            return torch.Size((size,))

    def rvs(self, size=None):
        return self.torch_dist.sample(self.shape(size)).squeeze()

    def logpdf(self, x):
        import pdb;pdb.set_trace()
        return self.torch_dist.log_prob(x)


def torch_distribution(torch_dist):
    def law(*args, **kwargs):
        return TorchModifiedDist(torch_dist(*args, **kwargs))

    return law
