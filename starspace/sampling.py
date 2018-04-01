import torch


class NegativeSampling():
    def __init__(self, n_output, n_negative=5, weights=None):
        super(NegativeSampling, self).__init__()
        self.n_output = n_output
        self.n_negative = n_negative
        self.weights = weights
        
    def sample(self, n_samples):
        if self.weights:
            samples = torch.multinomial(self.weights, n_samples, replacement=True)
        else:
            samples = torch.Tensor(n_samples).uniform_(0, self.n_output - 1).round().long()
        return torch.autograd.Variable(samples)
