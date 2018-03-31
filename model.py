import torch
import torch.nn.functional as F

from torch import nn


class MarginRankingLoss(nn.Module):
    def __init__(self, margin=1., aggregate=torch.mean):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
        self.aggregate = aggregate

    def forward(self, positive_similarity, negative_similarity):
        return self.aggregate(
            torch.clamp(self.margin - positive_similarity + negative_similarity, min=0))


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


class InnerProductSimilarity(nn.Module):
    def __init__(self):
        super(InnerProductSimilarity, self).__init__()

    def forward(self, a, b):
        # a => B x [n_a x] dim, b => B x [n_b x] dim

        if a.dim() == 2:
            a = a.unsqueeze(1)  # B x n_a x dim

        if b.dim() == 2:
            b = b.unsqueeze(1)  # B x n_b x dim

        return torch.bmm(a, b.transpose(2, 1))  # B x n_a x n_b


class StarSpace(nn.Module):
    def __init__(self, dim, n_input, n_output, similarity, max_norm=10, aggregate=torch.sum):
        super(StarSpace, self).__init__()

        self.n_input = n_input
        self.n_output = n_output
        self.similarity = similarity
        self.aggregate = aggregate
        
        self.input_embedding = nn.Embedding(n_input, dim, max_norm=max_norm)
        self.output_embedding = nn.Embedding(n_output, dim, max_norm=max_norm)

    def forward(self, input, output=None):
        if input.dim() == 1:
            input = input.unsqueeze(-1)  # B x L_i

        input_emb = self.input_embedding(input)  # B x L_i x dim
        input_repr = self.aggregate(input_emb, dim=1)  # B x dim
        
        if output is not None:
            if output.dim() == 1:
                output = output.unsqueeze(-1)  # B x L_o

            output_emb = self.output_embedding(output)  # B x L_o x dim
            output_repr = self.aggregate(output_emb, dim=1)  # B x dim
            return input_repr, output_repr
        
        return input_repr,

    #def predict(self, batch):
    #    input = batch.text # B x L
    #    
    #    batch_size = input.size(0)
    #    
    #    output = torch.autograd.Variable(torch.range(0, self.n_output - 1).long().expand(batch_size, -1)) # B x d_output
    #    
    #    if input.is_cuda:
    #        output = output.cuda()
    #    
    #    input_emb = self.input_embedding(input) # B x L x dim
    #    output_emb = self.output_embedding(output) # B x d_output x dim
    #    input_repr = self.aggregate_f(input_emb, dim=1) # B x dim
    #    output_repr = output_emb # B x d_output x dim
    #    
    #    if self.similarity == 'dot':
    #        similarity = torch.bmm(output_repr, input_repr.unsqueeze(dim=-1)).squeeze(dim=-1)
    #        
    #    return similarity.max(dim=-1)[1]