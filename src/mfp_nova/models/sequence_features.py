"""Small shared sequence inputs and supervised loss for the three candidates."""
import torch
from torch import nn
from torch.nn import functional as F

from .theraset import residue_encoding
from ..research_layers import observed_label_loss
from ..consistency import bernoulli_rdrop


class FrozenFeatures(nn.Module):
    def __init__(self, input_dim, dim, peptide_input_dim=0):
        super().__init__()
        self.primary_norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.peptide_norm = (nn.LayerNorm(peptide_input_dim, elementwise_affine=False)
                             if peptide_input_dim else None)
        self.projection = nn.Linear(input_dim+peptide_input_dim, dim)

    def forward(self, batch):
        x = self.primary_norm(batch["esm"].masked_fill(~batch["mask"][..., None], 0))
        if self.peptide_norm is not None:
            other = self.peptide_norm(batch["esm_peptide"].masked_fill(~batch["mask"][..., None], 0))
            x = torch.cat([x, other], -1)
        return self.projection(x).masked_fill(~batch["mask"][..., None], 0)


class ResidueCNN(nn.Module):
    def __init__(self, dropout, channels=64):
        super().__init__()
        self.register_buffer("encoding", residue_encoding())
        self.input = nn.Linear(34, 64)
        self.convs = nn.ModuleList([nn.ModuleList([
            nn.Conv1d(width, channels, k, padding=k//2) for k in (3, 5, 7, 9)])
            for width in (64, 4*channels)])
        self.norms = nn.ModuleList([nn.ModuleList([nn.LayerNorm(channels) for _ in range(4)]) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        mask = batch["mask"][..., None]
        x = self.input(self.encoding[batch["aa"]]).masked_fill(~mask, 0)
        for convs, norms in zip(self.convs, self.norms):
            x = torch.cat([F.gelu(norm(conv(x.transpose(1, 2)).transpose(1, 2)))
                           for conv, norm in zip(convs, norms)], -1)
            x = self.dropout(x).masked_fill(~mask, 0)
        return x


def prior_bias(statistics):
    return torch.tensor(statistics["observed_prior"]).clamp(1e-4, 1-1e-4).logit()


def supervised_loss(model, batch, settings, return_output=False):
    output = model(batch)
    loss = observed_label_loss(output["logits"], batch["y"], model.statistics,
                               settings.get("unlabeled_weight", .3))
    agreement = output["logits"].sum()*0
    if model.training and settings.get("rdrop", 0) > 0:
        second = model(batch)
        loss = .5*(loss+observed_label_loss(second["logits"], batch["y"], model.statistics,
                                            settings.get("unlabeled_weight", .3)))
        agreement = bernoulli_rdrop(output["logits"], second["logits"])
    terms = {"classification": loss, "rdrop": agreement}
    total = loss+settings.get("rdrop", 0)*agreement
    return (total, terms, output) if return_output else (total, terms)
