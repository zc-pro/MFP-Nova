import itertools

import torch
from torch.nn import functional as F

from mfp_nova.hh_v17 import fit_threshold, jaccard
from mfp_nova.models.hh_setreread_v17 import HHSetRereadV17, decisions, propose


def test_action_proposals_match_exhaustive_scores():
    base = torch.tensor([[True, True, False, False], [False] * 4, [True] * 4])
    scores = torch.tensor([[.24, .82, .71, .41], [.7, .6, .2, .1], [.2, .4, .6, .9]])
    proposal = propose(base, scores)
    for i, kept in enumerate(base):
        old = (scores[i] * kept).sum() / (kept.sum() + (scores[i] * ~kept).sum())
        old_ids = kept.nonzero().flatten().tolist()
        new_ids = (~kept).nonzero().flatten().tolist()
        options = [[(a, None) for a in new_ids] if old_ids else [],
                   [(None, d) for d in old_ids] if len(old_ids) > 1 else [],
                   list(itertools.product(new_ids, old_ids))]
        for action, allowed in enumerate(options):
            gains = []
            for add, drop in allowed:
                changed = kept.clone()
                if add is not None:
                    changed[add] = True
                if drop is not None:
                    changed[drop] = False
                gains.append((scores[i] * changed).sum() / (changed.sum() + (scores[i] * ~changed).sum()) - old)
            expected = max(gains) if gains else torch.tensor(-torch.inf)
            torch.testing.assert_close(proposal[0][i, action], expected)
    final, kind, added, removed = decisions(base, proposal, 0., "swap")
    assert final[0].tolist() == [False, True, True, False]
    assert (kind[0], added[0], removed[0]) == (2, 2, 0)
    assert torch.equal(final[1:], base[1:])


def test_threshold_search_handles_tied_action_scores():
    base = torch.tensor([[True, False]] * 3)
    scores = torch.tensor([[.1, .9], [.2, .8], [.2, .8]])
    truth = torch.tensor([[False, True], [True, False], [True, False]])
    proposal = propose(base, scores)
    threshold = fit_threshold(base, proposal, truth, "swap")
    final = decisions(base, proposal, threshold, "swap")[0]
    assert final.tolist() == [[False, True], [True, False], [True, False]]
    assert jaccard(final, truth).mean().item() == 1.


def test_local_readout_padding_and_coexistence_targets():
    torch.manual_seed(157)
    projection = torch.eye(15).tolist()
    model = HHSetRereadV17(projection, dim=8, neighbors=False)
    unary = torch.full((1, 15), -1.)
    unary[0, 4] = 1.
    batch = {"esm": torch.randn(1, 4, 1280), "esm_peptide": torch.randn(1, 4, 640),
             "aa": torch.tensor([[1, 2, 3, 4]]), "mask": torch.ones(1, 4, dtype=torch.bool),
             "base_logits": unary, "y": F.one_hot(torch.tensor([6]), 15).float()}
    padded = dict(batch, esm=F.pad(batch["esm"], (0, 0, 0, 3)),
                  esm_peptide=F.pad(batch["esm_peptide"], (0, 0, 0, 3)),
                  aa=F.pad(batch["aa"], (0, 3)), mask=F.pad(batch["mask"], (0, 3)))
    first, extended = model(batch), model(padded)
    torch.testing.assert_close(model.reader(batch), model.reader(padded))
    torch.testing.assert_close(first["coexist_logits"], extended["coexist_logits"])
    torch.testing.assert_close(first["logits"], unary)
    assert not first["eligible"][0, 0]  # Specific antimicrobial suppresses the generic reference axis.
    assert first["eligible"][0, 6]
    model.loss(batch, first, .2).backward()
    assert torch.isfinite(model.identity[-1].weight.grad).all()
    assert model.coexist[-1].weight.grad.abs().sum() > 0
