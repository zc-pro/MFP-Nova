"""NumPy-only threshold fitting on observed Validation labels."""
import numpy as np


GRID = np.round(np.arange(.05, .96, .01), 2)


def ratio(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=np.asarray(b) != 0)


class SetScores:
    """Evaluate one/two changed columns using cached sample/label counts."""
    def __init__(self, targets, predicted, function_spec=None):
        self.y = np.asarray(targets, dtype=bool)
        self.b = np.asarray(predicted, dtype=bool)
        self.actual = self.y.sum(1)
        self.guessed = self.b.sum(1)
        self.intersection = (self.y & self.b).sum(1)
        self.label_actual = self.y.sum(0)
        self.label_guessed = self.b.sum(0)
        self.label_tp = (self.y & self.b).sum(0)
        self.function_spec = function_spec
        if function_spec is not None:
            self.truth_functions = self.functions(self.y)

    def functions(self, values):
        generic, specific = self.function_spec
        result = values.copy()
        result[:, generic] &= ~result[:, specific].any(1)
        return result

    def score(self, columns=(), replacements=None):
        intersection, guessed = self.intersection.copy(), self.guessed.copy()
        label_guessed, label_tp = self.label_guessed.copy(), self.label_tp.copy()
        for j, new in zip(columns, () if replacements is None else replacements):
            delta = new.astype(int) - self.b[:, j]
            intersection += delta * self.y[:, j]
            guessed += delta
            label_guessed[j] += delta.sum()
            label_tp[j] += (delta * self.y[:, j]).sum()
        errors = self.actual + guessed - 2*intersection
        result = {
            'accuracy': float(ratio(intersection, self.actual+guessed-intersection).mean()),
            'precision': float(ratio(intersection, guessed).mean()),
            'coverage': float(ratio(intersection, self.actual).mean()),
            'absolute_true': float((errors == 0).mean()),
            'absolute_false': float(errors.sum()/self.y.size),
            'macro_f1': float(ratio(2*label_tp, self.label_actual+label_guessed).mean()),
            'micro_f1': float(2*intersection.sum()/max(1, self.actual.sum()+guessed.sum())),
        }
        if self.function_spec is not None:
            predicted = self.b.copy()
            for j, new in zip(columns, () if replacements is None else replacements):
                predicted[:, j] = new
            guessed_functions = self.functions(predicted)
            actual = self.truth_functions.sum(1) >= 2
            guessed = guessed_functions.sum(1) >= 2
            result['mfp_f1'] = float(2*(actual & guessed).sum()/max(1, actual.sum()+guessed.sum()))
            result['function_recovery'] = float(((self.truth_functions & guessed_functions).sum(1) >= 2).sum()
                                                / max(1, actual.sum()))
        return result


def fit_global(targets, probabilities):
    scores = [(SetScores(targets, probabilities >= t).score()['accuracy'], -abs(t-.5), -t, t)
              for t in GRID]
    return float(max(scores)[-1])


def fit_labels(targets, probabilities, global_threshold):
    thresholds = []
    for j in range(targets.shape[1]):
        y = targets[:, j].astype(bool)
        if not y.any():
            thresholds.append(global_threshold)
            continue
        candidates = probabilities[:, j, None] >= GRID
        f1 = ratio(2*(y[:, None] & candidates).sum(0), y.sum()+candidates.sum(0))
        index = max(range(len(GRID)), key=lambda k: (f1[k], -abs(GRID[k]-global_threshold), -GRID[k]))
        thresholds.append(float(GRID[index]))
    return np.asarray(thresholds)


def protected(score, baseline):
    """Fit-set constraints only; no promise on held-out data."""
    higher = ['accuracy', 'absolute_true', 'macro_f1']
    higher += [k for k in ('mfp_f1', 'function_recovery') if k in baseline]
    return (all(score[k] >= baseline[k]-1e-12 for k in higher)
            and score['absolute_false'] <= baseline['absolute_false']+1e-12)


def fit_thresholds(targets, probabilities, clusters, kappa=20., radius=.10, function_spec=None):
    """Four fixed methods; no hyperparameter search on the held-out fold.

    Positive-cluster counts control shrinkage and the joint search radius.
    Joint search protects Acc, AT, Macro-F1 and AF on the fit partition.
    Optional major15 function mapping also protects MFP-F1 and recovery of
    at least two observed independent functions. Search settings stay fixed.
    """
    y = np.asarray(targets, dtype=bool)
    t0 = fit_global(y, probabilities)
    global_vector = np.full(y.shape[1], t0)
    independent = fit_labels(y, probabilities, t0)
    support = np.asarray([len(np.unique(clusters[y[:, j]])) for j in range(y.shape[1])])
    strength = support/(support+kappa)
    shrunk = t0 + strength*np.clip(independent-t0, -radius, radius)
    lower, upper = np.maximum(.05, t0-strength*radius), np.minimum(.95, t0+strength*radius)
    grids = [np.unique(np.r_[t0, shrunk[j], GRID[(GRID >= lower[j]) & (GRID <= upper[j])]])
             for j in range(y.shape[1])]
    baseline = SetScores(y, probabilities >= t0, function_spec).score()
    def key(score, thresholds):
        return (score['accuracy'], score['absolute_true'], score['macro_f1'],
                -float(np.abs(thresholds-t0).sum()))
    current = global_vector.copy()
    score = baseline
    shrunk_score = SetScores(y, probabilities >= shrunk, function_spec).score()
    if protected(shrunk_score, baseline) and key(shrunk_score, shrunk) > key(score, current):
        current, score = shrunk.copy(), shrunk_score
    single_moves = pair_moves = 0
    # Three bounded passes; every accepted move improves the same set objective.
    for _ in range(3):
        changed = False
        for j, grid in enumerate(grids):
            cached = SetScores(y, probabilities >= current, function_spec)
            chosen, best = current.copy(), score
            for t in grid:
                proposal = current.copy(); proposal[j] = t
                trial = cached.score([j], [probabilities[:, j] >= t])
                if protected(trial, baseline) and key(trial, proposal) > key(best, chosen):
                    chosen, best = proposal, trial
            if not np.array_equal(chosen, current):
                current, score = chosen, best
                single_moves += 1; changed = True
        if changed:
            continue
        b = probabilities >= current
        fp, fn = (~y & b).astype(np.int64), (y & ~b).astype(np.int64)
        swaps = fp.T @ fn + fn.T @ fp
        pairs = [(j, k) for j in range(y.shape[1]) for k in range(j+1, y.shape[1]) if swaps[j, k] > 0]
        pairs.sort(key=lambda jk: (-swaps[jk], jk))
        cached = SetScores(y, b, function_spec)
        chosen, best = current.copy(), score
        for j, k in pairs[:6]:
            # Seven deterministic grid locations plus current value per column.
            gj = np.unique(np.r_[current[j], grids[j][np.linspace(0, len(grids[j])-1, 7).astype(int)]])
            gk = np.unique(np.r_[current[k], grids[k][np.linspace(0, len(grids[k])-1, 7).astype(int)]])
            for tj in gj:
                for tk in gk:
                    proposal = current.copy(); proposal[j], proposal[k] = tj, tk
                    trial = cached.score([j, k], [probabilities[:, j] >= tj, probabilities[:, k] >= tk])
                    if protected(trial, baseline) and key(trial, proposal) > key(best, chosen):
                        chosen, best = proposal, trial
        if np.array_equal(chosen, current):
            break
        current, score = chosen, best
        pair_moves += 1
    joint_name = 'joint_functions' if function_spec is not None else 'joint'
    return {'global': global_vector, 'per_label': independent, 'shrunk': shrunk, joint_name: current}, {
        'global_threshold': t0, 'positive_clusters': support.tolist(), 'shrinkage_strength': strength.tolist(),
        'single_moves': single_moves, 'pair_moves': pair_moves,
        'joint_fit_constraints_satisfied': protected(score, baseline)}


def cluster_folds(clusters):
    """Two balanced folds, assigning entire clusters by descending size."""
    unique, inverse, counts = np.unique(clusters, return_inverse=True, return_counts=True)
    if len(unique) < 2:
        raise ValueError('Cross-fitting requires at least two sequence clusters')
    membership = np.zeros(len(unique), dtype=int)
    sizes = [0, 0]
    for j in np.argsort(-counts, kind='stable'):
        fold = int(sizes[1] < sizes[0])
        membership[j] = fold; sizes[fold] += int(counts[j])
    assigned = membership[inverse]
    return [(np.flatnonzero(assigned != k), np.flatnonzero(assigned == k)) for k in (0, 1)]
