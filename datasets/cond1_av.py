"""Paired Cond_1 acoustic + vibration-1 windows for MCSAT_AV."""
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

FAULTS = ("Health", "IRC", "IRS", "IRW", "ORC", "ORS", "ORW")

def _windows(path, length, step):
    try:
        values = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1,), dtype=np.float32)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid numeric CSV content: {path}. "
                         "Check for binary/NULL bytes or an incomplete upload.") from exc
    for start in range(0, len(values) - length + 1, step):
        x = values[start:start + length].copy()
        x = (x - x.mean()) / (x.std() + 1e-8)
        yield torch.from_numpy(x[None, :])

class Cond1AVDataset(Dataset):
    def __init__(self, root, conditions, length, step, file_pairs=None):
        self.records = []
        self.class_indices = {label: [] for label in range(len(FAULTS))}
        root = Path(root)
        pairs = file_pairs
        if pairs is None:
            pairs = []
            for condition in conditions:
                for label, fault in enumerate(FAULTS):
                    folder = root / condition / fault
                    vib = {p.name: p for p in folder.glob("va (*.csv")}
                    for ap in sorted(folder.glob("acoustic__va (*.csv")):
                        vp = vib.get(ap.name.replace("acoustic__", ""))
                        if vp is None:
                            raise FileNotFoundError(f"No paired vibration file for {ap}")
                        pairs.append((ap, vp, label))
        for ap, vp, label in pairs:
                    aw, vw = list(_windows(ap, length, step)), list(_windows(vp, length, step))
                    # Acoustic and vibration recordings can differ by a few
                    # samples at the tail. Keep only the common window range
                    # so each returned item remains a valid paired example.
                    common_windows = min(len(aw), len(vw))
                    if len(aw) != len(vw):
                        print(f"[Cond_1 pair] trim unmatched tail: {ap.name} "
                              f"{len(aw)} windows vs {vp.name} {len(vw)} "
                              f"windows; using {common_windows}")
                    for a, v in zip(aw[:common_windows], vw[:common_windows]):
                        self.class_indices[label].append(len(self.records))
                        self.records.append((a, v, label))
        if not self.records:
            raise ValueError(f"No paired acoustic/vibration windows found under {root}")

    def __len__(self): return len(self.records)
    def __getitem__(self, i): return self.records[i]

def get_cond1_av_loaders(root, train_conditions, test_condition, batch_size, length, step,
                         train_batches_per_epoch=100, test_batches=55, seed=1):
    train_conditions = [x.strip() for x in train_conditions.split(",") if x.strip()]
    train = Cond1AVDataset(root, train_conditions, length, step)
    test = Cond1AVDataset(root, [test_condition], length, step)
    def balanced_indices(dataset, requested, rng, split_name):
        # Make normal count approximately equal to the combined fault count.
        normal_quota = (requested + 1) // 2
        fault_budget = requested - normal_quota
        fault_base, fault_rem = divmod(fault_budget, len(FAULTS) - 1)
        quotas = {0: normal_quota}
        for label in range(1, len(FAULTS)):
            quotas[label] = fault_base + int(label <= fault_rem)
        shortages = {FAULTS[label]: (quota, len(dataset.class_indices[label]))
                     for label, quota in quotas.items()
                     if len(dataset.class_indices[label]) < quota}
        if shortages:
            detail = "; ".join(f"{name}: need {need}, found {have}"
                                for name, (need, have) in shortages.items())
            raise ValueError(f"{split_name} lacks paired windows for balanced 7-class sampling: {detail}")
        chosen = []
        for label, quota in quotas.items():
            chosen.extend(rng.sample(dataset.class_indices[label], quota))
        rng.shuffle(chosen)
        return chosen, quotas

    train_requested = batch_size * train_batches_per_epoch
    test_requested = batch_size * test_batches
    rng = random.Random(seed)
    train_indices, train_quotas = balanced_indices(train, train_requested, rng, "Train")
    test_indices, test_quotas = balanced_indices(test, test_requested, rng, "Test")
    print(f"[Cond_1 AV balanced] train classes={train_quotas}; test classes={test_quotas}")
    print(f"[Cond_1 AV] train batches/epoch={len(train_indices) // batch_size}, "
          f"test batches={len(test_indices) // batch_size}")
    return (DataLoader(Subset(train, train_indices), batch_size=batch_size,
                       shuffle=True, drop_last=False),
            DataLoader(Subset(test, test_indices), batch_size=batch_size,
                       shuffle=False, drop_last=False))


def get_cond1_av_same_condition_loaders(root, condition, batch_size, length, step,
                                        test_fraction=0.2, train_batches_per_epoch=100,
                                        test_batches=55, seed=1):
    """Split paired recording files within one condition before windowing."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    root_path = Path(root)
    rng = random.Random(seed)
    train_pairs, test_pairs = [], []
    for label, fault in enumerate(FAULTS):
        folder = root_path / condition / fault
        vibrations = {p.name: p for p in folder.glob("va (*.csv")}
        pairs = []
        for acoustic in sorted(folder.glob("acoustic__va (*.csv")):
            vibration = vibrations.get(acoustic.name.replace("acoustic__", ""))
            if vibration is None:
                raise FileNotFoundError(f"Unpaired acoustic recording: {acoustic}")
            pairs.append((acoustic, vibration, label))
        if len(pairs) < 2:
            raise ValueError(f"Need at least 2 paired recording files for {condition}/{fault}; found {len(pairs)}")
        rng.shuffle(pairs)
        n_test = max(1, min(len(pairs) - 1, round(len(pairs) * test_fraction)))
        test_pairs.extend(pairs[:n_test])
        train_pairs.extend(pairs[n_test:])

    train = Cond1AVDataset(root_path, [], length, step, file_pairs=train_pairs)
    test = Cond1AVDataset(root_path, [], length, step, file_pairs=test_pairs)

    def cap_balanced(dataset, max_samples, split):
        per_class = max_samples // len(FAULTS)
        quota = {i: per_class for i in range(len(FAULTS))}
        extra = max_samples % len(FAULTS)
        for i in range(extra):
            quota[i] += 1
        missing = {FAULTS[i]: (n, len(dataset.class_indices[i])) for i, n in quota.items()
                   if len(dataset.class_indices[i]) < n}
        if missing:
            details = "; ".join(f"{k}: need {n}, have {m}" for k, (n, m) in missing.items())
            raise ValueError(f"{split} has too few windows for balanced sampling: {details}")
        picked = []
        for label, n in quota.items():
            picked.extend(rng.sample(dataset.class_indices[label], n))
        rng.shuffle(picked)
        return picked

    train_n = min(len(train), batch_size * train_batches_per_epoch)
    test_n = min(len(test), batch_size * test_batches)
    train_ids = cap_balanced(train, train_n, "Train")
    test_ids = cap_balanced(test, test_n, "Test")
    print(f"[Cond_1 AV same-condition] condition={condition}, split-by-recording-file, "
          f"train_files={len(train_pairs)}, test_files={len(test_pairs)}, "
          f"train_batches={len(train_ids) // batch_size}, test_batches={len(test_ids) // batch_size}")
    return (DataLoader(Subset(train, train_ids), batch_size=batch_size, shuffle=True),
            DataLoader(Subset(test, test_ids), batch_size=batch_size, shuffle=False))


class Cond1ModalityView(Dataset):
    """View paired windows as acoustic-only, vibration-only, or paired inputs."""
    def __init__(self, paired_dataset, modality):
        if modality not in {"acoustic", "vibration", "av"}:
            raise ValueError("modality must be one of: acoustic, vibration, av")
        self.paired_dataset = paired_dataset
        self.modality = modality

    def __len__(self):
        return len(self.paired_dataset)

    def __getitem__(self, index):
        acoustic, vibration, label = self.paired_dataset[index]
        if self.modality == "acoustic":
            return acoustic, label
        if self.modality == "vibration":
            return vibration, label
        return acoustic, vibration, label


def get_cond1_same_condition_loaders(root, condition, modality, batch_size, length, step,
                                    test_fraction=0.2, train_batches_per_epoch=100,
                                    test_batches=55, seed=1):
    """Build aligned train/test loaders from one shared file-level split.

    Files are split per class before windowing. Calling this function with the
    same root, condition, fractions, and seed gives acoustic-only, vibration-
    only, and paired AV runs identical recording files and sampled window IDs.
    """
    if modality not in {"acoustic", "vibration", "av"}:
        raise ValueError("modality must be one of: acoustic, vibration, av")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    root_path = Path(root)
    split_pairs = {"train": [], "test": []}
    split_counts = {name: {} for name in split_pairs}
    for label, fault in enumerate(FAULTS):
        folder = root_path / condition / fault
        vibrations = {p.name: p for p in folder.glob("va (*.csv")}
        pairs = []
        for acoustic in sorted(folder.glob("acoustic__va (*.csv")):
            vibration = vibrations.get(acoustic.name.replace("acoustic__", ""))
            if vibration is None:
                raise FileNotFoundError(f"Unpaired acoustic recording: {acoustic}")
            pairs.append((acoustic, vibration, label))
        if len(pairs) < 2:
            raise ValueError(f"Need at least 2 paired recording files for {condition}/{fault} "
                             f"to make train/test splits; found {len(pairs)}")

        # Class-specific RNG keeps the split identical for every modality.
        random.Random(seed + label).shuffle(pairs)
        n_test = max(1, round(len(pairs) * test_fraction))
        n_test = min(n_test, len(pairs) - 1)
        split_pairs["test"].extend(pairs[:n_test])
        split_pairs["train"].extend(pairs[n_test:])
        split_counts["test"][FAULTS[label]] = n_test
        split_counts["train"][FAULTS[label]] = len(pairs) - n_test

    datasets = {
        name: Cond1AVDataset(root_path, [], length, step, file_pairs=pairs)
        for name, pairs in split_pairs.items()
    }

    def balanced_window_ids(dataset, max_samples, split_name, rng):
        max_samples = min(len(dataset), max_samples)
        # Keep normal-vs-all-fault counts near 1:1, then distribute fault
        # examples evenly across the six fault classes.
        normal_quota = (max_samples + 1) // 2
        fault_budget = max_samples - normal_quota
        fault_base, fault_remainder = divmod(fault_budget, len(FAULTS) - 1)
        quotas = {0: normal_quota}
        for label in range(1, len(FAULTS)):
            quotas[label] = fault_base + int(label <= fault_remainder)
        insufficient = {
            FAULTS[label]: (quota, len(dataset.class_indices[label]))
            for label, quota in quotas.items()
            if len(dataset.class_indices[label]) < quota
        }
        if insufficient:
            details = "; ".join(f"{name}: need {need}, have {have}"
                                for name, (need, have) in insufficient.items())
            raise ValueError(f"{split_name} has too few windows for balanced sampling: {details}")
        selected = []
        for label, quota in quotas.items():
            selected.extend(rng.sample(dataset.class_indices[label], quota))
        rng.shuffle(selected)
        return selected, quotas

    rng = random.Random(seed + 10000)
    caps = {
        "train": batch_size * train_batches_per_epoch,
        "test": batch_size * test_batches,
    }
    selected = {}
    quotas = {}
    for split_name in ("train", "test"):
        selected[split_name], quotas[split_name] = balanced_window_ids(
            datasets[split_name], caps[split_name], split_name.capitalize(), rng)

    views = {name: Cond1ModalityView(dataset, modality)
             for name, dataset in datasets.items()}
    print(f"[Cond_1 same-condition] condition={condition}, modality={modality}, "
          f"split-by-recording-file train/test="
          f"{1.0 - test_fraction:.0%}/{test_fraction:.0%}")
    print(f"[Cond_1 files per class] {split_counts}")
    print(f"[Cond_1 sampled windows per class] {quotas}")
    print(f"[Cond_1 batches] train={len(selected['train']) // batch_size}, "
          f"test={len(selected['test']) // batch_size}")

    return (
        DataLoader(Subset(views["train"], selected["train"]), batch_size=batch_size, shuffle=True),
        DataLoader(Subset(views["test"], selected["test"]), batch_size=batch_size, shuffle=False),
    )
