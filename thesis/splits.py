from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np


def train_val_skf(data, ys, fold=5, random_state=42):

    assert len(data) == len(ys), f'len(data): {len(data)}, len(ys): {len(ys)}'

    indices = np.arange(len(data))
    skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)

    for train_indices, val_indices in skf.split(indices, ys):
        train_data = []
        train_ys = []
        val_data = []
        val_ys = []

        for idx in train_indices:
            train_data.append(data[idx])
            train_ys.append(ys[idx])
        for idx in val_indices:
            val_data.append(data[idx])
            val_ys.append(ys[idx])

        yield (train_data, train_ys), (val_data, val_ys), train_indices, val_indices


def train_val_test_stratified(data, ys, test_size=0.2, random_state=42):
    
    assert len(data) == len(ys), f'len(data): {len(data)}, len(ys): {len(ys)}'
    indices = np.arange(len(data))

    train_val_indices, test_indices, train_val_ys, test_ys = train_test_split(
        indices, ys, test_size=test_size,
        stratify=ys, random_state=random_state)

    train_indices, val_indices, train_ys, val_ys = train_test_split(
        train_val_indices, train_val_ys, test_size=test_size,
        stratify=train_val_ys, random_state=random_state)

    train_data = [data[idx] for idx in train_indices]
    val_data = [data[idx] for idx in val_indices]
    test_data = [data[idx] for idx in test_indices]

    return (train_data, train_ys), (val_data, val_ys), (test_data, test_ys)