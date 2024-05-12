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


def train_val_test(data, ys, test_size=0.2, random_state=42, stratify=True):
    
    assert len(data) == len(ys), f'len(data): {len(data)}, len(ys): {len(ys)}'
    indices = np.arange(len(data))

    if stratify:
        train_val_indices, test_indices, train_val_ys, test_ys = train_test_split(
            indices, ys, test_size=test_size,
            stratify=ys, random_state=random_state)

        train_indices, val_indices, train_ys, val_ys = train_test_split(
            train_val_indices, train_val_ys, test_size=test_size,
            stratify=train_val_ys, random_state=random_state)
    else:
        train_val_indices, test_indices, train_val_ys, test_ys = train_test_split(
            indices, ys, test_size=test_size,
            stratify=None, random_state=random_state)

        train_indices, val_indices, train_ys, val_ys = train_test_split(
            train_val_indices, train_val_ys, test_size=test_size/(1.-test_size),
            stratify=None, random_state=random_state)

    train_data = [data[idx] for idx in train_indices]
    val_data = [data[idx] for idx in val_indices]
    test_data = [data[idx] for idx in test_indices]

    return ((train_data, train_ys), (val_data, val_ys), (test_data, test_ys), 
             train_indices, val_indices, test_indices)


def equal_chunks_indices(ys, n_chunks=5, stratify=False, random_state=42):
    N = len(ys)
    chunks = list()
    if stratify:
        indices = np.arange(N)
        skf = StratifiedKFold(n_splits=n_chunks, shuffle=True, random_state=random_state)
        for index, _ in skf.split(indices):
            chunks.append(index)
        return chunks
    np.random.seed(random_state)
    indices = np.random.permutation(N)
    delimiters = np.linspace(0, N, n_chunks + 1, dtype=int)
    for i in range(n_chunks):
        chunks.append(indices[delimiters[i]:delimiters[i+1]])
    return chunks


def train_val_test_iterate(chunks):
    N = len(chunks)
    for val_index, test_index in it.permutations(range(N), 2):
        val_chunk = chunks[val_index]
        test_chunk = chunks[test_index]
        a = min(val_index, test_index)
        b = max(val_index, test_index)
        train_chunk = np.concatenate((
            chunks[:a] + chunks[a+1:b] + chunks[b+1:]
        ))
        yield train_chunk, val_chunk, test_chunk


def train_val_test_random(chunks, random_state=42):
    N = len(chunks)
    np.random.seed(random_state)
    val_index, test_index = np.random.choice(N, size=2, replace=False)
    a = min(val_index, test_index)
    b = max(val_index, test_index)
    val_chunk = chunks[val_index]
    test_chunk = chunks[test_index]
    train_chunk = np.concatenate((
        chunks[:a] + chunks[a+1:b] + chunks[b+1:]
    ))
    return train_chunk, val_chunk, test_chunk


def train_val_test_subsets(data, ys, chunks):
    train_data = []
    train_ys = []
    val_data = []
    val_ys = []
    test_data = []
    test_ys = []

    for piece, data_, ys_ in zip(
        (0, 1, 2),
        (train_data, val_data, test_data),
        (train_ys, val_ys, test_ys)
    ):
        for idx in chunks[piece]:
            data_.append(data[idx])
            ys_.append(ys[idx])

    return (train_data, train_ys), (val_data, val_ys), (test_data, test_ys)
