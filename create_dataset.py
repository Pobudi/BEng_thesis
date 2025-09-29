import os, pickle, numpy as np 

def unpickle(file):
    with open(f"cifar-10-batches-py/{file}", 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_

dicts = []
for f in os.listdir("cifar-10-batches-py"):
    if f.startswith("data"):
        dicts.append(unpickle(f))
meta = unpickle("batches.meta")
label_map = meta[b"label_names"]

X = np.concatenate([i[b'data'] for i in dicts], axis=0)
X = X.reshape(-1, 3, 32, 32)
X = X.transpose(0, 2, 3, 1)
y = np.concatenate([i[b'labels'] for i in dicts], axis=0)

with open ("dataset.pkl", "wb") as f:
    pickle.dump({"X": X, "y": y, "labels": label_map}, f)
