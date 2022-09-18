from os import path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from pathlib import Path
from torch.utils.data import DataLoader
import torch
from torch import nn
from config import settings

VOCAB_SIZE = 256 + 3


class CharByteEncoder(nn.Module):

    def __init__(self):
        super(CharByteEncoder, self).__init__()
        self.start_token = "<s>"
        self.end_token = "</s>"
        self.pad_token = "<pad>"

        self.start_idx = 256
        self.end_idx = 257
        self.pad_idx = 258

    def forward(self, s: str, pad_to=0):
        encoded = s.encode()
        n_pad = pad_to - len(encoded) if pad_to > len(encoded) else 0
        return torch.LongTensor(
            [self.pad_idx]
            + [c for c in encoded]
            + [self.end_idx]
            + [self.pad_idx for _ in range(n_pad)]
        )

    def decode(self, char_ids_tensor: torch.LongTensor):
        char_ids = char_ids_tensor.cpu().detach().tolist()

        out = []
        buf = []

        for c in char_ids:
            if c < 256:
                buf.append(c)
            else:
                if buf:
                    out.append(bytes(buf).decode())
                    buf = []
                if c == self.start_idx:
                    out.append(self.start_idx)
                elif c == self.end_idx:
                    out.append(self.end_idx)
                elif c == self.pad_idx:
                    out.append(self.pad_idx)
        if buf:
            out.append(bytes(buf).decode())
        return "".join(out)

    def __len__(self):
        return 259


class NamesDataset(Dataset):
    def __init__(self, root: str):
        self.root = Path(root)

        self.labels = list({lang_file.stem for lang_file in self.root.iterdir()})
        self.labels_dict = {label: i for i, label in enumerate(self.labels)}
        self.encoder = CharByteEncoder()
        self.samples = self.construct_samples()

    def construct_samples(self):
        samples = []
        for langFile in self.root.iterdir():
            label_name = langFile.stem
            label_id = self.labels_dict[label_name]
            with open(langFile, "r") as fin:
                for row in fin:
                    samples.append(
                        (self.encoder(row.strip()), torch.tensor(label_id).long())
                    )
        return samples

    def __getitem__(self, i):
        return self.samples[i]

    def __len__(self):
        return len(self.samples)

    def label_count(self):
        cnt = {}
        for _x, y in self.samples:
            label = self.labels[int(y)]
            cnt[label] = cnt.get(label, 0) + 1
        return cnt


def padded_collate(batch, padding_idx=0):
    x = pad_sequence(
        [elem[0] for elem in batch], batch_first=True, padding_value=padding_idx
    )
    y = torch.stack([elem[1] for elem in batch]).long()

    return x, y

names_folder = path.join(settings.data_path)
train_split = 0.8
batch_size = 800

ds = NamesDataset(names_folder)
train_len = int(train_split * len(ds))
test_len = len(ds) - train_len

train_ds, test_ds = torch.utils.data.random_split(ds, [train_len, test_len])

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    pin_memory=True,
    collate_fn=padded_collate
)

test_loader = DataLoader(
    test_ds,
    batch_size=2 * batch_size,
    shuffle=False,
    pin_memory=True,
    collate_fn=padded_collate
)
