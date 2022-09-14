import os
import torch
import httpx

from statistics import mean
from torch import nn
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from typing import OrderedDict
from .model import init_name_classifier_model


class Peer:
    def __init__(
        self,
        model: nn.Module = init_name_classifier_model(),
        bootstrap_peers: list[str] = [],
        sync_epoch=10,
        max_epoch=100,
    ) -> None:
        self.current_epoch = 0
        self.max_epoch = max_epoch
        self.sync_epoch = sync_epoch
        self.bootstrap_peers = bootstrap_peers
        self.initiated_peers: list[str] = []
        self.model = model

    def sync_state_with_other_peers(self, state: dict):
        for host in self.bootstrap_peers:
            httpx.post(f"{host}/sync/{self.current_epoch}", json=state)

    def update_model(self, cache_dir: str):
        current_state = self.model.state_dict()
        _, _, files = os.walk(cache_dir)
        for file in files:
            with open(os.path.join(cache_dir, file)) as f:
                state: OrderedDict = torch.load(f)

            for layer in state:
                current_state[layer] = torch.add(current_state[layer], state[layer])

        for layer in current_state:
            current_state[layer] = torch.divide(current_state[layer], len(files))

        self.model.load_state_dict(current_state)

    def broadcast_model(self):
        dir = os.path.join("temp", "checkpoints")
        os.mkdirs(dir)
        with open(os.path.join(dir, f"{self.current_epoch}"), "w") as f:
            torch.save(self.model.state_dict(), f)

        with open(os.path.join(dir, f"{self.current_epoch}"), "rb") as f:
            for peer in self.bootstrap_peers:
                httpx.post(f"{peer}/sync/{self.current_epoch}", files={"file": f})

    def train(
        self,
        criterion,
        optimizer,
        train_loader: DataLoader,
        epoch: int,
        privacy_engine: PrivacyEngine,
        device="cpu",
    ):
        accs = []
        losses = []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = self.model(x)
            loss = criterion(logits, y)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            preds = logits.argmax(-1)
            n_correct = float(preds.eq(y).sum())
            batch_accuracy = n_correct / len(y)

            accs.append(batch_accuracy)
            losses.append(float(loss))

        printstr = (
            f"\t Epoch {epoch}. Accuracy: {mean(accs):.6f} | Loss: {mean(losses):.6f}"
        )

        if privacy_engine:
            epsilon = privacy_engine.get_epsilon(self.delta)
            printstr += f" | (ε = {epsilon:.2f}, δ = {self.delta})"

        print(printstr)

        self.broadcast_model()

        dir = os.path.join("temp", "states", epoch)
        os.makedirs(dir)
        _, _, files = os.walk(dir)
        if len(files) == len(self.bootstrap_peers) and epoch == self.current_epoch:
            self.update_model(dir)

    def test(self, test_loader, privacy_engine, device="cpu"):
        accs = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                preds = self.model(x).argmax(-1)
                n_correct = float(preds.eq(y).sum())
                batch_accuracy = n_correct / len(y)

                accs.append(batch_accuracy)
        print_str = (
            "\n----------------------------\n" f"Test Accuracy: {mean(accs):.6f}"
        )
        if privacy_engine:
            epsilon = privacy_engine.get_epsilon(self.delta)
            print_str += f" (ε = {epsilon:.2f}, δ = {self.delta})"
        print(print_str + "\n----------------------------\n")
