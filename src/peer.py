import os
import time
from statistics import mean
from typing import OrderedDict

import httpx
import torch
from opacus import PrivacyEngine
from torch import nn
from torch.utils.data import DataLoader

from config import settings
from src.data import test_loader


class Peer:
    def __init__(
            self,
            model: nn.Module,
            criterion,
            optimizer,
            train_loader: DataLoader,
            privacy_engine: PrivacyEngine,
            device="cpu",
            bootstrap_peers=None,
            sync_epoch=settings.sync_epochs,
            max_epoch=settings.max_epochs,
    ) -> None:
        if bootstrap_peers is None:
            bootstrap_peers = []
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.privacy_engine = privacy_engine
        self.device = device
        self.current_epoch = 0
        self.max_epoch = max_epoch
        self.sync_epoch = sync_epoch
        self.bootstrap_peers = bootstrap_peers
        self.initiated_peers: list[str] = []
        self.model = model
        self.is_training = False

    def update_model(self, cache_dir: str):
        print("Updating model")
        current_state = self.model.state_dict()
        for _, _, files in os.walk(cache_dir):
            if len(files) == 0:
                return
            for file in files:
                with open(os.path.join(cache_dir, file), 'rb') as f:
                    state: OrderedDict = torch.load(f)

                for layer in state:
                    current_state[layer] = torch.add(current_state[layer], state[layer])

            for layer in current_state:
                current_state[layer] = torch.divide(current_state[layer], len(files))

            self.model.load_state_dict(current_state)
            break
        print("Model is updated")

    def __broadcast_model(self):
        print("broadcast model to other peers.")
        dir = os.path.join(settings.temp_dir, "checkpoints")
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(os.path.join(dir, f"{self.current_epoch}"), "wb") as f:
            torch.save(self.model.state_dict(), f)

        with open(os.path.join(dir, f"{self.current_epoch}"), "rb") as f:
            for peer in self.bootstrap_peers:
                print(f"Sending model state to {peer}")
                response = httpx.post(f"{peer}/sync/{self.current_epoch}", files={"state": f})
                print(f"Model has been sent to {peer}")
        print("Finished broadcasting model.")

    def train(
            self
    ):
        self.is_training = True
        while self.current_epoch < self.max_epoch:
            self.current_epoch += 1

            acc_array = []
            losses = []

            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                preds = logits.argmax(-1)
                n_correct = float(preds.eq(y).sum())
                batch_accuracy = n_correct / len(y)

                acc_array.append(batch_accuracy)
                losses.append(float(loss))

            log = f"Epoch {self.current_epoch}. Accuracy: {mean(acc_array):.6f} | Loss: {mean(losses):.6f}"
            if self.privacy_engine:
                epsilon = self.privacy_engine.get_epsilon(settings.delta)
                log += f" | (ε = {epsilon:.2f}, δ = {settings.delta})"
            print(log)

            if self.__should_update_state():
                self.__broadcast_model()
                print("Suspend training. Waiting for other peers' state.")
                self.__wait_for_other_state()
                self.update_model(os.path.join(settings.temp_dir, "states", str(self.current_epoch)))
                print("Resume training.")

        self.is_training = False
        model_path = os.path.join(settings.temp_dir, 'model.pt')
        torch.save(self.model, model_path)
        print(f"The training is finished. The model has been saved to {model_path}")

        self.test(data_loader=test_loader)

    def __wait_for_other_state(self):
        while True:
            if self.has_received_all_state():
                break
            time.sleep(1)

    def __get_state_dir(self):
        return os.path.join(settings.temp_dir, "states", str(self.current_epoch))

    def __should_update_state(self) -> bool:
        return self.current_epoch % self.sync_epoch == 0 and self.current_epoch != 0

    def has_received_all_state(self) -> bool:
        state_dir = self.__get_state_dir()
        if not os.path.exists(state_dir):
            os.makedirs(state_dir)
        for (_, _, files) in os.walk(state_dir):
            return len(files) == len(self.bootstrap_peers)

    def test(self, data_loader: DataLoader):
        acc_array = []
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                preds = self.model(x).argmax(-1)
                n_correct = float(preds.eq(y).sum())
                batch_accuracy = n_correct / len(y)

                acc_array.append(batch_accuracy)
        print_str = (
            "\n----------------------------\n" f"Test Accuracy: {mean(acc_array):.6f}"
        )
        if self.privacy_engine:
            epsilon = self.privacy_engine.get_epsilon(settings.delta)
            print_str += f" (ε = {epsilon:.2f}, δ = {settings.delta})"
        print(print_str + "\n----------------------------\n")
