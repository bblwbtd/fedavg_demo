import torch

from torch import nn
from config import settings
from src.data import ds, train_loader
from opacus.layers import DPLSTM
from opacus import PrivacyEngine


class CharNNClassifier(nn.Module):
    def __init__(
        self,
        embedding_size,
        hidden_size,
        output_size,
        num_lstm_layers=1,
        bidirectional=False,
        vocab_size=256 + 3,
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = DPLSTM(
            embedding_size,
            hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.out_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, _ = self.lstm(x, hidden)
        x = x[:, -1, :]
        x = self.out_layer(x)
        return x


def init_name_classifier_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CharNNClassifier(
        settings.embedding_size,
        settings.hidden_size,
        len(ds.labels),
        settings.n_lstm_layers,
        settings.bidirectional_lstm,
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=settings.learning_rate)
    privacy_engine = PrivacyEngine(secure_mode=settings.secure_mode)
    model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        max_grad_norm=settings.max_per_sample_grad_norm,
        target_delta=settings.delta,
        target_epsilon=settings.epsilon,
        epochs=settings.epochs,
    )

    return model, optimizer, data_loader, privacy_engine
