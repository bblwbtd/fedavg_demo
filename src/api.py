import shutil
import os
import uuid
import threading

from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from torch import nn

from src.model import init_name_classifier_model
from src.peer import Peer
from config import settings

app = FastAPI()
model, optimizer, data_loader, privacy_engine = init_name_classifier_model()
peer = Peer(
    model=model,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optimizer,
    train_loader=data_loader,
    privacy_engine=privacy_engine,
    bootstrap_peers=settings.bootstrap_peers,
)


class InitParams(BaseModel):
    sync_epoch = 10
    max_epoch = 100


def __start_training_thread():
    threading.Thread(target=peer.train).start()


@app.post("/sync/{epoch}", status_code=200)
def sync_state(epoch: int, state: UploadFile):
    print("Receive state from other peer.")

    state_dir = os.path.join(settings.temp_dir, "states", str(epoch))
    if not os.path.exists(state_dir):
        os.makedirs(state_dir)
    with open(os.path.join(state_dir, f"{uuid.uuid4()}.pth"), "wb") as file:
        shutil.copyfileobj(state.file, file)


@app.post("/start", status_code=201)
async def start_training():
    if not peer.is_training:
        print("Start training")
        __start_training_thread()
    else:
        raise HTTPException(400, "peer is training")
