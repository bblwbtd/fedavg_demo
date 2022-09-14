import shutil
import os
import uuid

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

from .peer import Peer
from config import settings

app = FastAPI()
peer = Peer(bootstrap_peers=settings.bootstrap_peers)


class InitParams(BaseModel):
    sync_epoch = 10
    max_epoch = 100


@app.post("/sync/{epoch}")
def sync_state(epoch: int, state: UploadFile):
    dir = os.path.join("temp", "states", epoch)
    os.makedirs(dir)
    with open(os.path.join(dir, f"{uuid.uuid4()}.pth"), "w") as file:
        shutil.copyfileobj(state.file, file)

    _, _, files = os.walk(dir)
    if len(files) == len(peer.bootstrap_peers) and epoch == peer.current_epoch:
        peer.update_model(dir)
