# FedAvg Demo
Simple, stupid, but it works!

This is a demo project of FedAvg algorithm. Unlike other FedAvg Demo, the peer
of this demo sync their model state without a central server.

## Environment Requirement

- Python >= 3.10
- Pipenv

## Quickstart

### 1. Install dependencies

We use pipenv to manage the dependencies. Run the following command to install
dependencies.

```shell
pipenv install
```

### 2. Update settings

The file [settings.yaml](settings.yaml) contains all the settings of this program.
You need to add other peer's url the *bootstrap_peers* section. For example:
```yaml
boostrap_peers:
 - http://localhost:9999
```

### 3. Run the program

Use this command to run the program.
```shell
pipenv run python main.py
```

Then you can visit the swagger document (http://server_host/docs) and use the start API to start the training.
You can start the training on another peer. Each peer will sync their model state with
each other periodically. Finally, the model will be exported to temp directory once
the training is finished.
