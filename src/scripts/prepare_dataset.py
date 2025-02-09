import os

import toml

from src.data_loader import DataLoader
from src.scripts.helper import clear_folder_contents


def prepare_dataset(pyproject_path: str, clients_dataset_folder_path: str):
    if os.path.exists(pyproject_path):
        with open(pyproject_path, "r") as file:
            pyproject_data = toml.load(file)

        config = (
            pyproject_data.get("tool", {})
            .get("flwr", {})
            .get("app", {})
            .get("config", {})
        )
        prepare_dataset = config.get("prepare-dataset", None)

        if not prepare_dataset:
            return

        num_of_clients = config.get("num-of-clients", None)
        num_server_rounds = config.get("num-server-rounds", None)
        num_batches_each_round = config.get("num-batches-each-round", None)
        batch_size = config.get("batch-size", None)
        dataset_name = config.get("dataset-name", None)
        alpha = config.get("data-loader-alpha", None)
        unlearning_trigger_client = config.get("unlearning-trigger-client", None)

        clear_folder_contents(clients_dataset_folder_path)

        dataloader = DataLoader(dataset_name=str(dataset_name))

        dataloader.save_datasets(
            num_clients=num_of_clients,
            num_rounds=num_server_rounds,
            num_batches_each_round=num_batches_each_round,
            batch_size=batch_size,
            alpha=alpha,
            clients_dataset_folder_path=clients_dataset_folder_path,
            unlearning_trigger_client=unlearning_trigger_client,
        )
