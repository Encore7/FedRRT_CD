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
        dataset_name = config.get("dataset-name", None)
        alpha = config.get("data-loader-alpha", None)

        clear_folder_contents(clients_dataset_folder_path)

        dataloader = DataLoader(dataset_name=str(dataset_name))

        dataloader.save_datasets(
            num_clients=num_of_clients,
            alpha=alpha,
            clients_dataset_folder_path=clients_dataset_folder_path,
        )
