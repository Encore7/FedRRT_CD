import os

from src.data_loader import DataLoader
from src.scripts.helper import clear_folder_contents


def prepare_dataset(pyproject_path: str, config: dict):
    if os.path.exists(pyproject_path):

        clear_folder_contents(config.get("drift-dataset-indexes-folder-path", None))
        clear_folder_contents(config.get("remaining-dataset-folder-path", None))
        clear_folder_contents(config.get("drifted-dataset-folder-path", None))

        should_prepare_dataset = config.get("prepare-dataset", None)

        if not should_prepare_dataset:
            return

        num_of_clients = config.get("num-of-clients", None)
        dataset_name = config.get("dataset-name", None)
        alpha = config.get("data-loader-alpha", None)

        dataset_folder_path = config.get("dataset-folder-path", None)

        clear_folder_contents(dataset_folder_path)

        dataloader = DataLoader(dataset_name=str(dataset_name))

        dataloader.save_datasets(
            num_clients=num_of_clients,
            alpha=alpha,
            dataset_folder_path=dataset_folder_path,
        )
