import json
import os

import numpy as np
import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset, Subset
from torchvision import transforms


class DataLoader:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        if self.dataset_name == "mnist":
            self.pytorch_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
        else:
            raise ValueError("Unsupported dataset")

    def _apply_transforms(self, batch):
        batch["image"] = [self.pytorch_transforms(img) for img in batch["image"]]
        return batch

    def _load_partition(self, num_clients: int, alpha: float, split: str):
        """
        Loads a federated partition for the given split ("train" or "test").

        Parameters:
            num_clients (int): Number of clients/partitions.
            alpha (float): Dirichlet parameter controlling the degree of heterogeneity.
            split (str): Which split to load ("train" or "test").
        """
        if self.dataset_name == "mnist":
            partition_by = "label"
        else:
            raise ValueError("Unknown dataset")

        # for non iid data
        # partitioner = DirichletPartitioner(
        #     num_partitions=num_clients,
        #     partition_by=partition_by,
        #     alpha=alpha,
        #     self_balancing=True,
        # )

        federated_dataset = FederatedDataset(
            dataset=self.dataset_name, partitioners={split: num_clients}
        )

        return federated_dataset

    def save_datasets(
        self,
        num_clients: int,
        alpha: float,
        dataset_folder_path: str,
    ):
        # Create federated datasets for both train and test splits.
        federated_train_dataset = self._load_partition(
            num_clients, alpha, split="train"
        )
        federated_test_dataset = self._load_partition(num_clients, alpha, split="test")

        for client_id in range(num_clients):
            client_dir = os.path.join(dataset_folder_path, f"client_{client_id}")
            os.makedirs(client_dir, exist_ok=True)

            # Load each client's partition and apply the same transform.
            partition_train = federated_train_dataset.load_partition(
                client_id
            ).with_transform(self._apply_transforms)
            partition_test = federated_test_dataset.load_partition(
                client_id
            ).with_transform(self._apply_transforms)

            # Define file paths.
            train_path = os.path.join(client_dir, "train_data.pt")
            val_path = os.path.join(client_dir, "val_data.pt")

            # Save the partitions using torch.save.
            torch.save(partition_train, train_path)
            torch.save(partition_test, val_path)


def load_client_data(
    file_name: str,
    num_batches_each_round: int,
    batch_size: int,
    client_dataset_folder_path: str,
    is_drift: bool = False,
    abrupt_drift_labels_swap=None,
    client_drift_dataset_indexes_folder_path: str = None,
):

    client_data_file_path = os.path.join(client_dataset_folder_path, f"{file_name}.pt")

    dataset = torch.load(client_data_file_path, weights_only=False)
    dataset_length = len(dataset)
    total_samples = num_batches_each_round * batch_size

    if dataset_length < total_samples:
        raise ValueError(
            f"Dataset size ({dataset_length}) is smaller than the requested number of samples ({total_samples})."
        )

    # Randomly select indices without replacement
    indices = np.random.choice(dataset_length, total_samples, replace=False)
    drift_dataset_indexes = []

    if is_drift:
        # Prepare a list to hold the swapped data samples
        swapped_data = []
        # Loop over each selected index and swap labels if needed
        for idx in indices:
            # Get the sample (assumed to be in the format (image, label))
            image = dataset[int(idx)]["image"]
            label = dataset[int(idx)]["label"]

            # Check each swap rule
            for rule in abrupt_drift_labels_swap:
                if label == rule["label1"]:
                    label = rule["label2"]
                    drift_dataset_indexes.append(int(idx))
                elif label == rule["label2"]:
                    label = rule["label1"]
                    drift_dataset_indexes.append(int(idx))
            # Append the (possibly modified) sample to our swapped data list
            swapped_data.append({"image": image, "label": label})

        # Save the swapped dataset.
        # For example, we append '_drift' to the filename.

        os.makedirs(client_drift_dataset_indexes_folder_path, exist_ok=True)
        client_drift_dataset_indexes_file_path = os.path.join(
            client_drift_dataset_indexes_folder_path, f"{file_name}.json"
        )
        # Save the drift indexes to file
        if os.path.exists(client_drift_dataset_indexes_file_path):
            with open(
                client_drift_dataset_indexes_file_path, "r", encoding="utf-8"
            ) as file:
                existing_indexes = json.load(file)
            drift_dataset_indexes = existing_indexes + drift_dataset_indexes

        with open(
            client_drift_dataset_indexes_file_path, "w", encoding="utf-8"
        ) as file:
            json.dump(drift_dataset_indexes, file)

        # Create a new DataLoader using the swapped data
        data_loader = TorchDataLoader(swapped_data, batch_size=batch_size, shuffle=True)
    else:
        subset = Subset(dataset, indices)
        # Create a DataLoader over the subset
        data_loader = TorchDataLoader(subset, batch_size=batch_size, shuffle=True)

    return data_loader


def remove_client_drifted_dataset(
    client_dataset_folder_path,
    client_drift_dataset_indexes_folder_path,
    client_remaining_dataset_folder_path,
):
    os.makedirs(client_remaining_dataset_folder_path, exist_ok=True)

    for file_name in os.listdir(client_drift_dataset_indexes_folder_path):
        file_name_without_ext = os.path.splitext(file_name)[0]

        client_remaining_dataset_file_path = os.path.join(
            client_remaining_dataset_folder_path, f"{file_name_without_ext}.pt"
        )
        # Load client dataset
        client_data_file_path = os.path.join(
            client_dataset_folder_path, f"{file_name_without_ext}.pt"
        )

        client_dataset = torch.load(client_data_file_path, weights_only=False)

        if file_name_without_ext == "train_data":

            client_drift_dataset_indexes_file_path = os.path.join(
                client_drift_dataset_indexes_folder_path,
                f"{file_name_without_ext}.json",
            )

            with open(
                client_drift_dataset_indexes_file_path, "r", encoding="utf-8"
            ) as file:
                drift_dataset_indexes = set(json.load(file))

            # Filter out the drifted dataset indices
            remaining_dataset = [
                data
                for idx, data in enumerate(client_dataset)
                if idx not in drift_dataset_indexes
            ]

            # Save the remaining dataset
            torch.save(remaining_dataset, client_remaining_dataset_file_path)
        else:
            torch.save(client_dataset, client_remaining_dataset_file_path)


def add_client_drifted_dataset(
    client_dataset_folder_path,
    client_drift_dataset_indexes_folder_path,
    client_drifted_dataset_folder_path,
    abrupt_drift_labels_swap,
):
    os.makedirs(client_drifted_dataset_folder_path, exist_ok=True)

    for file_name in os.listdir(client_drift_dataset_indexes_folder_path):

        file_name_without_ext = os.path.splitext(file_name)[0]

        client_data_file_path = os.path.join(
            client_dataset_folder_path, f"{file_name_without_ext}.pt"
        )

        client_dataset = list(torch.load(client_data_file_path, weights_only=False))

        client_drift_dataset_indexes_file_path = os.path.join(
            client_drift_dataset_indexes_folder_path, f"{file_name_without_ext}.json"
        )

        with open(
            client_drift_dataset_indexes_file_path, "r", encoding="utf-8"
        ) as file:
            drift_dataset_indexes = set(json.load(file))

        # Loop over each selected index and swap labels if needed
        for drift_dataset_index in drift_dataset_indexes:
            # Get the sample (assumed to be in the format (image, label))
            label = client_dataset[int(drift_dataset_index)]["label"]

            # Check each swap rule
            for rule in abrupt_drift_labels_swap:
                if label == rule["label1"]:
                    client_dataset[int(drift_dataset_index)]["label"] = rule["label2"]
                elif label == rule["label2"]:
                    client_dataset[int(drift_dataset_index)]["label"] = rule["label1"]

        client_drifted_dataset_file_path = os.path.join(
            client_drifted_dataset_folder_path, f"{file_name_without_ext}.pt"
        )
        torch.save(client_dataset, client_drifted_dataset_file_path)
