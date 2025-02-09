import os

import numpy as np
import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Subset
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
        clients_dataset_folder_path: str,
    ):
        # Create federated datasets for both train and test splits.
        federated_train_dataset = self._load_partition(
            num_clients, alpha, split="train"
        )
        federated_test_dataset = self._load_partition(num_clients, alpha, split="test")

        for client_id in range(num_clients):
            client_dir = os.path.join(
                clients_dataset_folder_path, f"client_{client_id}"
            )
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
    type: str, client_number: int, num_batches_each_round: int, batch_size: int
):
    client_dir = os.path.join("src", "clients_dataset", f"client_{client_number}")

    if type == "val":
        # Load validation dataset
        path = os.path.join(client_dir, "val_data.pt")

    elif type == "train":
        # Load train dataset for the specified round
        path = os.path.join(client_dir, "train_data.pt")

    dataset = torch.load(path, weights_only=False)
    dataset_length = len(dataset)
    total_samples = num_batches_each_round * batch_size

    if dataset_length < total_samples:
        raise ValueError(
            f"Dataset size ({dataset_length}) is smaller than the requested number of samples ({total_samples})."
        )

    # Randomly select indices without replacement
    indices = np.random.choice(dataset_length, total_samples, replace=False)
    subset = Subset(dataset, indices)

    # Create a DataLoader over the subset
    data_loader = TorchDataLoader(subset, batch_size=batch_size, shuffle=True)
    return data_loader
