import os

import numpy as np
import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from PIL import ImageDraw
from torch.utils.data import DataLoader as TorchDataLoader
from torchvision import transforms


class DataLoader:
    def __init__(
        self,
        dataset_name: str,
    ):
        self.dataset_name = dataset_name
        if self.dataset_name == "mnist":
            self.pytorch_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )

    def _apply_transforms(self, batch):
        batch["image"] = [self.pytorch_transforms(img) for img in batch["image"]]
        return batch

    def _load_partition(self, num_clients: int, alpha: float):
        if self.dataset_name == "mnist":
            partition_by = "label"
        else:
            raise ValueError("Unknown dataset")

        partitioner = DirichletPartitioner(
            num_partitions=num_clients,
            partition_by=partition_by,
            alpha=alpha,
            self_balancing=True,
        )
        return FederatedDataset(
            dataset=self.dataset_name, partitioners={"train": partitioner}
        )

    def save_datasets(
        self,
        num_clients: int,
        num_rounds: int,
        num_batches_each_round: int,
        batch_size: int,
        alpha: float,
        clients_dataset_folder_path: str,
        unlearning_trigger_client: int,
    ):
        fds = self._load_partition(num_clients, alpha)
        for client_id in range(num_clients):
            client_dir = os.path.join(
                clients_dataset_folder_path, f"client_{client_id}"
            )
            os.makedirs(client_dir, exist_ok=True)

            partition = fds.load_partition(client_id)

            partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
            partition_train_test = partition_train_test.with_transform(
                self._apply_transforms
            )

            valloader = TorchDataLoader(
                partition_train_test["test"], batch_size=batch_size
            )

            val_path = os.path.join(client_dir, "val_data.pt")
            torch.save(list(valloader), val_path)

            # Train batches for each round
            train_data = partition_train_test["train"]
            train_indices = set(np.arange(len(train_data)))

            for round_num in range(num_rounds):
                round_batches = []
                used_indices = set()

                for _ in range(num_batches_each_round):
                    available_indices = list(train_indices - used_indices)
                    if len(available_indices) < batch_size:
                        raise ValueError(
                            "Not enough unique data to create a batch without duplicates"
                        )

                    selected_indices = np.random.choice(
                        available_indices, size=batch_size, replace=False
                    )
                    used_indices.update(selected_indices)

                    train_batch = {
                        key: (
                            torch.stack(
                                [train_data[int(idx)][key] for idx in selected_indices]
                            )
                            if isinstance(train_data[0][key], torch.Tensor)
                            else torch.tensor(
                                [train_data[int(idx)][key] for idx in selected_indices]
                            )
                        )
                        for key in train_data[0]
                    }
                    round_batches.append(train_batch)

                round_path = os.path.join(
                    client_dir, f"train_data_for_round_{round_num + 1}.pt"
                )
                torch.save(round_batches, round_path)


def load_client_data(type: str, client_id: int, current_round: int = None):
    client_dir = os.path.join("src", "clients_dataset", f"client_{client_id}")

    if type == "val":
        # Load validation dataset
        val_path = os.path.join(client_dir, "val_data.pt")
        return torch.load(val_path, weights_only=False)

    if type == "train":
        # Load train dataset for the specified round
        round_path = os.path.join(
            client_dir, f"train_data_for_round_{current_round}.pt"
        )
        return torch.load(round_path, weights_only=False)


def dataset_length(dataset):
    return len(dataset[0]["image"]) * (len(dataset) - 1) + len(dataset[-1]["image"])
