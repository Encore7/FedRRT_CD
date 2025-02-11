import json
import os
from os import remove

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from src.data_loader import (
    add_client_drifted_dataset,
    load_client_data,
    remove_client_drifted_dataset,
)
from src.ml_models.net import Net, test, train
from src.ml_models.utils import get_weights, set_weights
from src.utils.logger import get_logger


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(
        self,
        client_number,
        local_epochs,
        learning_rate,
        momentum,
        num_batches_each_round,
        batch_size,
        drift_start_round,
        drift_end_round,
        drift_clients,
        abrupt_drift_labels_swap,
        drift_dataset_indexes_folder_path,
        remaining_dataset_folder_path,
        drifted_dataset_folder_path,
        dataset_folder_path,
        mode,
    ):
        super().__init__()
        self.net = Net()
        self.client_number = client_number
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.momentum = momentum
        self.num_batches_each_round = num_batches_each_round
        self.batch_size = batch_size
        self.drift_start_round = drift_start_round
        self.drift_end_round = drift_end_round
        self.drift_clients = drift_clients
        self.abrupt_drift_labels_swap = abrupt_drift_labels_swap
        self.client_drift_dataset_indexes_folder_path = os.path.join(
            drift_dataset_indexes_folder_path, f"client_{client_number}"
        )
        self.client_remaining_dataset_folder_path = os.path.join(
            remaining_dataset_folder_path, f"client_{client_number}"
        )
        self.client_drifted_dataset_folder_path = os.path.join(
            drifted_dataset_folder_path, f"client_{client_number}"
        )
        self.client_dataset_folder_path = os.path.join(
            dataset_folder_path, f"client_{client_number}"
        )
        self.mode = mode
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Configure logging
        self.logger = get_logger(f"{__name__}_Client_{client_number}", client_number)

        self.logger.info("Client %s initiated", self.client_number)

    def _get_dataset_folder_path(self, current_round):
        is_drift = False
        client_dataset_folder_path = self.client_dataset_folder_path

        if self.drift_start_round <= current_round < self.drift_end_round:
            if self.client_number in self.drift_clients:
                is_drift = True
                self.logger.info(
                    "Drift initiated by the client: %s", self.client_number
                )
        elif (
            current_round == self.drift_end_round
            and self.client_number in self.drift_clients
        ):
            if self.mode == "retraining-case":
                remove_client_drifted_dataset(
                    self.client_dataset_folder_path,
                    self.client_drift_dataset_indexes_folder_path,
                    self.client_remaining_dataset_folder_path,
                )
                client_dataset_folder_path = self.client_remaining_dataset_folder_path
            elif self.mode == "drift-case":
                add_client_drifted_dataset(
                    self.client_dataset_folder_path,
                    self.client_drift_dataset_indexes_folder_path,
                    self.client_drifted_dataset_folder_path,
                    self.abrupt_drift_labels_swap,
                )
                client_dataset_folder_path = self.client_drifted_dataset_folder_path

        elif (
            current_round > self.drift_end_round
            and self.client_number in self.drift_clients
        ):
            if self.mode == "retraining-case":
                client_dataset_folder_path = self.client_remaining_dataset_folder_path
            elif self.mode == "drift-case":
                client_dataset_folder_path = self.client_drifted_dataset_folder_path

        return client_dataset_folder_path, is_drift

    def fit(self, parameters, config):
        # Fetching configuration settings from the server for the fit operation (server.configure_fit)
        current_round = config.get("current_round", 0)

        client_dataset_folder_path, is_drift = self._get_dataset_folder_path(
            current_round
        )

        self.logger.info("config: %s", config)
        self.logger.info("Client %s | Round %s", self.client_number, current_round)

        results = {}

        # Unlearning initiated by the client
        # if current_round == -1 and self.client_number == -1:
        #     self.logger.info(
        #         "Unlearning initiated by the client: %s", self.client_number
        #     )
        #     results = {"unlearn_client_number": self.client_number}
        #     unlearn_client_number = self.client_number

        train_batches = load_client_data(
            "train_data",
            self.num_batches_each_round,
            self.batch_size,
            client_dataset_folder_path,
            is_drift,
            self.abrupt_drift_labels_swap,
            self.client_drift_dataset_indexes_folder_path,
        )
        val_batches = load_client_data(
            "val_data",
            self.num_batches_each_round,
            self.batch_size,
            client_dataset_folder_path,
            is_drift,
            self.abrupt_drift_labels_swap,
            self.client_drift_dataset_indexes_folder_path,
        )

        set_weights(self.net, parameters)

        train_results = train(
            self.net,
            train_batches,
            val_batches,
            self.local_epochs,
            self.lr,
            self.device,
            self.momentum,
        )

        results.update(train_results)

        self.logger.info("results %s", results)
        self.logger.info("dataset_length %s", len(train_batches.dataset))
        self.logger.info("learning_rate: %s", self.lr)
        self.logger.info("momentum: %s", self.momentum)

        return (
            get_weights(self.net),
            len(train_batches.dataset),
            results,
        )

    def _evaluate_model(self, parameters, is_drift, client_dataset_folder_path):
        set_weights(self.net, parameters)

        val_batches = load_client_data(
            "val_data",
            self.num_batches_each_round,
            self.batch_size,
            client_dataset_folder_path,
            False,
            is_drift,
            self.abrupt_drift_labels_swap,
            self.client_drift_dataset_indexes_folder_path,
        )
        loss, accuracy = test(self.net, val_batches, self.device)
        val_dataset_length = len(val_batches.dataset)

        self.logger.info("loss: %s", loss)
        self.logger.info("accuracy: %s", accuracy)
        self.logger.info("val_dataset_length: %s", val_dataset_length)

        return loss, accuracy, val_dataset_length

    def evaluate(self, parameters, config):
        is_drift = False
        self.logger.info("config: %s", config)

        current_round = config.get("current_round", 0)

        client_dataset_folder_path, is_drift = self._get_dataset_folder_path(
            current_round
        )

        loss, accuracy, val_dataset_length = self._evaluate_model(
            parameters,
            is_drift,
            client_dataset_folder_path,
        )

        return (
            loss,
            val_dataset_length,
            {"accuracy": accuracy},
        )


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    partition_id = context.node_config["partition-id"]
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    momentum = context.run_config["momentum"]
    num_batches_each_round = context.run_config["num-batches-each-round"]
    batch_size = context.run_config["batch-size"]
    drift_start_round = context.run_config["drift-start-round"]
    drift_end_round = context.run_config["drift-end-round"]
    drift_clients = json.loads(context.run_config["drift-clients"])
    abrupt_drift_labels_swap = json.loads(
        context.run_config["abrupt-drift-labels-swap"]
    )
    drift_dataset_indexes_folder_path = context.run_config[
        "drift-dataset-indexes-folder-path"
    ]
    remaining_dataset_folder_path = context.run_config["remaining-dataset-folder-path"]
    drifted_dataset_folder_path = context.run_config["drifted-dataset-folder-path"]
    dataset_folder_path = context.run_config["dataset-folder-path"]
    mode = context.run_config["mode"]

    # Return Client instance
    return FlowerClient(
        partition_id,
        local_epochs,
        learning_rate,
        momentum,
        num_batches_each_round,
        batch_size,
        drift_start_round,
        drift_end_round,
        drift_clients,
        abrupt_drift_labels_swap,
        drift_dataset_indexes_folder_path,
        remaining_dataset_folder_path,
        drifted_dataset_folder_path,
        dataset_folder_path,
        mode,
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
