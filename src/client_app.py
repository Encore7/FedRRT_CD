import json
import os

import numpy as np
import torch
import torch.nn as nn
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from src.data_loader import (
    add_client_drifted_dataset,
    load_client_data,
    remove_client_drifted_dataset,
)
from src.ml_models.net import Net, rapid_train, test, train
from src.ml_models.utils import get_weights, set_weights
from src.utils.logger import get_logger


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(
        self,
        client_number,
        local_epochs,
        learning_rate,
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
            if (
                self.mode == "retraining-case"
                or self.mode == "rapid-retraining-case"
                or self.mode == "fedau-case"
                or self.mode == "fluid-case"
            ):
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
            if (
                self.mode == "retraining-case"
                or self.mode == "rapid-retraining-case"
                or self.mode == "fedau-case"
                or self.mode == "fluid-case"
            ):
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

        train_batches = load_client_data(
            "train_data",
            self.num_batches_each_round,
            self.batch_size,
            client_dataset_folder_path,
            True,
            is_drift,
            self.abrupt_drift_labels_swap,
            self.client_drift_dataset_indexes_folder_path,
        )
        val_batches = load_client_data(
            "val_data",
            self.num_batches_each_round,
            self.batch_size,
            client_dataset_folder_path,
            True,
            is_drift,
            self.abrupt_drift_labels_swap,
            self.client_drift_dataset_indexes_folder_path,
        )

        set_weights(self.net, parameters)

        if (
            self.mode == "rapid-retraining-case" or self.mode == "fluid-case"
        ) and current_round >= self.drift_end_round:
            train_results = rapid_train(
                self.net,
                train_batches,
                val_batches,
                self.local_epochs,
                self.lr,
                self.device,
                self.batch_size,
            )
        else:
            train_results = train(
                self.net,
                train_batches,
                val_batches,
                self.local_epochs,
                self.lr,
                self.device,
            )

        results.update(train_results)

        self.logger.info("results %s", results)
        self.logger.info("dataset_length %s", len(train_batches.dataset))
        self.logger.info("learning_rate: %s", self.lr)

        # Training aux model for the federated AU case

        if (
            (self.mode == "fedau-case" or self.mode == "fluid-case")
            and self.client_number in self.drift_clients
            and (self.drift_start_round <= current_round < self.drift_end_round)
        ):
            aux_model = Net()
            set_weights(aux_model, parameters)

            aux_model.fc3 = nn.Linear(
                aux_model.fc3.in_features, aux_model.fc3.out_features
            )

            aux_train_batches = load_client_data(
                "train_data",
                self.num_batches_each_round,
                self.batch_size,
                client_dataset_folder_path,
                False,
                False,
                self.abrupt_drift_labels_swap,
                self.client_drift_dataset_indexes_folder_path,
                self.mode,
            )

            train(
                aux_model,
                aux_train_batches,
                val_batches,
                self.local_epochs,
                self.lr,
                self.device,
                0.9,
            )

            aux_last_layer_index = []
            aux_last_layer_array = [
                val.cpu().numpy()
                for idx, (name, val) in enumerate(aux_model.state_dict().items())
                if "fc3" in name and aux_last_layer_index.append(idx) is None
            ]

            results.update(
                {
                    "mode": self.mode,
                    "aux_last_layer_weights_index": aux_last_layer_index[0],
                    "aux_last_layer_bias_index": aux_last_layer_index[1],
                }
            )

            weights = get_weights(self.net)
            weights.append(aux_last_layer_array[0])
            weights.append(aux_last_layer_array[1])

            return (
                weights,
                len(train_batches.dataset),
                results,
            )

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
            {
                "accuracy": accuracy,
                "client_number": self.client_number,
            },
        )


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    partition_id = context.node_config["partition-id"]
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
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
