import json
import os
import time
from functools import reduce
from typing import List, Tuple

import numpy as np
from flwr.common import (
    Context,
    FitIns,
    Metrics,
    NDArrays,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate_inplace

from src.ml_models.net import Net
from src.ml_models.utils import get_weights


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def custom_aggregate(results: list[tuple[NDArrays, float]]) -> NDArrays:
    """
    Aggregate model parameters with custom weightages.
    Parameters:
    results: List of tuples, where each tuple contains:
        - NDArrays: Model parameters
        - float: Weightage for this model (e.g., 0.1 for 10%, 0.9 for 90%)
    Returns:
    NDArrays: Aggregated model parameters.
    """
    # Ensure weightages sum up to 1 for valid aggregation
    total_weight = sum(weight for _, weight in results)
    if not np.isclose(total_weight, 1.0):
        raise ValueError("Weightages must sum up to 1.0")
    # Multiply model weights by their respective weightage
    weighted_weights = [
        [layer * weight for layer in weights] for weights, weight in results
    ]
    # Sum up the weighted layers across models
    aggregated_weights: NDArrays = [
        reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)
    ]
    return aggregated_weights


class UnlearningFedAvg(FedAvg):
    def __init__(
        self,
        num_of_clients,
        num_server_rounds,
        mode,
        drift_start_round,
        incremental_drift_rounds,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_of_clients = num_of_clients
        self.client_plot = {}
        self.num_server_rounds = num_server_rounds
        self.mode = mode
        self.drift_start_round = drift_start_round
        self.incremental_drift_rounds = incremental_drift_rounds

    def configure_fit(self, server_round, parameters, client_manager):
        # Waiting till all clients are connected
        client_manager.wait_for(self.num_of_clients)

        config = {
            "current_round": server_round,
        }
        print("fit_ins.config", config)

        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        client_fit_pairs = []
        for client in clients:
            client_fit_pairs.append((client, fit_ins))

        # Return client/config pairs
        return client_fit_pairs

    def configure_evaluate(self, server_round, parameters, client_manager):
        # Calling the parent class's configure_evaluate method
        client_evaluate_pairs = super().configure_evaluate(
            server_round, parameters, client_manager
        )

        for _, evaluate_ins in client_evaluate_pairs:
            # Add the current round to the config
            evaluate_ins.config["current_round"] = server_round

        return client_evaluate_pairs

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}

        if (
            (
                self.incremental_drift_rounds
                and server_round in map(int, self.incremental_drift_rounds.keys())
            )
            or server_round == self.drift_start_round
        ) and (self.mode == "retraining-case" or self.mode == "rapid-retraining-case"):
            re_init_parameters = get_weights(Net())
            return ndarrays_to_parameters(re_init_parameters), metrics_aggregated

        filtered_results = []
        aux_models_classifier_layer_list = []
        aux_last_layer_weights_index = None
        aux_last_layer_bias_index = None
        for client_proxy, fit_res in results:
            print("fit_res.metrics", fit_res.metrics)

            client_number = fit_res.metrics["client_number"]

            if client_number not in self.client_plot:
                self.client_plot[client_number] = {
                    "global_accuracy": [],
                    "global_loss": [],
                    "local_accuracy": [],
                    "local_loss": [],
                }
            self.client_plot[client_number]["local_accuracy"].append(
                fit_res.metrics["val_accuracy"]
            )
            self.client_plot[client_number]["local_loss"].append(
                fit_res.metrics["val_loss"]
            )

            if "mode" in fit_res.metrics and (
                fit_res.metrics["mode"] == "fedau-case"
                or fit_res.metrics["mode"] == "fluid-case"
            ):
                print("Aggregating aux models fit")
                fit_res_parameters_ndarray = parameters_to_ndarrays(fit_res.parameters)
                aux_last_layer_weights_index = fit_res.metrics[
                    "aux_last_layer_weights_index"
                ]
                aux_last_layer_bias_index = fit_res.metrics["aux_last_layer_bias_index"]

                aux_models_classifier_layer_list.append(fit_res_parameters_ndarray[-2:])
                fit_res.parameters = ndarrays_to_parameters(
                    fit_res_parameters_ndarray[:-2]
                )

            filtered_results.append((client_proxy, fit_res))

        aggregated_ndarrays = aggregate_inplace(filtered_results)

        if aux_models_classifier_layer_list:
            print("Aggregating aux models")
            aux_models_classifier_layer_list_len = len(aux_models_classifier_layer_list)
            aux_model_classifier_layer_aggregated = custom_aggregate(
                [
                    (
                        aux_models_classifier_layer,
                        1 / aux_models_classifier_layer_list_len,
                    )
                    for aux_models_classifier_layer in aux_models_classifier_layer_list
                ]
            )

            aggregated_ndarrays[
                aux_last_layer_weights_index : aux_last_layer_bias_index + 1
            ] = custom_aggregate(
                [
                    (
                        aggregated_ndarrays[
                            aux_last_layer_weights_index : aux_last_layer_bias_index + 1
                        ],
                        1.03,
                    ),
                    (aux_model_classifier_layer_aggregated, -0.03),
                ]
            )

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        for _, eval_res in results:
            client_number = eval_res.metrics["client_number"]
            accuracy = eval_res.metrics["accuracy"]
            loss = eval_res.loss

            # Append accuracy and loss
            self.client_plot[client_number]["global_accuracy"].append(accuracy)
            self.client_plot[client_number]["global_loss"].append(loss)

        if server_round == self.num_server_rounds:
            with open(
                os.path.join("src/plots", f"results_{self.mode}.json"),
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(self.client_plot, file)

        return super().aggregate_evaluate(server_round, results, failures)


def server_fn(context: Context):
    print("context.node_config", context)
    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    strategy = UnlearningFedAvg(
        fraction_evaluate=float(context.run_config["fraction-evaluate"]),
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
        num_of_clients=int(context.run_config["num-of-clients"]),
        num_server_rounds=int(context.run_config["num-server-rounds"]),
        mode=context.run_config["mode"],
        drift_start_round=int(context.run_config["drift-start-round"]),
        incremental_drift_rounds=json.loads(
            context.run_config["incremental-drift-rounds"]
        ),
    )
    config = ServerConfig(num_rounds=int(context.run_config["num-server-rounds"]))

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
