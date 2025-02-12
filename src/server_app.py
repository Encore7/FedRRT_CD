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


class UnlearningFedAvg(FedAvg):
    def __init__(self, num_of_clients, **kwargs):
        super().__init__(**kwargs)

        self.num_of_clients = num_of_clients

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

        filtered_results = []
        for client_proxy, fit_res in results:
            print("fit_res.metrics", fit_res.metrics)

            filtered_results.append((client_proxy, fit_res))

        parameters_aggregated = ndarrays_to_parameters(
            aggregate_inplace(filtered_results)
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(self, server_round, results, failures):

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
    )
    config = ServerConfig(num_rounds=int(context.run_config["num-server-rounds"]))

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
