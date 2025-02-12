import os
import subprocess

import toml

from src.scripts.helper import clear_folder_contents
from src.scripts.prepare_dataset import prepare_dataset


def pre_flwr_run():
    print("Running pre-flwr setup...")
    logs_folder_path = "log"
    pyproject_toml_file_path = "pyproject.toml"

    # clearing logs before running the experiment
    clear_folder_contents(logs_folder_path)

    with open(pyproject_toml_file_path, "r", encoding="utf-8") as file:
        pyproject_data = toml.load(file)

    config = (
        pyproject_data.get("tool", {}).get("flwr", {}).get("app", {}).get("config", {})
    )

    mode = config.get("mode")
    file_path = "scr/plots/results_" + mode + ".json"
    if os.path.exists(file_path):
        os.remove(file_path)

    # preparing the dataset

    prepare_dataset(pyproject_toml_file_path, config)


if __name__ == "__main__":
    pre_flwr_run()

    subprocess.run(["flwr", "run"])
