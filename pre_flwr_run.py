import subprocess

from src.scripts.helper import clear_folder_contents
from src.scripts.prepare_dataset import prepare_dataset


def pre_flwr_run():
    print("Running pre-flwr setup...")
    logs_folder_path = "log"
    clients_dataset_folder_path = "src/clients_dataset"
    pyproject_toml_file_path = "pyproject.toml"

    # clearing logs before running the experiment
    clear_folder_contents(logs_folder_path)

    # preparing the dataset

    prepare_dataset(pyproject_toml_file_path, clients_dataset_folder_path)


if __name__ == "__main__":
    pre_flwr_run()

    subprocess.run(["flwr", "run"])
