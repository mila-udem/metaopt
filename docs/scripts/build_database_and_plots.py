import argparse
import os
import shutil
import subprocess
import sys

from orion.client import get_experiment
from orion.core.cli.db.rm import process_exp_rm
from orion.core.utils.singleton import update_singletons
from orion.core.worker.trial import Trial
from orion.storage.base import get_storage, setup_storage

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DOC_SRC_DIR = os.path.join(ROOT_DIR, "..", "src")
os.chdir(DOC_SRC_DIR)

EXAMPLE_DIR = os.path.abspath("../../examples")

MAIN_DB_HOST = f"{EXAMPLE_DIR}/db.pkl"
BASE_DB_HOST = f"{EXAMPLE_DIR}/base_db.pkl"
TMP_DB_HOST = f"{EXAMPLE_DIR}/tmp.pkl"


# CP base db to database.pkl (overwrite database.pkl)

names = {
    "code_1_python_api": ["random-rosenbrock", "tpe-rosenbrock"],
    # "code_2_computation_time": ["tpe-cifar10-comp-time"],
    "code_2_hyperband_checkpoint": ["hyperband-cifar10"],
}


def execute(example):
    command = f"python {EXAMPLE_DIR}/tutorials/{example}.py"
    os.chdir(f"{EXAMPLE_DIR}")
    print("executing", command)
    process = subprocess.Popen(command.split(" "))
    return_code = process.wait()
    tmp_files = "tmp_" + "_".join(example.split("_")[2:])
    print("removing tmp files", tmp_files)
    shutil.rmtree(tmp_files, ignore_errors=True)
    os.chdir(DOC_SRC_DIR)
    print("done")
    return return_code


def prepare_dbs():

    if os.path.exists(MAIN_DB_HOST):
        print("Removing", MAIN_DB_HOST)
        os.remove(MAIN_DB_HOST)

    print("Copying", BASE_DB_HOST, "->", TMP_DB_HOST)
    shutil.copy(BASE_DB_HOST, TMP_DB_HOST)


def setup_tmp_storage(host):
    # Clear singletons
    update_singletons()

    setup_storage(
        storage={
            "type": "legacy",
            "database": {
                "type": "pickleddb",
                "host": host,
            },
        }
    )

    return get_storage()


def load_data(host):
    print("Loading data from", host)
    storage = setup_tmp_storage(host)
    data = {"experiments": {}, "trials": {}}
    for experiment in storage.fetch_experiments({}):
        data["experiments"][experiment["_id"]] = experiment
        data["trials"][experiment["_id"]] = storage.fetch_trials(uid=experiment["_id"])

    return data


def copy_data(data, host=TMP_DB_HOST):
    print("Copying data to", host)
    storage = setup_tmp_storage(host)
    for exp_id, experiment in data["experiments"].items():
        del experiment["_id"]
        storage.create_experiment(experiment)
        assert exp_id != experiment["_id"]
        trials = []
        for trial in data["trials"][exp_id]:
            trial.experiment = experiment["_id"]
            trials.append(trial.to_dict())
        storage._db.write("trials", trials)


def plot_exps(experiment_names, host=TMP_DB_HOST):
    print("Plotting experiments from", host)
    storage = setup_tmp_storage(host)
    # Plot exps
    for experiment_name in experiment_names:
        print("   ", experiment_name)
        experiment = get_experiment(experiment_name)
        for plot in ["regret", "lpi", "partial_dependencies", "parallel_coordinates"]:
            experiment.plot(kind=plot).write_html(
                f"_static/{experiment.name}_{plot}.html"
            )


def main(argv=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("examples", nargs="*", choices=names.keys())
    options = parser.parse_args(argv)

    if options.examples:
        run = {name for name in options.examples if name in names}
    else:
        run = set(names.keys())

    prepare_dbs()

    for example, example_experiments in names.items():

        example_db_host = f"{EXAMPLE_DIR}/{example}_db.pkl"

        if example in run:
            execute(example)
            print("Moving", MAIN_DB_HOST, "->", example_db_host)
            os.rename(MAIN_DB_HOST, example_db_host)

        data = load_data(example_db_host)
        copy_data(data, TMP_DB_HOST)
        plot_exps(example_experiments, TMP_DB_HOST)

    print("Moving", TMP_DB_HOST, "->", MAIN_DB_HOST)
    os.rename(TMP_DB_HOST, MAIN_DB_HOST)


if __name__ == "__main__":
    main()
