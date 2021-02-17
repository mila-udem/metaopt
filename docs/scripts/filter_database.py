"""
Script to turn the database ``examples/plotting/database.pkl`` into a clean
version ``examples/base_db.pkl`` for the examples.
"""
import pprint
import shutil

from orion.client import get_experiment
from orion.core.io.orion_cmdline_parser import OrionCmdlineParser
from orion.storage.base import get_storage, setup_storage


shutil.copy("./examples/plotting/database.pkl", "./examples/base_db.pkl")

setup_storage(
    dict(
        type="legacy",
        database=dict(type="pickleddb", host="./examples/base_db.pkl"),
    )
)

filter_exps = {
    ("lateral-view-pa4", 1): "2-dim-exp",
    ("lateral-view-dualnet2", 1): "2-dim-shape-exp",
    ("lateral-view-multitask2", 1): "4-dim-cat-shape-exp",
    ("lateral-view-multitask3", 1): "3-dim-cat-shape-exp",
}

storage = get_storage()


def update_dropout(experiment_config):
    metadata = experiment_config["metadata"]
    user_script = metadata.get("user_script", "")
    user_args = metadata.get("user_args", [])
    try:
        index = user_args.index("--dropout")
    except ValueError:
        print(
            f"No dropout for {experiment_config['metadata']}-v{experiment_config['version']}"
        )
        return

    user_args[index + 1] = (
        user_args[index + 1]
        .replace("5,", "0.5,")
        .replace(", discrete=True", ", precision=1")
    )
    cmdline_parser = OrionCmdlineParser(allow_non_existing_files=True)
    cmdline_parser.parse([user_script] + user_args)
    metadata["parser"] = cmdline_parser.get_state_dict()
    experiment_config["space"] = metadata["priors"] = dict(cmdline_parser.priors)

    # Update config in db
    storage.update_experiment(uid=experiment_config["_id"], **experiment_config)

    # Update all trials in db (arf)
    n_trials_before = len(storage.fetch_trials(uid=experiment_config["_id"]))
    for trial in storage.fetch_trials(uid=experiment_config["_id"]):
        previous_id = trial.id
        for param in trial._params:
            if param.name == "/dropout":
                param.value /= 10
                assert 0 <= param.value <= 0.5, param.value

        storage.delete_trials(uid=experiment_config["_id"], where=dict(_id=previous_id))
        storage.register_trial(trial)

    trials = storage.fetch_trials(uid=experiment_config["_id"])
    assert len(trials) == n_trials_before, len(trials)
    for trial in trials:
        assert 0 <= trial.params["/dropout"] <= 0.5, trial


for experiment_config in storage.fetch_experiments({}):
    key = (experiment_config["name"], experiment_config["version"])

    if key in filter_exps:
        print(
            f"Saving {experiment_config['name']}-v{experiment_config['version']} as {filter_exps[key]}"
        )
        update_dropout(experiment_config)

        storage.update_experiment(
            uid=experiment_config["_id"], name=filter_exps[key], version=1
        )
    else:
        print(f"Deleting {experiment_config['name']}-v{experiment_config['version']}")
        storage.delete_experiment(uid=experiment_config["_id"])
        storage._db.remove("trials", query={"experiment": experiment_config["_id"]})
