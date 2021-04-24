#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.client.experiment`."""
import atexit
import copy
import datetime
import logging

import pandas.testing
import pytest

import orion.core
from orion.core.io.database import DuplicateKeyError
from orion.core.utils.exceptions import BrokenExperiment, SampleTimeout
from orion.core.worker.trial import Trial
from orion.storage.base import get_storage
from orion.testing import create_experiment, mock_space_iterate

config = dict(
    name="supernaekei",
    space={"x": "uniform(0, 200)"},
    metadata={
        "user": "tsirif",
        "orion_version": "XYZ",
        "VCS": {
            "type": "git",
            "is_dirty": False,
            "HEAD_sha": "test",
            "active_branch": None,
            "diff_sha": "diff",
        },
    },
    version=1,
    pool_size=1,
    max_trials=10,
    max_broken=5,
    working_dir="",
    algorithms={"random": {"seed": 1}},
    producer={"strategy": "NoParallelStrategy"},
    refers=dict(root_id="supernaekei", parent_id=None, adapter=[]),
)


base_trial = {
    "experiment": 0,
    "status": "new",  # new, reserved, suspended, completed, broken
    "worker": None,
    "start_time": None,
    "end_time": None,
    "heartbeat": None,
    "results": [],
    "params": [],
}


def compare_trials(trials_a, trials_b):
    """Compare two trials by using their configuration"""

    def to_dict(trial):
        return trial.to_dict()

    assert list(map(to_dict, trials_a)) == list(map(to_dict, trials_b))


def compare_without_heartbeat(trial_a, trial_b):
    """Compare trials configuration ommiting heartbeat"""
    trial_a_dict = trial_a.to_dict()
    trial_b_dict = trial_b.to_dict()
    trial_a_dict.pop("heartbeat")
    trial_b_dict.pop("heartbeat")
    assert trial_a_dict == trial_b_dict


def test_plot_is_defined():
    """Tests plot() method is defined"""
    with create_experiment(config, base_trial) as (_, _, client):
        assert client.plot()


def test_experiment_fetch_trials():
    """Test compliance of client and experiment `fetch_trials()`"""
    with create_experiment(config, base_trial) as (cfg, experiment, client):
        assert len(experiment.fetch_trials()) == 5
        compare_trials(experiment.fetch_trials(), client.fetch_trials())


def test_experiment_get_trial():
    """Test compliance of client and experiment `get_trial()`"""
    with create_experiment(config, base_trial) as (cfg, experiment, client):
        assert experiment.get_trial(uid=0) == client.get_trial(uid=0)


def test_experiment_fetch_trials_by_status():
    """Test compliance of client and experiment `fetch_trials_by_status()`"""
    with create_experiment(config, base_trial) as (cfg, experiment, client):
        compare_trials(
            experiment.fetch_trials_by_status("completed"),
            client.fetch_trials_by_status("completed"),
        )


def test_experiment_fetch_non_completed_trials():
    """Test compliance of client and experiment `fetch_noncompleted_trials()`"""
    with create_experiment(config, base_trial) as (cfg, experiment, client):
        compare_trials(
            experiment.fetch_noncompleted_trials(), client.fetch_noncompleted_trials()
        )


def test_experiment_to_pandas():
    """Test compliance of client and experiment `to_pandas()`"""
    with create_experiment(config, base_trial) as (cfg, experiment, client):
        pandas.testing.assert_frame_equal(experiment.to_pandas(), client.to_pandas())


@pytest.mark.usefixtures("version_XYZ")
class TestInsert:
    """Tests for ExperimentClient.insert"""

    def test_insert_params_wo_results(self):
        """Test insertion without results without reservation"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = client.insert(dict(x=100))
            assert trial.status == "interrupted"
            assert trial.params["x"] == 100
            assert trial.id in set(trial.id for trial in experiment.fetch_trials())
            compare_without_heartbeat(trial, client.get_trial(uid=trial.id))

            assert client._pacemakers == {}

    def test_insert_params_with_results(self):
        """Test insertion with results without reservation"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            timestamp = datetime.datetime.utcnow()
            trial = client.insert(
                dict(x=100), [dict(name="objective", type="objective", value=101)]
            )
            assert trial.status == "completed"
            assert trial.params["x"] == 100
            assert trial.objective.value == 101
            assert trial.end_time >= timestamp
            assert trial.id in set(trial.id for trial in experiment.fetch_trials())
            compare_without_heartbeat(trial, client.get_trial(uid=trial.id))
            assert client.get_trial(uid=trial.id).objective.value == 101

            assert client._pacemakers == {}

    def test_insert_params_with_results_and_reserve(self):
        """Test insertion with results and reservation"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            with pytest.raises(ValueError) as exc:
                client.insert(
                    dict(x=100),
                    [dict(name="objective", type="objective", value=101)],
                    reserve=True,
                )

            assert "Cannot observe a trial and reserve it" in str(exc.value)

    def test_insert_existing_params(self, monkeypatch):
        """Test that duplicated trials cannot be saved in storage"""
        mock_space_iterate(monkeypatch)
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            with pytest.raises(DuplicateKeyError) as exc:
                client.insert(dict(x=1))

            assert (
                "A trial with params {'x': 1} already exist for experiment supernaekei-v1"
                == str(exc.value)
            )

            assert client._pacemakers == {}

    def test_insert_partial_params(self):
        """Test that trial with missing dimension that has a default value can be saved"""
        config_with_default = copy.deepcopy(config)
        config_with_default["space"]["y"] = "uniform(0, 10, default_value=5)"
        trial_with_default = copy.deepcopy(base_trial)
        trial_with_default["params"].append({"name": "y", "type": "real", "value": 1})
        with create_experiment(config_with_default, trial_with_default) as (
            _,
            experiment,
            client,
        ):
            trial = client.insert(dict(x=100))

            assert trial.status == "interrupted"
            assert trial.params["x"] == 100
            assert trial.params["y"] == 5
            assert trial.id in set(trial.id for trial in experiment.fetch_trials())
            compare_without_heartbeat(trial, client.get_trial(uid=trial.id))

            assert client._pacemakers == {}

    def test_insert_partial_params_missing(self):
        """Test that trial with missing dimension cannot be saved"""
        config_with_default = copy.deepcopy(config)
        config_with_default["space"]["y"] = "uniform(0, 10)"
        trial_with_default = copy.deepcopy(base_trial)
        trial_with_default["params"].append({"name": "y", "type": "real", "value": 1})
        with create_experiment(config_with_default, trial_with_default) as (
            _,
            _,
            client,
        ):
            with pytest.raises(ValueError) as exc:
                client.insert(dict(x=1))

            assert (
                "Dimension y not specified and does not have a default value."
                == str(exc.value)
            )

    def test_insert_params_and_reserve(self):
        """Test that new trial is reserved properly with `reserve=True`"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = client.insert(dict(x=100), reserve=True)
            assert trial.status == "reserved"
            assert client._pacemakers[trial.id].is_alive()
            client._pacemakers.pop(trial.id).stop()

    def test_insert_params_fails_not_reserved(self, monkeypatch):
        """Test that failed insertion because of duplicated trials will not reserve the original
        trial
        """
        mock_space_iterate(monkeypatch)
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            with pytest.raises(DuplicateKeyError):
                client.insert(dict(x=1), reserve=True)

            assert client._pacemakers == {}

    def test_insert_bad_params(self):
        """Test that bad params cannot be registered in storage"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            with pytest.raises(ValueError) as exc:
                client.insert(dict(x="bad bad bad"))

            assert (
                "Dimension x value bad bad bad is outside of prior uniform(0, 200)"
                == str(exc.value)
            )
            assert client._pacemakers == {}

    def test_insert_params_bad_results(self):
        """Test that results with from format cannot be saved (trial is registered anyhow)"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            with pytest.raises(ValueError) as exc:
                client.insert(
                    dict(x=100), [dict(name="objective", type="bad bad bad", value=0)]
                )

            assert "Given type, bad bad bad, not one of: " in str(exc.value)
            assert client._pacemakers == {}


@pytest.mark.usefixtures("version_XYZ")
class TestReserve:
    """Tests for ExperimentClient.reserve"""

    def test_reserve(self):
        """Test reservation of registered trials"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = experiment.get_trial(uid=cfg.trials[1]["_id"])
            assert trial.status != "reserved"
            client.reserve(trial)
            assert trial.status == "reserved"
            assert experiment.get_trial(trial).status == "reserved"
            assert client._pacemakers[trial.id].is_alive()
            client._pacemakers.pop(trial.id).stop()

    def test_reserve_dont_exist(self):
        """Verify that unregistered trials cannot be reserved."""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = Trial(experiment="idontexist", params=cfg.trials[0]["params"])
            with pytest.raises(ValueError) as exc:
                client.reserve(trial)

            assert "Trial {} does not exist in database.".format(trial.id) == str(
                exc.value
            )
            assert client._pacemakers == {}

    def test_reserve_reserved_locally(self, caplog):
        """Verify that a trial cannot be reserved twice locally (warning, no exception)"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = experiment.get_trial(uid=cfg.trials[1]["_id"])
            assert trial.status != "reserved"
            client.reserve(trial)
            with caplog.at_level(logging.WARNING):
                client.reserve(trial)

            assert (
                "Trial {} is already reserved.".format(trial.id)
                == caplog.records[-1].message
            )

            assert client._pacemakers[trial.id].is_alive()
            client._pacemakers.pop(trial.id).stop()

    def test_reserve_reserved_remotely(self):
        """Verify that a trial cannot be reserved if already reserved by another process"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = Trial(**cfg.trials[1])
            assert trial.status == "interrupted"
            client.reserve(trial)
            remote_pacemaker = client._pacemakers.pop(trial.id)
            assert experiment.get_trial(trial).status == "reserved"

            trial = Trial(**cfg.trials[1])
            assert trial.status == "interrupted"
            with pytest.raises(RuntimeError) as exc:
                client.reserve(trial)

            assert "Could not reserve trial {}.".format(trial.id) == str(exc.value)

            assert trial.status == "interrupted"
            assert experiment.get_trial(trial).status == "reserved"
            assert client._pacemakers == {}
            remote_pacemaker.stop()

    def test_reserve_race_condition(self):
        """Verify that race conditions during `reserve` is detected and raises a comprehensible
        error
        """
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = client.get_trial(uid=cfg.trials[0]["_id"])
            experiment.set_trial_status(trial, "reserved")
            trial.status = "new"  # Let's pretend it is still available

            with pytest.raises(RuntimeError) as exc:
                client.reserve(trial)

            assert "Could not reserve trial {}.".format(trial.id) == str(exc.value)
            assert client._pacemakers == {}


@pytest.mark.usefixtures("version_XYZ")
class TestRelease:
    """Tests for ExperimentClient.release"""

    def test_release(self):
        """Test releasing (to interrupted)"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = experiment.get_trial(uid=cfg.trials[1]["_id"])
            client.reserve(trial)
            pacemaker = client._pacemakers[trial.id]
            client.release(trial)
            assert trial.status == "interrupted"
            assert experiment.get_trial(trial).status == "interrupted"
            assert trial.id not in client._pacemakers
            assert not pacemaker.is_alive()

    def test_release_status(self):
        """Test releasing with a specific status"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = experiment.get_trial(uid=cfg.trials[1]["_id"])
            client.reserve(trial)
            pacemaker = client._pacemakers[trial.id]
            client.release(trial, "broken")
            assert trial.status == "broken"
            assert experiment.get_trial(trial).status == "broken"
            assert trial.id not in client._pacemakers
            assert not pacemaker.is_alive()

    def test_release_invalid_status(self):
        """Test releasing with a specific status"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = experiment.get_trial(uid=cfg.trials[1]["_id"])
            client.reserve(trial)
            with pytest.raises(ValueError) as exc:
                client.release(trial, "mouf mouf")

            assert exc.match("Given status `mouf mouf` not one of")

    def test_release_dont_exist(self, monkeypatch):
        """Verify that unregistered trials cannot be released"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = Trial(experiment="idontexist", params=cfg.trials[1]["params"])

            def do_nada(trial, **kwargs):
                """Don't do anything"""
                return None

            monkeypatch.setattr(client, "_release_reservation", do_nada)

            with pytest.raises(ValueError) as exc:
                client.release(trial)

            assert "Trial {} does not exist in database.".format(trial.id) == str(
                exc.value
            )
            assert client._pacemakers == {}

    def test_release_race_condition(self):
        """Verify that race conditions during `release` is detected and raises a comprehensible
        error
        """
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = client.get_trial(uid=cfg.trials[1]["_id"])
            client.reserve(trial)
            pacemaker = client._pacemakers[trial.id]
            # Woops! Trial got failed over from another process.
            experiment.set_trial_status(trial, "interrupted")
            trial.status = "reserved"  # Let's pretend we don't know.

            with pytest.raises(RuntimeError) as exc:
                client.release(trial)

            assert "Reservation for trial {} has been lost before release.".format(
                trial.id
            ) in str(exc.value)
            assert client._pacemakers == {}
            assert not pacemaker.is_alive()

    def test_release_unreserved(self):
        """Verify that unreserved trials cannot be released"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = client.get_trial(uid=cfg.trials[1]["_id"])
            with pytest.raises(RuntimeError) as exc:
                client.release(trial)

            assert "Trial {} was already released locally.".format(trial.id) == str(
                exc.value
            )

            assert client._pacemakers == {}

    def test_release_already_released_but_incorrectly(self):
        """Verify that incorrectly released trials have its pacemaker stopped properly"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = client.get_trial(uid=cfg.trials[1]["_id"])
            client.reserve(trial)
            pacemaker = client._pacemakers[trial.id]
            assert trial.status == "reserved"
            experiment.set_trial_status(trial, "interrupted")
            assert trial.status == "interrupted"

            with pytest.raises(RuntimeError) as exc:
                client.release(trial)

            assert "Trial {} was already released locally.".format(trial.id) == str(
                exc.value
            )

            assert client._pacemakers == {}
            assert not pacemaker.is_alive()


@pytest.mark.usefixtures("version_XYZ")
class TestClose:
    """Test close method of the client"""

    def test_close_empty(self):
        """Test client can close when no trial is reserved"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            client.close()

    def test_close_with_reserved(self):
        """Test client cannot be closed if trials are reserved."""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            with client.suggest() as trial:
                with pytest.raises(RuntimeError) as exc:
                    client.close()

                assert "There is still reserved trials" in str(exc.value)

    def test_close_unregister_atexit(self, monkeypatch):
        """Test close properly unregister the atexit function"""

        def please_dont_call_me(client):
            raise RuntimeError("Please don't call me!!!")

        monkeypatch.setattr(
            "orion.client.experiment.set_broken_trials", please_dont_call_me
        )

        with create_experiment(config, base_trial) as (cfg, experiment, client):
            # The registered function in atexit is called as expected
            with pytest.raises(RuntimeError) as exc:
                atexit._run_exitfuncs()

            assert "Please don't call me!!!" == str(exc.value)

            # Unregister the function
            client.close()


@pytest.mark.usefixtures("version_XYZ")
class TestBroken:
    """Test handling of broken trials with atexit()"""

    def test_broken_trial(self):
        """Test that broken trials are detected"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            with pytest.raises(RuntimeError):
                with client.suggest() as trial:
                    assert trial.status == "reserved"
                    raise RuntimeError("Dummy failure!")

            assert client._pacemakers == {}
            assert client.get_trial(trial).status == "broken"

    def test_atexit_with_multiple_clients(self):
        """Test that each client has a separate atexit function"""
        config1 = copy.deepcopy(config)
        config2 = copy.deepcopy(config)
        config2["name"] = "cloned"
        with create_experiment(exp_config=config1, trial_config=base_trial) as (
            _,
            _,
            client1,
        ):
            with create_experiment(exp_config=config2, trial_config=base_trial) as (
                _,
                _,
                client2,
            ):
                with pytest.raises(RuntimeError):
                    with client1.suggest() as trial1, client2.suggest() as trial2:
                        assert trial1.status == "reserved"
                        assert trial2.status == "reserved"
                        raise RuntimeError("Dummy failure!")

                assert client1._pacemakers == {}
                assert client2._pacemakers == {}
                assert client1.get_trial(trial1).status == "broken"
                assert client2.get_trial(trial2).status == "broken"

    def test_atexit_with_multiple_clients_unregister(self, monkeypatch):
        """Test that each client has a separate atexit function that can be unregistered"""
        config1 = copy.deepcopy(config)
        config2 = copy.deepcopy(config)
        config2["name"] = "cloned"
        with create_experiment(exp_config=config1, trial_config=base_trial) as (
            _,
            _,
            client1,
        ):

            def please_dont_call_me(client):
                raise RuntimeError("Please don't call me!!!")

            monkeypatch.setattr(
                "orion.client.experiment.set_broken_trials", please_dont_call_me
            )

            with create_experiment(exp_config=config2, trial_config=base_trial) as (
                _,
                _,
                client2,
            ):
                trial1 = client1.suggest()
                trial2 = client2.suggest()

                # The registered function in atexit is called as expected
                with pytest.raises(RuntimeError) as exc:
                    atexit._run_exitfuncs()

                assert "Please don't call me!!!" == str(exc.value)

                # Unregister the function
                client2.release(trial2)
                client2.close()

                # It should not be called
                atexit._run_exitfuncs()

                assert client1._pacemakers == {}
                assert client2._pacemakers == {}
                assert client1.get_trial(trial1).status == "broken"
                assert client2.get_trial(trial2).status == "interrupted"

    def test_interrupted_trial(self):
        """Test that interrupted trials are not set to broken"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            with pytest.raises(KeyboardInterrupt):
                with client.suggest() as trial:
                    assert trial.status == "reserved"
                    raise KeyboardInterrupt

            assert client._pacemakers == {}
            assert client.get_trial(trial).status == "interrupted"

    def test_completed_then_interrupted_trial(self):
        """Test that interrupted trials are not set to broken"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            with pytest.raises(KeyboardInterrupt):
                with client.suggest() as trial:
                    assert trial.status == "reserved"
                    assert trial.results == []
                    assert get_storage().get_trial(trial).objective is None
                    client.observe(
                        trial, [dict(name="objective", type="objective", value=101)]
                    )
                    assert get_storage().get_trial(trial).objective.value == 101
                    assert trial.status == "completed"
                    raise KeyboardInterrupt

            assert client._pacemakers == {}
            assert client.get_trial(trial).status == "completed"


@pytest.mark.usefixtures("version_XYZ")
class TestSuggest:
    """Tests for ExperimentClient.suggest"""

    def test_suggest(self, monkeypatch):
        """Verify that suggest reserved availabe trials."""
        mock_space_iterate(monkeypatch)
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = client.suggest()
            assert trial.status == "reserved"
            assert trial.params["x"] == 1

            assert len(experiment.fetch_trials()) == 5
            assert client._pacemakers[trial.id].is_alive()
            client._pacemakers.pop(trial.id).stop()

    def test_suggest_new(self):
        """Verify that suggest can create, register and reserved new trials."""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            for _ in range(3):
                trial = client.suggest()
                assert trial.status == "reserved"
                assert len(experiment.fetch_trials()) == 5
                assert client._pacemakers[trial.id].is_alive()
                client._pacemakers[trial.id].stop()

            trial = client.suggest()
            assert trial.status == "reserved"
            assert trial.params["x"] == 57.57
            assert len(experiment.fetch_trials()) == 6

            assert client._pacemakers[trial.id].is_alive()
            for trial_id in list(client._pacemakers.keys()):
                client._pacemakers.pop(trial_id).stop()

    def test_suggest_race_condition(self, monkeypatch):
        """Verify that race conditions to register new trials is handled"""
        mock_space_iterate(monkeypatch)
        new_value = 50.0

        # algo will suggest once an already existing trial
        def amnesia(num=1):
            """Suggest a new value and then always suggest the same"""
            if amnesia.count == 0:
                value = [0]
            else:
                value = [new_value]

            amnesia.count += 1

            return [value]

        amnesia.count = 0

        with create_experiment(config, base_trial, statuses=["completed"]) as (
            cfg,
            experiment,
            client,
        ):

            monkeypatch.setattr(experiment.algorithms, "suggest", amnesia)

            assert len(experiment.fetch_trials()) == 1

            trial = client.suggest()
            assert trial.status == "reserved"
            assert trial.params["x"] == new_value
            assert amnesia.count == 2

            assert len(experiment.fetch_trials()) == 2
            assert client._pacemakers[trial.id].is_alive()
            client._pacemakers.pop(trial.id).stop()

    def test_suggest_algo_opt_out(self, monkeypatch):
        """Verify that None is returned when algo cannot sample new trials (opting opt)"""

        def opt_out(num=1):
            """Never suggest a new trial"""
            return None

        monkeypatch.setattr(orion.core.config.worker, "max_idle_time", -1)

        with create_experiment(config, base_trial, statuses=["completed"]) as (
            cfg,
            experiment,
            client,
        ):

            monkeypatch.setattr(experiment.algorithms, "suggest", opt_out)

            assert len(experiment.fetch_trials()) == 1

            with pytest.raises(SampleTimeout):
                client.suggest()

    def test_suggest_is_done(self):
        """Verify that completed experiments cannot suggest new trials"""
        with create_experiment(config, base_trial, statuses=["completed"] * 10) as (
            cfg,
            experiment,
            client,
        ):

            assert len(experiment.fetch_trials()) == 10
            assert client.is_done

            assert client.suggest() is None

    def test_suggest_is_broken(self):
        """Verify that broken experiments cannot suggest new trials"""
        with create_experiment(config, base_trial, statuses=["broken"] * 10) as (
            cfg,
            experiment,
            client,
        ):

            assert len(experiment.fetch_trials()) == 10
            assert client.is_broken

            with pytest.raises(BrokenExperiment):
                client.suggest()

    def test_suggest_is_done_race_condition(self, monkeypatch):
        """Verify that inability to suggest because is_done becomes True during produce() is
        handled.
        """
        with create_experiment(config, base_trial, statuses=["completed"] * 5) as (
            cfg,
            experiment,
            client,
        ):

            def is_done(self):
                """Experiment is done"""
                return True

            def set_is_done():
                """Set is_done while algo is trying to suggest"""
                monkeypatch.setattr(experiment.__class__, "is_done", property(is_done))

            monkeypatch.setattr(client._producer, "produce", set_is_done)

            assert len(experiment.fetch_trials()) == 5
            assert not client.is_done

            assert client.suggest() is None

            assert len(experiment.fetch_trials()) == 5
            assert client.is_done

    def test_suggest_is_broken_race_condition(self, monkeypatch):
        """Verify that experiments that gets broken during local algo.suggest gets properly
        handled
        """
        with create_experiment(config, base_trial, statuses=["broken"] * 1) as (
            cfg,
            experiment,
            client,
        ):

            def is_broken(self):
                """Experiment is broken"""
                return True

            def set_is_broken():
                """Set is_broken while algo is trying to suggest"""
                monkeypatch.setattr(
                    experiment.__class__, "is_broken", property(is_broken)
                )

            monkeypatch.setattr(client._producer, "produce", set_is_broken)

            assert len(experiment.fetch_trials()) == 1
            assert not client.is_broken

            with pytest.raises(BrokenExperiment):
                client.suggest()

            assert len(experiment.fetch_trials()) == 1
            assert client.is_broken

    def test_suggest_hierarchical_space(self):
        """Verify that suggest returns trial with proper hierarchical parameter."""
        exp_config = copy.deepcopy(config)
        exp_config["space"] = {
            "a": {"x": "uniform(0, 10, discrete=True)"},
            "b": {"y": "loguniform(1e-08, 1)", "z": "choices(['voici', 'voila', 2])"},
        }
        with create_experiment(
            exp_config=exp_config, trial_config=base_trial, statuses=[]
        ) as (cfg, experiment, client):
            trial = client.suggest()
            assert trial.status == "reserved"
            assert len(trial.params) == 2
            assert "x" in trial.params["a"]
            assert "y" in trial.params["b"]
            assert "z" in trial.params["b"]

            assert client._pacemakers[trial.id].is_alive()
            client._pacemakers.pop(trial.id).stop()


@pytest.mark.usefixtures("version_XYZ")
class TestObserve:
    """Tests for ExperimentClient.observe"""

    def test_observe(self):
        """Verify that `observe()` will update the storage"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = Trial(**cfg.trials[1])
            assert trial.results == []
            client.reserve(trial)
            assert get_storage().get_trial(trial).objective is None
            client.observe(trial, [dict(name="objective", type="objective", value=101)])
            assert get_storage().get_trial(trial).objective.value == 101

    def test_observe_unreserved(self):
        """Verify that `observe()` will fail on non-reserved trials"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = Trial(**cfg.trials[1])
            with pytest.raises(RuntimeError) as exc:
                client.observe(
                    trial, [dict(name="objective", type="objective", value=101)]
                )

            assert "Trial {} had no pacemakers. Was it reserved properly?".format(
                trial.id
            ) == str(exc.value)

    def test_observe_dont_exist(self):
        """Verify that `observe()` will fail on non-registered trials"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = Trial(experiment="idontexist", params=cfg.trials[0]["params"])
            with pytest.raises(ValueError) as exc:
                client.observe(
                    trial, [dict(name="objective", type="objective", value=101)]
                )

            assert "Trial {} does not exist in database.".format(trial.id) == str(
                exc.value
            )
            assert client._pacemakers == {}

    def test_observe_bad_results(self):
        """Verify that bad results type is detected and ValueError is raised"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = Trial(**cfg.trials[1])
            client.reserve(trial)
            with pytest.raises(ValueError) as exc:
                client.observe(
                    trial, [dict(name="objective", type="bad bad bad", value=101)]
                )

            assert "Given type, bad bad bad, not one of: " in str(exc.value)
            assert client._pacemakers[trial.id].is_alive()
            client._pacemakers.pop(trial.id).stop()

    def test_observe_race_condition(self):
        """Verify that race condition during `observe()` is detected and raised"""
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            trial = client.get_trial(uid=cfg.trials[1]["_id"])
            client.reserve(trial)
            experiment.set_trial_status(trial, "interrupted")
            trial.status = "reserved"  # Let's pretend it is still reserved

            with pytest.raises(RuntimeError) as exc:
                client.observe(
                    trial, [dict(name="objective", type="objective", value=101)]
                )

            assert "Reservation for trial {} has been lost.".format(trial.id) == str(
                exc.value
            )
            assert client._pacemakers == {}

    def test_observe_under_with(self):
        with create_experiment(config, base_trial) as (cfg, experiment, client):
            with client.suggest() as trial:
                assert trial.status == "reserved"
                assert trial.results == []
                assert get_storage().get_trial(trial).objective is None
                client.observe(
                    trial, [dict(name="objective", type="objective", value=101)]
                )
                assert get_storage().get_trial(trial).objective.value == 101
                assert trial.status == "completed"

            assert trial.status == "completed"  # Still completed after __exit__


@pytest.mark.usefixtures("version_XYZ")
class TestWorkon:
    """Tests for ExperimentClient.workon"""

    def test_workon(self):
        """Verify that workon processes properly"""

        def foo(x):
            return [dict(name="result", type="objective", value=x * 2)]

        with create_experiment(config, base_trial, statuses=[]) as (
            cfg,
            experiment,
            client,
        ):
            client.workon(foo, max_trials=5)
            assert len(experiment.fetch_trials()) == 5
            assert client._pacemakers == {}

    def test_workon_partial(self):
        """Verify that partial is properly passed to the function"""

        def foo(x, y):
            return [dict(name="result", type="objective", value=x * 2 + y)]

        with create_experiment(config, base_trial, statuses=[]) as (
            cfg,
            experiment,
            client,
        ):
            client.workon(foo, max_trials=10, y=2)
            assert len(experiment.fetch_trials()) == 10
            assert client._pacemakers == {}

    def test_workon_partial_with_override(self):
        """Verify that partial is overriden by trial.params"""

        def foo(x, y):
            return [dict(name="result", type="objective", value=x * 2 + y)]

        ext_config = copy.deepcopy(config)
        ext_config["space"]["y"] = "uniform(0, 10)"

        with create_experiment(
            exp_config=ext_config, trial_config=base_trial, statuses=[]
        ) as (cfg, experiment, client):
            default_y = 2
            assert len(experiment.fetch_trials()) == 0
            client.workon(foo, max_trials=1, y=default_y)
            assert len(experiment.fetch_trials()) == 1
            assert experiment.fetch_trials()[0].params["y"] != 2

    def test_workon_hierarchical_partial_with_override(self):
        """Verify that hierarchical partial is overriden by trial.params"""
        default_y = 2
        default_z = "voila"

        def foo(a, b):
            assert b["y"] != default_y
            assert b["z"] == default_z
            return [dict(name="result", type="objective", value=a["x"] * 2 + b["y"])]

        ext_config = copy.deepcopy(config)
        ext_config["space"] = {
            "a": {"x": "uniform(0, 10, discrete=True)"},
            "b": {"y": "loguniform(1e-08, 1)"},
        }

        with create_experiment(
            exp_config=ext_config, trial_config=base_trial, statuses=[]
        ) as (cfg, experiment, client):
            assert len(experiment.fetch_trials()) == 0
            client.workon(foo, max_trials=5, b={"y": default_y, "z": default_z})
            assert len(experiment.fetch_trials()) == 5
            params = experiment.fetch_trials()[0].params
            assert len(params)
            assert "x" in params["a"]
            assert "y" in params["b"]
