""" Interface for the Knowledge Base, which is not currently in the Orion codebase.
"""

from abc import ABC, abstractmethod
from logging import getLogger
from typing import Dict, List, Union

from orion.client import ExperimentClient
from orion.core.worker.experiment import Experiment
from orion.core.worker.trial import Trial
from orion.storage.base import Storage

log = getLogger(__file__)


class AbstractKnowledgeBase(ABC):
    """ Abstract Base Class for the KnowledgeBase, which currently isn't part of the
    Orion codebase.
    """
    @abstractmethod
    def get_related_trials(
        self,
        target_experiment: Union[Experiment, ExperimentClient],
        max_trials: int = None,
    ) -> Dict["ExperimentInfo", List[Trial]]:
        """ Retrieve experiments 'similar' to `target_experiment` and their trials.

        When `max_trials` is given, only up to `max_trials` are returned in total for
        all experiments.

        Parameters
        ----------
        target_experiment : Union[Experiment, ExperimentClient]
            The target experiment, or experiment client.
        max_trials : int, optional
            Maximum total number of trials to fetch. By default `None`, in which case
            all trials from all 'related' experiments are returned. 

        Returns
        -------
        Dict[ExperimentInfo, List[Trial]]
            Dictionary mapping from `ExperimentInfo` objects to a list of trials from
            that experiment.
        """
        pass

    @property
    @abstractmethod
    def n_stored_experiments(self) -> int:
        """ Returns the current number of experiments registered the Knowledge base. """
        pass

    @abstractmethod
    def add_storage(self, storage: Storage) -> None:
        """Adds the experiments from the given `storage`.

        Parameters
        ----------
        storage : Storage
            Storage object.
        """
        pass

    @abstractmethod
    def add_experiment(self, experiment: Union[Experiment, ExperimentClient]) -> None:
        """Adds trials from the given experiment to the knowledge base.

        Parameters
        ----------
        experiment : Union[Experiment, ExperimentClient]
            Experiment or experiment client to add to the KB.
        """
        pass
