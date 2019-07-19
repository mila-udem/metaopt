# -*- coding: utf-8 -*-
"""
:mod:`orion.storage.legacy` -- Legacy storage
=============================================================================

.. module:: legacy
   :platform: Unix
   :synopsis: Old Storage implementation

"""

from orion.core.io.convert import JSONConverter
from orion.core.io.database import Database
from orion.core.worker.trial import Trial
from orion.storage.base import BaseStorageProtocol


class Legacy(BaseStorageProtocol):
    """Legacy protocol, store all experiments and trials inside the Database()

    Parameters
    ----------
    uri: str
        database uri specifying how to connect to the database
        the uri follows the following format
        `mongodb://[username:password@]host1[:port1][,...hostN[:portN]][/[database][?options]]`

    """

    def __init__(self, uri=None):
        self._db = Database()
        self._setup_db()

    def _setup_db(self):
        """Database index setup"""
        self._db.ensure_index('experiments',
                              [('name', Database.ASCENDING),
                               ('metadata.user', Database.ASCENDING)],
                              unique=True)
        self._db.ensure_index('experiments', 'metadata.datetime')

        self._db.ensure_index('trials', 'experiment')
        self._db.ensure_index('trials', 'status')
        self._db.ensure_index('trials', 'results')
        self._db.ensure_index('trials', 'start_time')
        self._db.ensure_index('trials', [('end_time', Database.DESCENDING)])

    def create_experiment(self, config):
        """See :func:`~orion.storage.BaseStorageProtocol.create_experiment`"""
        return self._db.write('experiments', config)

    def update_experiment(self, experiment, where=None, **kwargs):
        """See :func:`~orion.storage.BaseStorageProtocol.update_experiment`"""
        if where is None:
            where = dict()

        where['_id'] = experiment._id
        return self._db.write('experiments', data=kwargs, query=where)

    def fetch_experiments(self, query):
        """See :func:`~orion.storage.BaseStorageProtocol.fetch_experiments`"""
        return self._db.read('experiments', query)

    def fetch_trials(self, query, selection=None):
        """See :func:`~orion.storage.BaseStorageProtocol.fetch_trials`"""
        return [Trial(**t) for t in self._db.read('trials', query=query, selection=selection)]

    def register_trial(self, trial):
        """See :func:`~orion.storage.BaseStorageProtocol.register_trial`"""
        self._db.write('trials', trial.to_dict())
        return trial

    def register_lie(self, trial):
        """See :func:`~orion.storage.BaseStorageProtocol.register_lie`"""
        self._db.write('lying_trials', trial.to_dict())

    def retrieve_result(self, trial, results_file=None, **kwargs):
        """Parse the results file that was generated by the trial process.

        Parameters
        ----------
        trial: Trial
            The trial object to be updated

        results_file: str
            the file handle to read the result from

        Returns
        -------
        returns the updated trial object

        Note
        ----
        This does not update the database!

        """
        results = JSONConverter().parse(results_file.name)

        trial.results = [
            Trial.Result(
                name=res['name'],
                type=res['type'],
                value=res['value']) for res in results
        ]

        assert trial.objective is not None, 'Trial should have returned an objective value!'
        return trial

    def update_trial(self, trial: Trial, where=None, **kwargs) -> Trial:
        """See :func:`~orion.storage.BaseStorageProtocol.update_trial`"""
        if where is None:
            where = dict()

        where['_id'] = trial.id
        return self._db.write('trials', data=kwargs, query=where)

    def fetch_pending_trials(self, experiment):
        """Fetch trials that have not run yet"""
        query = dict(
            experiment=experiment._id,
            status={'$in': ['new', 'suspended', 'interrupted']}
        )
        return self.fetch_trials(query)

