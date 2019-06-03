# -*- coding: utf-8 -*-
# pylint:disable=protected-access,too-many-public-methods
"""
:mod:`orion.core.worker.experiment` -- Description of an optimization attempt
=============================================================================

.. module:: experiment
   :platform: Unix
   :synopsis: Manage history of trials corresponding to a black box process

"""
import copy
import datetime
import getpass
import logging
import random
import sys

from orion.algo.base import PrimaryAlgo
from orion.core.cli.evc import fetch_branching_configuration
from orion.core.evc.adapters import Adapter, BaseAdapter
from orion.core.evc.conflicts import detect_conflicts
from orion.core.io.database import Database, DuplicateKeyError, ReadOnlyDB
from orion.core.io.experiment_branch_builder import ExperimentBranchBuilder
from orion.core.io.interactive_commands.branching_prompt import BranchingPrompt
from orion.core.io.space_builder import SpaceBuilder
from orion.core.utils.format_trials import trial_to_tuple
from orion.core.worker.strategy import (BaseParallelStrategy,
                                        Strategy)
from orion.core.worker.trial import Trial

log = logging.getLogger(__name__)


# pylint: disable=too-many-public-methods
class Experiment(object):
    """Represents an entry in database/experiments collection.

    Attributes
    ----------
    name : str
       Unique identifier for this experiment per `user`.
    id: object
       id of the experiment in the database if experiment is configured. Value is `None`
       if the experiment is not configured.
    refers : dict or list of `Experiment` objects, after initialization is done.
       A dictionary pointing to a past `Experiment` id, ``refers[parent_id]``, whose
       trials we want to add in the history of completed trials we want to re-use.
       For convenience and database effiency purpose, all experiments of a common tree shares
       `refers[root_id]`, with the root experiment refering to itself.
    metadata : dict
       Contains managerial information about this `Experiment`.
    pool_size : int
       How many workers can participate asynchronously in this `Experiment`.
    max_trials : int
       How many trials must be evaluated, before considering this `Experiment` done.
       This attribute can be updated if the rest of the experiment configuration
       is the same. In that case, if trying to set to an already set experiment,
       it will overwrite the previous one.
    algorithms : dict of dicts or an `PrimaryAlgo` object, after initialization is done.
       Complete specification of the optimization and dynamical procedures taking
       place in this `Experiment`.

    Metadata
    --------
    user : str
       System user currently owning this running process, the one who invoked **Oríon**.
    datetime : `datetime.datetime`
       When was this particular configuration submitted to the database.
    orion_version : str
       Version of **Oríon** which suggested this experiment. `user`'s current
       **Oríon** version.
    user_script : str
       Full absolute path to `user`'s executable.
    user_args : list of str
       Contains separate arguments to be passed when invoking `user_script`,
       possibly templated for **Oríon**.
    user_vcs : str, optional
       User's version control system for this executable's code repository.
    user_version : str, optional
       Current user's repository version.
    user_commit_hash : str, optional
       Current `Experiment`'s commit hash for **Oríon**'s invocation.

    """

    __slots__ = ('name', 'refers', 'metadata', 'pool_size', 'max_trials',
                 'algorithms', 'producer', '_db', '_init_done', '_id',
                 '_node', '_last_fetched')
    non_branching_attrs = ('pool_size', 'max_trials')

    def __init__(self, name, user=None):
        """Initialize an Experiment object with primary key (:attr:`name`, :attr:`user`).

        Try to find an entry in `Database` with such a key and config this object
        from it import, if successful. Else, init with default/empty values and
        insert new entry with this object's attributes in database.

        .. note::
           Practically initialization has not finished until `config`'s setter
           is called.

        :param name: Describe a configuration with a unique identifier per :attr:`user`.
        :type name: str
        """
        log.debug("Creating Experiment object with name: %s", name)
        self._init_done = False
        self._db = Database()  # fetch database instance
        self._setup_db()  # build indexes for collections

        self._id = None
        self.name = name
        self._node = None
        self.refers = {}
        if user is None:
            user = getpass.getuser()
        self.metadata = {'user': user}
        self.pool_size = None
        self.max_trials = None
        self.algorithms = None
        self.producer = {'strategy': None}

        config = self._db.read('experiments',
                               {'name': name, 'metadata.user': user})
        if config:
            log.debug("Found existing experiment, %s, under user, %s, registered in database.",
                      name, user)
            if len(config) > 1:
                log.warning("Many (%s) experiments for (%s, %s) are available but "
                            "only the most recent one can be accessed. "
                            "Experiment branches will be supported soon.", len(config), name, user)
            config = sorted(config, key=lambda x: x['metadata']['datetime'],
                            reverse=True)[0]
            for attrname in self.__slots__:
                if not attrname.startswith('_') and attrname in config:
                    setattr(self, attrname, config[attrname])
            self._id = config['_id']

        self._last_fetched = self.metadata.get("datetime", datetime.datetime.utcnow())

    def _setup_db(self):
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

    def fetch_trials(self, query, selection=None):
        """Fetch trials of the experiment in the database

        .. note::

            The query is always updated with `{"experiment": self._id}`

        .. seealso::

            :meth:`orion.core.io.database.AbstractDB.read` for more information about the
            arguments.

        """
        query["experiment"] = self._id

        return Trial.build(self._db.read('trials', query, selection))

    def fetch_trials_tree(self, query, selection=None):
        """Fetch trials recursively in the EVC tree

        .. seealso::

            :meth:`orion.core.worker.Experiment.fetch_trials` for more information about the
            arguments.

            :class:`orion.core.evc.experiment.ExperimentNode` for more information about the EVC
            tree.

        """
        if self._node is None:
            return self.fetch_trials(query, selection)

        return self._node.fetch_trials(query, selection)

    def connect_to_version_control_tree(self, node):
        """Connect the experiment to its node in a version control tree

        .. seealso::

            :class:`orion.core.evc.experiment.ExperimentNode`

        :param node: Node giving access to the experiment version control tree.
        :type name: None or `ExperimentNode`
        """
        self._node = node

    def reserve_trial(self, score_handle=None):
        """Find *new* trials that exist currently in database and select one of
        them based on the highest score return from `score_handle` callable.

        :param score_handle: A way to decide which trial out of the *new* ones to
           to pick as *reserved*, defaults to a random choice.
        :type score_handle: callable
        :return: selected `Trial` object, None if could not find any.
        """
        if score_handle is not None and not callable(score_handle):
            raise ValueError("Argument `score_handle` must be callable with a `Trial`.")

        query = dict(
            experiment=self._id,
            status={'$in': ['new', 'suspended', 'interrupted']}
            )
        new_trials = self.fetch_trials(query)

        if not new_trials:
            return None

        if score_handle is not None and self.space:
            scores = list(map(score_handle,
                              map(lambda x: trial_to_tuple(x, self.space), new_trials)))
            scored_trials = zip(scores, new_trials)
            best_trials = filter(lambda st: st[0] == max(scores), scored_trials)
            new_trials = list(zip(*best_trials))[1]
        elif score_handle is not None:
            log.warning("While reserving trial: `score_handle` was provided, but "
                        "parameter space has not been defined yet.")

        selected_trial = random.sample(new_trials, 1)[0]

        # Query on status to ensure atomicity. If another process change the
        # status meanwhile, read_and_write will fail, because query will fail.
        query = {'_id': selected_trial.id, 'status': selected_trial.status}

        update = dict(status='reserved')

        if selected_trial.status == 'new':
            update["start_time"] = datetime.datetime.utcnow()

        selected_trial_dict = self._db.read_and_write(
            'trials', query=query, data=update)

        if selected_trial_dict is None:
            selected_trial = self.reserve_trial(score_handle=score_handle)
        else:
            selected_trial = Trial(**selected_trial_dict)

        return selected_trial

    def push_completed_trial(self, trial):
        """Inform database about an evaluated `trial` with results.

        :param trial: Corresponds to a successful evaluation of a particular run.
        :type trial: `Trial`

        .. note::

            Change status from *reserved* to *completed*.

        """
        trial.end_time = datetime.datetime.utcnow()
        trial.status = 'completed'
        self._db.write('trials', trial.to_dict(), query={'_id': trial.id})

    def register_lie(self, lying_trial):
        """Register a *fake* trial created by the strategist.

        The main difference between fake trial and orignal ones is the addition of a fake objective
        result, and status being set to completed. The id of the fake trial is different than the id
        of the original trial, but the original id can be computed using the hashcode on parameters
        of the fake trial. See mod:`orion.core.worker.strategy` for more information and the
        Strategist object and generation of fake trials.

        Parameters
        ----------
        trials: `Trial` object
            Fake trial to register in the database

        Raises
        ------
        orion.core.io.database.DuplicateKeyError
            If a trial with the same id already exist in the database. Since the id is computed
            based on a hashing of the trial, this should mean that an identical trial already exist
            in the database.

        """
        lying_trial.status = 'completed'
        lying_trial.end_time = datetime.datetime.utcnow()

        self._db.write('lying_trials', lying_trial.to_dict())

    def register_trial(self, trial):
        """Register new trial in the database.

        Inform database about *new* suggested trial with specific parameter values. Trials may only
        be registered one at a time to avoid registration of duplicates.

        Parameters
        ----------
        trials: `Trial` object
            Trial to register in the database

        Raises
        ------
        orion.core.io.database.DuplicateKeyError
            If a trial with the same id already exist in the database. Since the id is computed
            based on a hashing of the trial, this should mean that an identical trial already exist
            in the database.

        """
        stamp = datetime.datetime.utcnow()
        trial.experiment = self._id
        trial.status = 'new'
        trial.submit_time = stamp

        self._db.write('trials', trial.to_dict())

    def fetch_completed_trials(self):
        """Fetch recent completed trials that this `Experiment` instance has not
        yet seen.

        .. note::

            It will return only those with `Trial.end_time` after `_last_fetched`, for performance
            reasons.

        :return: list of completed `Trial` objects
        """
        query = dict(
            status='completed',
            end_time={'$gte': self._last_fetched}
            )
        completed_trials = self.fetch_trials_tree(query)
        self._last_fetched = datetime.datetime.utcnow()

        return completed_trials

    def fetch_noncompleted_trials(self):
        """Fetch non-completed trials of this `Experiment` instance.

        .. note::

            It will return all non-completed trials, including new, reserved, suspended,
            interruped and broken ones.

        :return: list of non-completed `Trial` objects
        """
        query = dict(
            status={'$ne': 'completed'}
            )

        return self.fetch_trials_tree(query)

    # pylint: disable=invalid-name
    @property
    def id(self):
        """Id of the experiment in the database if configured.

        Value is `None` if the experiment is not configured.
        """
        return self._id

    @property
    def node(self):
        """Node of the experiment in the version control tree.

        Value is `None` if the experiment is not connected to the version control tree.

        .. seealso::

            :py:meth:`orion.core.worker.experiment.Experiment.connect_to_version_control_tree`

        """
        return self._node

    @property
    def is_done(self):
        """Return True, if this experiment is considered to be finished.

        1. Count how many trials have been completed and compare with `max_trials`.
        2. Ask `algorithms` if they consider there is a chance for further improvement.

        .. note::

            To be used as a terminating condition in a ``Worker``.

        """
        query = dict(
            experiment=self._id,
            status='completed'
            )
        num_completed_trials = self._db.count('trials', query)

        return ((num_completed_trials >= self.max_trials) or
                (self._init_done and self.algorithms.is_done))

    @property
    def space(self):
        """Return problem's parameter `orion.algo.space.Space`.

        .. note:: It will return None, if experiment init is not done.
        """
        if self._init_done:
            return self.algorithms.space
        return None

    @property
    def configuration(self):
        """Return a copy of an `Experiment` configuration as a dictionary."""
        config = dict()
        for attrname in self.__slots__:
            if attrname.startswith('_'):
                continue
            attribute = getattr(self, attrname)
            if self._init_done and attrname == 'algorithms':
                config[attrname] = attribute.configuration
            else:
                config[attrname] = attribute

            if self._init_done and attrname == "refers" and attribute.get("adapter"):
                config[attrname] = copy.deepcopy(config[attrname])
                config[attrname]['adapter'] = config[attrname]['adapter'].configuration

            if self._init_done and attrname == "producer" and attribute.get("strategy"):
                config[attrname] = copy.deepcopy(config[attrname])
                config[attrname]['strategy'] = config[attrname]['strategy'].configuration

        # Reason for deepcopy is that some attributes are dictionaries
        # themselves, we don't want to accidentally change the state of this
        # object from a getter.
        return copy.deepcopy(config)

    def configure(self, config, enable_branching=True):
        """Set `Experiment` by overwriting current attributes.

        If `Experiment` was already set and an overwrite is needed, a *branch*
        is advised with a different :attr:`name` for this particular configuration.

        .. note::

            Calling this property is necessary for an experiment's initialization process to be
            considered as done. But it can be called only once.

        """
        if self._init_done:
            raise RuntimeError("Configuration is done; cannot reset an Experiment.")

        # Experiment was build using db, but config was build before experiment got in db.
        # Fake a DuplicateKeyError to force reinstantiation of experiment with proper config.
        if self._id is not None and "datetime" not in config['metadata']:
            raise DuplicateKeyError("Cannot register an existing experiment with a new config")

        # Copy and simulate instantiating given configuration
        experiment = Experiment(self.name)
        experiment._instantiate_config(self.configuration)
        experiment._instantiate_config(config)
        experiment._init_done = True

        # If id is None in this object, then database did not hit a config
        # with same (name, user's name) pair. Everything depends on the user's
        # orion_config to set.
        if self._id is None:
            if config['name'] != self.name or \
                    config['metadata']['user'] != self.metadata['user']:
                raise ValueError("Configuration given is inconsistent with this Experiment.")
            is_new = True
        else:
            # Branch if it is needed
            # TODO: When refactoring experiment managenent, is_different_from
            # will be used when EVC is not available.
            # is_new = self._is_different_from(experiment.configuration)
            branching_configuration = fetch_branching_configuration(config)
            conflicts = detect_conflicts(self.configuration, experiment.configuration)
            is_new = len(conflicts.get()) > 1 or branching_configuration.get('branch')
            if is_new and not enable_branching:
                raise ValueError("Configuration is different and generate a "
                                 "branching event")
            elif is_new:
                experiment._branch_config(conflicts, branching_configuration)

        final_config = experiment.configuration
        self._instantiate_config(final_config)

        self._init_done = True

        # If everything is alright, push new config to database
        if is_new:
            final_config['metadata']['datetime'] = datetime.datetime.utcnow()
            self.metadata['datetime'] = final_config['metadata']['datetime']
            # This will raise DuplicateKeyError if a concurrent experiment with
            # identical (name, metadata.user) is written first in the database.

            self._db.write('experiments', final_config)
            # XXX: Reminder for future DB implementations:
            # MongoDB, updates an inserted dict with _id, so should you :P
            self._id = final_config['_id']

            # Update refers in db if experiment is root
            if not self.refers:
                self.refers = {'root_id': self._id, 'parent_id': None, 'adapter': []}
                update = {'refers': self.refers}
                query = {'_id': self._id}
                self._db.write('experiments', data=update, query=query)

        else:
            # Writing the final config to an already existing experiment raises
            # a DuplicatKeyError because of the embedding id `metadata.user`.
            # To avoid this `final_config["name"]` is popped out before
            # `db.write()`, thus seamingly breaking  the compound index
            # `(name, metadata.user)`
            final_config.pop("name")
            self._db.write('experiments', final_config, {'_id': self._id})

    @property
    def stats(self):
        """Calculate a stats dictionary for this particular experiment.

        Returns
        -------
        stats : dict

        Stats
        -----
        trials_completed : int
           Number of completed trials
        best_trials_id : int
           Unique identifier of the `Trial` object in the database which achieved
           the best known objective result.
        best_evaluation : float
           Evaluation score of the best trial
        start_time : `datetime.datetime`
           When Experiment was first dispatched and started running.
        finish_time : `datetime.datetime`
           When Experiment reached terminating condition and stopped running.
        duration : `datetime.timedelta`
           Elapsed time.

        """
        query = dict(
            experiment=self._id,
            status='completed'
            )
        selection = {
            'end_time': 1,
            'results': 1,
            'experiment': 1,
            'params': 1
            }
        completed_trials = self.fetch_trials(query, selection)
        if not completed_trials:
            return dict()
        stats = dict()
        stats['trials_completed'] = len(completed_trials)
        stats['best_trials_id'] = None
        trial = completed_trials[0]
        stats['best_evaluation'] = trial.objective.value
        stats['best_trials_id'] = trial.id
        stats['start_time'] = self.metadata['datetime']
        stats['finish_time'] = stats['start_time']
        for trial in completed_trials:
            # All trials are going to finish certainly after the start date
            # of the experiment they belong to
            if trial.end_time > stats['finish_time']:  # pylint:disable=no-member
                stats['finish_time'] = trial.end_time
            objective = trial.objective.value
            if objective < stats['best_evaluation']:
                stats['best_evaluation'] = objective
                stats['best_trials_id'] = trial.id
        stats['duration'] = stats['finish_time'] - stats['start_time']

        return stats

    def _instantiate_config(self, config):
        """Check before dispatching experiment whether configuration corresponds
        to a executable experiment environment.

        1. Check `refers` and instantiate `Adapter` objects from it.
        2. Try to build parameter space from user arguments.
        3. Check whether configured algorithms correspond to [known]/valid
           implementations of the ``Algorithm`` class. Instantiate these objects.
        4. Check if experiment `is_done`, prompt for larger `max_trials` if it is. (TODO)

        """
        # Just overwrite everything else given
        for section, value in config.items():
            if section not in self.__slots__:
                log.info("Found section '%s' in configuration. Experiments "
                         "do not support this option. Ignoring.", section)
                continue
            if section.startswith('_'):
                log.info("Found section '%s' in configuration. "
                         "Cannot set private attributes. Ignoring.", section)
                continue

            # Copy sub configuration to value confusing side-effects
            # Only copy at this level, not `config` directly to avoid TypeErrors if config contains
            # non-serializable objects (copy.deepcopy complains otherwise).
            if isinstance(value, dict):
                value = copy.deepcopy(value)

            setattr(self, section, value)

        try:
            space_builder = SpaceBuilder()
            space = space_builder.build_from(config['metadata']['user_args'])

            if space_builder.userconfig:
                with open(space_builder.userconfig) as f:
                    self.metadata['script_config_file'] = f.read()

            if not space:
                raise ValueError("Parameter space is empty. There is nothing to optimize.")

            # Instantiate algorithms
            self.algorithms = PrimaryAlgo(space, self.algorithms)
        except KeyError:
            pass

        if self.refers and not isinstance(self.refers.get('adapter'), BaseAdapter):
            self.refers['adapter'] = Adapter.build(self.refers['adapter'])

        if not self.producer.get('strategy'):
            log.warning('You have not set a producer strategy, the basic '
                        'NoParallelStrategy will be used')
            strategy = Strategy({"NoParallelStrategy": {}})
            self.producer = {'strategy': strategy}
        elif not isinstance(self.producer.get('strategy'), BaseParallelStrategy):
            strategy = Strategy({self.producer['strategy']: {}})
            self.producer = {'strategy': strategy}

    def _branch_config(self, conflicts, branching_configuration):
        """Ask for a different identifier for this experiment. Set :attr:`refers`
        key to previous experiment's name, the one that we branched from.

        :param config: Conflicting configuration that will change based on prompt.
        """
        experiment_brancher = ExperimentBranchBuilder(conflicts, branching_configuration)

        needs_manual_resolution = (not experiment_brancher.is_resolved or
                                   experiment_brancher.auto_resolution)

        if needs_manual_resolution:
            branching_prompt = BranchingPrompt(experiment_brancher)

            if not sys.__stdin__.isatty():
                raise ValueError(
                    "Configuration is different and generates a branching event:\n{}".format(
                        branching_prompt.get_status()))

            branching_prompt.cmdloop()

            if branching_prompt.abort or not experiment_brancher.is_resolved:
                sys.exit()

        adapter = experiment_brancher.create_adapters()
        self._instantiate_config(experiment_brancher.conflicting_config)
        self.refers['adapter'] = adapter
        self.refers['parent_id'] = self._id

    def _is_different_from(self, config):
        """Return True, if current `Experiment`'s configuration as described by
        its attributes is different from the one suggested in `config`.
        """
        is_diff = False
        for section, value in config.items():
            if section in self.non_branching_attrs or \
                    section not in self.__slots__ or \
                    section.startswith('_'):
                continue
            item = getattr(self, section)
            if item != value:
                log.warning("Config given is different from config found in db at section: %s",
                            section)
                log.warning("Config+ :\n%s", value)
                log.warning("Config- :\n%s", item)
                is_diff = True
                break

        return is_diff

    def __repr__(self):
        """Represent the object as a string."""
        return "Experiment(name=%s, metadata.user=%s)" % (self.name, self.metadata['user'])


# pylint: disable=too-few-public-methods
class ExperimentView(object):
    """Non-writable view of an experiment

    .. seealso::

        :py:class:`orion.core.worker.experiment.Experiment` for writable experiments.

    """

    __slots__ = ('_experiment', )

    #                     Attributes
    valid_attributes = (["_id", "name", "refers", "metadata", "pool_size", "max_trials"] +
                        # Properties
                        ["id", "node", "is_done", "space", "algorithms", "stats", "configuration"] +
                        # Methods
                        ["fetch_trials", "fetch_trials_tree", "fetch_completed_trials",
                         "connect_to_version_control_tree"])

    def __init__(self, name, user=None):
        """Initialize viewed experiment object with primary key (:attr:`name`, :attr:`user`).

        Build an experiment from configuration found in `Database` with a key (name, user).

        .. note::

            A view is fully configured at initialiation. It cannot be reconfigured.
            If no experiment is found for the key (name, user), a `ValueError` will be raised.

        :param name: Describe a configuration with a unique identifier per :attr:`user`.
        :type name: str
        """
        self._experiment = Experiment(name, user)

        if self._experiment.id is None:
            raise ValueError("No experiment with given name '%s' for user '%s' inside database, "
                             "no view can be created." %
                             (self._experiment.name, self._experiment.metadata['user']))

        try:
            self._experiment.configure(self._experiment.configuration, enable_branching=False)
        except ValueError as e:
            if "Configuration is different and generates a branching event" in str(e):
                raise RuntimeError(
                    "Configuration in the database does not correspond to the one generated by "
                    "Experiment object. This is likely due to a backward incompatible update in "
                    "Oríon. Please report to https://github.com/epistimio/orion/issues.") from e

            raise

        self._experiment._db = ReadOnlyDB(self._experiment._db)

    def __getattr__(self, name):
        """Get attribute only if valid"""
        if name not in self.valid_attributes:
            raise AttributeError("Cannot access attribute %s on view-only experiments." % name)

        return getattr(self._experiment, name)

    def __repr__(self):
        """Represent the object as a string."""
        return "ExperimentView(name=%s, metadata.user=%s)" % (self.name, self.metadata['user'])
