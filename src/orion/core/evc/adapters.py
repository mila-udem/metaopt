# -*- coding: utf-8 -*-
# pylint: disable=too-many-lines
"""
:mod:`orion.core.evc.adapters` -- Adapters to connect experiments within the EVC system
=======================================================================================

.. module:: adapters
   :platform: Unix
   :synopsis: Adapters to connect experiments within the EVC system

Experiment branches because of changes in the configuration or the user's code. This make
experiments incompatible with one another unless we define adapters such that a branched experiment
B can access trials from parent experiment A.

Adapters have two main methods, forward and backward. The method forward defines how trials from
parent experiment A are filtered or adapted to be compatible with experiment B. Modifications are
only applied at execution time and are not saved anywhere in the database. Adapters only provides a
view on an experiment.

There is adapters for

    * Dimension addition
    * Dimension deletion
    * Dimension renaming
    * Change of dimension prior
    * Change of algorithm
    * Change of code
    * Combining different adapters

Adapters all have a `to_dict` method which provides sufficient information to rebuild the adapter.
This is to facilitate save of adapters in a database and retrieval.

Adapters can be build using the factory class `Adapter(**kwargs)` or using
`Adapter.build(list_of_dicts)`.

"""
from abc import (ABCMeta, abstractmethod)
import copy

from orion.core.io.space_builder import DimensionBuilder
from orion.core.utils import Factory
from orion.core.worker.trial import Trial


class BaseAdapter(object, metaclass=ABCMeta):
    """Base class describing what an adapter can do."""

    @abstractmethod
    def forward(self, trials):
        """Adapt trials of the parent experiment such that they are compatible to the child
        experiment

        Parameters
        ----------
        trials: list of :class:`orion.core.worker.trial.Trial`
            List of :class:`orion.core.worker.trial.Trial` coming from the parent experiment

        """
        pass

    @abstractmethod
    def backward(self, trials):
        """Adapt trials of the child experiment such that they are compatible to the parent
        experiment

        Parameters
        ----------
        trials: list of :class:`orion.core.worker.trial.Trial`
            List of :class:`orion.core.worker.trial.Trial` coming from the child experiment

        """
        pass

    @abstractmethod
    def to_dict(self):
        """Provide the configuration of the adapter as a dictionary

        This method is intended for single adapters only.
        For coherence, method configuration is used by all adapters, which is a list of dictionary
        configurations.

        .. seealso::

            :meth:`orion.core.evc.adapters.BaseAdapter.configuration`
        """
        pass

    @property
    def configuration(self):
        """Provide the configuration of the adapter.

        For simplicity, the configuration is always returned as if the adapter is a
        CompositeAdapter, therefore no matter if the adapter is composite or not it will return a
        list of configurations.

        The dictionaries of the list can be used to recreate adapters, for any adapter class.

        Examples
        --------
        .. code-block:: python
           :linenos:

            configuration = a_dummy_adapter.configuration[0]
            another_dummy_adapter = Adapter.build([configuration])
            assert another_dummy_adapter.configuration[0] == configuration

        Returns
        -------
        dict
            Configuration as a dictionary

        """
        return [self.to_dict()]


class CompositeAdapter(BaseAdapter):
    """Adapter which group many other adapters needed to connect two experiments

    Attributes
    ----------
    adapters: instances of `orion.core.evc.adapters.BaseAdapter`
        List of adaptors which are applied sequentially

    """

    def __init__(self, *adapters):
        """Initialize with adapters

        Parameters
        ----------
        adapters: instances of `orion.core.evc.adapters.BaseAdapter`
            List of adaptors which are applied sequentially

        """
        if any(not isinstance(adapter, BaseAdapter) for adapter in adapters):
            wrong_object_type = [type(adapter) for adapter in adapters
                                 if not isinstance(adapter, BaseAdapter)][0]
            raise TypeError("Provided adapters must be adapter objects, not '%s'" %
                            str(wrong_object_type))

        self.adapters = adapters

    def forward(self, trials):
        """Apply the adaptors on the parent's trials

        .. seealso::

            :meth:`orion.core.evc.BaseAdapter.forward`
        """
        for adapter in self.adapters:
            trials = adapter.forward(trials)

        return trials

    def backward(self, trials):
        """Apply the adaptors backward and in reverse order on the parent's trials

        .. seealso::

            :meth:`orion.core.evc.BaseAdapter.backward`
        """
        for adapter in self.adapters[::-1]:
            trials = adapter.backward(trials)

        return trials

    def to_dict(self):
        """Return nothing since it is not valid for CompositeAdapter

        .. seealso::

            :meth:`orion.core.evc.adapters.BaseAdapter.to_dict`
            :meth:`orion.core.evc.adapters.CompositeAdapter.configuration`
        """
        pass

    @property
    def configuration(self):
        """Provide the configuration of the adapter.

        .. seealso::

            :meth:`orion.core.evc.BaseAdapter.configuration`
        """
        if len(self.adapters) > 1:
            return [adapter.configuration
                    if len(adapter.configuration) > 1
                    else adapter.configuration[0]
                    for adapter in self.adapters]
        elif self.adapters:
            return self.adapters[0].configuration

        return []


def apply_if_valid(name, trial, callback=None, raise_if_not=True):
    """Detect a parameter in trial and call a callback on it if provided

    Parameters
    ----------
    name: str
        Name of the param to look for
    trial: `orion.core.worker.trial.Trial`
        Instance of trial to investigate
    callback: None of callable
        Function to call with (trial, param) with a parameter is found.
        Defaults to None
    raise_if_not: bool
        raises RuntimeError if no parameter is found.
        Defaults to True.

    Returns
    -------
    bool
        False if parameter is not found and `raise_if_not is False`.
        True if parameter is found and callback is None.
        Else, output of callback(trial, item).

    """
    for param in trial.params:
        if param.name == name:
            return callback is None or callback(trial, param)

    if raise_if_not:
        raise RuntimeError("Provided trial does not have a compatible configuration. "
                           "A dimension named '%s' should be present.\n %s" %
                           (name, trial))

    return False


class DimensionAddition(BaseAdapter):
    """Adapter which adds a new dimension to parent's trials

    This adaptation is based on the assumption that the trials of the parent experiment
    are equivalent if we add the new dimension for a given default value.

    On forward, the adapter add a dimension with provided default value to each trials.
    On backward, the adapter filters trials and only keep with with the default value.

    Attributes
    ----------
    param: instances of `orion.core.worker.trial.Trial.Param`
        A parameter object which defines the name and default value of the name dimension.

    """

    def __init__(self, param):
        """Initialize and instantiate the param if necessary

        Parameters
        ----------
        param: instance of `orion.core.worker.trial.Trial.Param` or `dict`
            A parameter object which defines the name and default value of the name dimension.
            It can be either a dictionary definition or an instance of Param. If the former, then
            a parameter is instantiated from the dictionary.

        """
        if isinstance(param, dict):
            param = Trial.Param(**param)
        elif not isinstance(param, Trial.Param):
            raise TypeError("Invalid param argument type ('%s'). "
                            "Param argument must be a Param object or a dictionnary "
                            "as defined by Trial.Param.to_dict()." %
                            str(type(param)))

        self.param = param

    def forward(self, trials):
        """Add a dimension with provided default value to each trials of the parent

        .. seealso::

            :meth:`orion.core.evc.BaseAdapter.forward`
        """
        adapted_trials = []

        for trial in trials:
            if apply_if_valid(self.param.name, trial, raise_if_not=False):
                raise RuntimeError("Provided trial does not have a compatible configuration. "
                                   "A dimension named '%s' was already present.\n %s" %
                                   (self.param.name, trial))

            adapted_trial = copy.deepcopy(trial)
            adapted_trial.params.append(copy.deepcopy(self.param))
            adapted_trials.append(adapted_trial)

        return adapted_trials

    def backward(self, trials):
        """Filter out trials which have values different than the default one.

        .. seealso::

            :meth:`orion.core.evc.BaseAdapter.backward`
        """
        adapted_trials = []

        def remove_dimension(trial, param):
            """Remove the param and keep the trial if param has default value"""
            if param == self.param:
                adapted_trial = copy.deepcopy(trial)
                del adapted_trial.params[adapted_trial.params.index(self.param)]
                adapted_trials.append(adapted_trial)
                return True

            return False

        for trial in trials:
            apply_if_valid(self.param.name, trial, remove_dimension, raise_if_not=True)

        return adapted_trials

    def to_dict(self):
        """Provide the configuration of the adapter as a dictionary

        .. seealso::

            :meth:`orion.core.evc.adapters.BaseAdapter.to_dict`
        """
        ret = dict(
            of_type=self.__class__.__name__.lower(),
            param=self.param.to_dict())
        return ret


class DimensionDeletion(BaseAdapter):
    """Adapter which remove a dimension to parent's trials

    .. note::

        This adapter is the opposite of `orion.core.evc.adapters.DimensionAddition`.

    This adaptation is based on the assumption that the trials of the children experiment
    are equivalent if we remove the new dimension for a given default value.

    On forward, the adapter filters trials and only keep with with the default value.
    On backward, the adapter add a dimension with provided default value to each trials.

    Attributes
    ----------
    dimension_addition_adapter: instance of `orion.core.evc.adapters.DimensionAddition`
        An adapter to add a new dimension, it is used by DimensionDeletion inversely to remove
        a dimension.

    """

    def __init__(self, param):
        """Initialize and instantiate the param if necessary

        Parameters
        ----------
        param: instance of `orion.core.worker.trial.Trial.Param` or `dict`
            A parameter object which defines the name and default value of the name dimension.
            It can be either a dictionary definition or an instance of Param. If the former, then
            a parameter is instantiated from the dictionary.

        """
        self.dimension_addition_adapter = DimensionAddition(param)

    @property
    def param(self):
        """Parameter containing name and default value"""
        return self.dimension_addition_adapter.param

    def forward(self, trials):
        """Filter out trials which have values different than the default one.

        .. seealso::

            :meth:`orion.core.evc.DimensionAddition.backward`
            :meth:`orion.core.evc.BaseAdapter.forward`
        """
        return self.dimension_addition_adapter.backward(trials)

    def backward(self, trials):
        """Add a dimension with provided default value to each trials of the child.

        .. seealso::

            :meth:`orion.core.evc.DimensionAddition.forward`
            :meth:`orion.core.evc.BaseAdapter.backward`
        """
        return self.dimension_addition_adapter.forward(trials)

    def to_dict(self):
        """Provide the configuration of the adapter as a dictionary

        .. seealso::

            :meth:`orion.core.evc.adapters.BaseAdapter.to_dict`
        """
        ret = self.dimension_addition_adapter.to_dict()
        ret['of_type'] = 'dimensiondeletion'
        return ret


class DimensionPriorChange(BaseAdapter):
    """Adapter which filters parent's trials based on a new prior

    On forward, the adapter filters out trials based on the child's prior.
    On backward, the adapter filters out trials based on the parent's prior.

    Attributes
    ----------
    name: `str`
        Name of the dimension.
    old_prior: `str`
        string definition as parsable by `orion.core.io.space_builder`.
    new_prior: `str`
        string definition as parsable by `orion.core.io.space_builder`.
    old_dimension: instance of `orion.algo.space.Dimension`
        The dimension of the parent experiment
    new_dimension: instance of `orion.algo.space.Dimension`
        The dimension of the child experiment

    """

    def __init__(self, name, old_prior, new_prior):
        """Initialize and instantiate dimensions

        Parameters
        ----------
        name: `str`
            Name of the dimension.
        old_prior: `str`
            string definition as parsable by `orion.core.io.space_builder`.
        new_prior: `str`
            string definition as parsable by `orion.core.io.space_builder`.

        """
        self.name = name
        self.old_prior = old_prior
        self.new_prior = new_prior
        self.old_dimension = DimensionBuilder().build('old', old_prior)
        self.new_dimension = DimensionBuilder().build('new', new_prior)

        if self.old_dimension.shape != self.new_dimension.shape:
            raise NotImplementedError("Oríon does not support yet adaptations on prior "
                                      "shape changes.")

    def forward(self, trials):
        """Filter out trials which have out of bound values based on the child's prior.

        .. seealso::

            :meth:`orion.core.evc.BaseAdapter.forward`
        """
        # pylint: disable=unused-argument
        def is_in_bound(trial, param):
            """Test if param's value is in the new prior's bounds"""
            return param.value in self.new_dimension

        return [trial for trial in trials if apply_if_valid(self.name, trial, callback=is_in_bound)]

    def backward(self, trials):
        """Filter out trials which have out of bound values based on the parent's prior.

        .. seealso::

            :meth:`orion.core.evc.BaseAdapter.backward`
        """
        return DimensionPriorChange(self.name, self.new_prior, self.old_prior).forward(trials)

    def to_dict(self):
        """Provide the configuration of the adapter as a dictionary

        .. seealso::

            :meth:`orion.core.evc.adapters.BaseAdapter.to_dict`
        """
        ret = dict(
            of_type=self.__class__.__name__.lower(),
            name=self.name,
            old_prior=self.old_prior,
            new_prior=self.new_prior)
        return ret


class DimensionRenaming(BaseAdapter):
    """Adapter which change the name of a dimension in parent's trials

    On forward, the adapter change dimensions name from A to B.
    On backward, the adapter change dimensions name from B to A.

    Attributes
    ----------
    old_name: `str`
        Name of the parent's dimension.
    new_name: `str`
        Name of the child's dimension.

    """

    def __init__(self, old_name, new_name):
        """Initialize

        Parameters
        ----------
        old_name: `str`
            Name of the parent's dimension.
        new_name: `str`
            Name of the child's dimension.

        """
        if any(not isinstance(name, str) for name in [old_name, new_name]):
            wrong_object_type = [type(name) for name in [old_name, new_name]
                                 if not isinstance(name, str)][0]
            raise TypeError("Invalid name type '%s'. Names must be strings." %
                            str(wrong_object_type))
        self.old_name = old_name
        self.new_name = new_name

    def forward(self, trials):
        """Change name of dimension `old_name` to `new_name`.

        .. seealso::

            :meth:`orion.core.evc.BaseAdapter.forward`
        """
        # pylint: disable=unused-argument
        def rename(trial, param):
            """Rename param to given new name"""
            param.name = self.new_name
            return True

        adapted_trials = copy.deepcopy(trials)

        for trial in adapted_trials:
            apply_if_valid(self.old_name, trial, callback=rename, raise_if_not=True)

        return adapted_trials

    def backward(self, trials):
        """Change name of dimension `new_name` to `old_name`.

        .. seealso::

            :meth:`orion.core.evc.BaseAdapter.forward`
        """
        return DimensionRenaming(self.new_name, self.old_name).forward(trials)

    def to_dict(self):
        """Provide the configuration of the adapter as a dictionary

        .. seealso::

            :meth:`orion.core.evc.adapters.BaseAdapter.to_dict`
        """
        ret = dict(
            of_type=self.__class__.__name__.lower(),
            old_name=self.old_name,
            new_name=self.new_name)
        return ret


class AlgorithmChange(BaseAdapter):
    """Adapter for changes in algorithm definition

    .. note::

        Current implementation does nothing

    """

    def forward(self, trials):
        """Pass all trials from parent experiment to child experiment

        .. seealso::

            :meth:`orion.core.evc.BaseAdapter.forward`
        """
        return trials

    def backward(self, trials):
        """Pass all trials from child experiment to parent experiment

        .. seealso::

            :meth:`orion.core.evc.BaseAdapter.forward`
        """
        return trials

    def to_dict(self):
        """Provide the configuration of the adapter as a dictionary

        .. seealso::

            :meth:`orion.core.evc.adapters.BaseAdapter.to_dict`
        """
        ret = dict(
            of_type=self.__class__.__name__.lower())
        return ret


class CodeChange(BaseAdapter):
    """Adapter which filters parent's trials based on the type of code change

    This adapter let pass all trials if the code change didn't break the compatibility with parent's
    experiment. If the effect of the change is UNSURE, the trials may only pass from parent to child
    but not from child to parent. This is to ensure parent experiment does not get corrupted with
    possibly incompatible results.

    On forward, the adapter filters out parent's trials if type of code change is BREAK.
    On backward, the adapter filters out child's trials if type of code change is UNSURE or BREAK.

    Attributes
    ----------
    change_type: `str`
        Type of change of the code. Can be one of `CodeChange.NOEFFET`, `CodeChange.BREAK` or
        `CodeChange.UNSURE`.

    """

    NOEFFECT = "noeffect"
    BREAK = "break"
    UNSURE = "unsure"
    types = [NOEFFECT, BREAK, UNSURE]

    def __init__(self, change_type):
        """Initialize and check change type's validity

        Parameters
        ----------
        change_type: `str`
            Type of change of the code. Can be one of `CodeChange.NOEFFET`, `CodeChange.BREAK` or
            `CodeChange.UNSURE`.

        """
        self.validate(change_type)
        self.change_type = change_type

    @classmethod
    def validate(cls, change_type):
        """Validate change type and raise ValueError if invalid"""
        if change_type not in cls.types:
            raise ValueError("Invalid code change type '%s'. Should be one of %s" %
                             (change_type, str(cls.types)))

    def forward(self, trials):
        """Filter out parent's trials if type of code change is BREAK.

        .. seealso::

            :meth:`orion.core.evc.BaseAdapter.forward`
        """
        if self.change_type == self.BREAK:
            return []

        return trials

    def backward(self, trials):
        """Filter out child's trials if type of code change is UNSURE or BREAK.

        .. seealso::

            :meth:`orion.core.evc.BaseAdapter.backward`
        """
        if self.change_type in [self.BREAK, self.UNSURE]:
            return []

        return trials

    def to_dict(self):
        """Provide the configuration of the adapter as a dictionary

        .. seealso::

            :meth:`orion.core.evc.adapters.BaseAdapter.to_dict`
        """
        ret = dict(
            of_type=self.__class__.__name__.lower(),
            change_type=self.change_type)
        return ret


class CommandLineChange(BaseAdapter):
    """Adapter which filters parent's trials based on the type of command line change

    This adapter let pass all trials if the command line change didn't break the compatibility with
    parent's experiment. If the effect of the change is UNSURE, the trials may only pass from parent
    to child but not from child to parent. This is to ensure parent experiment does not get
    corrupted with possibly incompatible results.

    On forward, the adapter filters out parent's trials if type of change is BREAK.
    On backward, the adapter filters out child's trials if type of change is UNSURE or BREAK.

    Attributes
    ----------
    change_type: `str`
        Type of change of the command line. Can be one of `CommandLineChange.NOEFFET`,
        `CommandLineChange.BREAK` or `CommandLineChange.UNSURE`.

    """

    NOEFFECT = "noeffect"
    BREAK = "break"
    UNSURE = "unsure"
    types = [NOEFFECT, BREAK, UNSURE]

    def __init__(self, change_type):
        """Initialize and check change type's validity

        Parameters
        ----------
        change_type: `str`
            Type of change of the command line. Can be one of `Change.NOEFFET`,
            `CommandLineChange.BREAK` or `CommandLineChange.UNSURE`.

        """
        self.validate(change_type)
        self.change_type = change_type

    @classmethod
    def validate(cls, change_type):
        """Validate change type and raise ValueError if invalid"""
        if change_type not in cls.types:
            raise ValueError("Invalid cli change type '%s'. Should be one of %s" %
                             (change_type, str(cls.types)))

    def forward(self, trials):
        """Filter out parent's trials if type of cli change is BREAK.

        .. seealso::

            :meth:`orion.core.evc.BaseAdapter.forward`
        """
        if self.change_type == self.BREAK:
            return []

        return trials

    def backward(self, trials):
        """Filter out child's trials if type of cli change is UNSURE or BREAK.

        .. seealso::

            :meth:`orion.core.evc.BaseAdapter.backward`
        """
        if self.change_type in [self.BREAK, self.UNSURE]:
            return []

        return trials

    def to_dict(self):
        """Provide the configuration of the adapter as a dictionary

        .. seealso::

            :meth:`orion.core.evc.adapters.BaseAdapter.to_dict`
        """
        ret = dict(
            of_type=self.__class__.__name__.lower(),
            change_type=self.change_type)
        return ret


class ScriptConfigChange(BaseAdapter):
    """Adapter which filters parent's trials based on the type of user's script's config change

    This adapter let pass all trials if the change in the user's script's configuration file
    didn't break the compatibility with parent's experiment. If the effect of the change is UNSURE,
    the trials may only pass from parent to child but not from child to parent. This is to ensure
    parent experiment does not get corrupted with possibly incompatible results.

    On forward, the adapter filters out parent's trials if type of change is BREAK.
    On backward, the adapter filters out child's trials if type of change is UNSURE or BREAK.

    Attributes
    ----------
    change_type: `str`
        Type of change of the command line. Can be one of `ScriptConfigChange.NOEFFET`,
        `ScriptConfigChange.BREAK` or `ScriptConfigChange.UNSURE`.

    """

    NOEFFECT = "noeffect"
    BREAK = "break"
    UNSURE = "unsure"
    types = [NOEFFECT, BREAK, UNSURE]

    def __init__(self, change_type):
        """Initialize and check change type's validity

        Parameters
        ----------
        change_type: `str`
            Type of change of the script's config. Can be one of `ScriptConfigChange.NOEFFET`,
            `ScriptConfigChange.BREAK` or `ScriptConfigChange.UNSURE`.

        """
        self.validate(change_type)
        self.change_type = change_type

    @classmethod
    def validate(cls, change_type):
        """Validate change type and raise ValueError if invalid"""
        if change_type not in cls.types:
            raise ValueError("Invalid script's config change type '%s'. Should be one of %s" %
                             (change_type, str(cls.types)))

    def forward(self, trials):
        """Filter out parent's trials if type of script's config change is BREAK.

        .. seealso::

            :meth:`orion.core.evc.BaseAdapter.forward`
        """
        if self.change_type == self.BREAK:
            return []

        return trials

    def backward(self, trials):
        """Filter out child's trials if type of script's config change is UNSURE or BREAK.

        .. seealso::

            :meth:`orion.core.evc.BaseAdapter.backward`
        """
        if self.change_type in [self.BREAK, self.UNSURE]:
            return []

        return trials

    def to_dict(self):
        """Provide the configuration of the adapter as a dictionary

        .. seealso::

            :meth:`orion.core.evc.adapters.BaseAdapter.to_dict`
        """
        ret = dict(
            of_type=self.__class__.__name__.lower(),
            change_type=self.change_type)
        return ret


# pylint: disable=too-few-public-methods,abstract-method
class Adapter(BaseAdapter, metaclass=Factory):
    """Class used to inject dependency on an adapter implementation.

    .. seealso:: `orion.core.utils.Factory` metaclass and `BaseAlgorithm` interface.
    """

    @classmethod
    def build(cls, adapter_dicts):
        """Builder method for a list of adapters.

        Parameters
        ----------
        adapter_dicts: list of `dict`
            List of adapter representation in dictionary form as expected to be saved in a database.

        Returns
        -------
        `orion.core.evc.adapters.CompositeAdapter`
            An adapter which may contain many adapters

        """
        adapters = []
        for adapter_dict in adapter_dicts:
            if isinstance(adapter_dict, (list, tuple)):
                adapter = Adapter.build(adapter_dict)
            else:
                new_adapter_dict = copy.deepcopy(adapter_dict)
                new_adapter_dict['of_type'] = ('orion.core.evc.adapters',
                                               new_adapter_dict['of_type'])
                adapter = cls(**new_adapter_dict)
            adapters.append(adapter)

        return CompositeAdapter(*adapters)
