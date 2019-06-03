# -*- coding: utf-8 -*-
"""
:mod:`orion.core.utils` -- Package-wide useful routines
=======================================================

.. module:: utils
   :platform: Unix
   :synopsis: Helper functions useful in possibly all :mod:`orion.core`'s modules.
"""

from abc import ABCMeta
from collections import defaultdict
from glob import glob
from importlib import import_module
import logging
import os

import pkg_resources


log = logging.getLogger(__name__)


# Define type of arbitrary nested defaultdicts
def nesteddict():
    """Extend defaultdict to arbitrary nested levels."""
    return defaultdict(nesteddict)


def get_qualified_name(package, name):
    """Return the qualified name of the module and the class inside that module.
    Ex. package: orion.algo.random
    name: Random
    returns: orion.algo.random.random
    """
    return package.lower() + '.' + name.lower()


class SingletonError(ValueError):
    """Exception to be raised when someone provides arguments to build
    an object from a already-instantiated `SingletonType` class.
    """

    def __init__(self):
        """Pass the same constant message to ValueError underneath."""
        super().__init__("A singleton instance has already been instantiated.")


class SingletonType(type):
    """Metaclass that implements the singleton pattern for a Python class."""

    def __init__(cls, name, bases, dictionary):
        """Create a class instance variable and initiate it to None object."""
        super(SingletonType, cls).__init__(name, bases, dictionary)
        cls.instance = None

    def __call__(cls, *args, **kwargs):
        """Create an object if does not already exist, otherwise return what there is."""
        if cls.instance is None:
            cls.instance = super(SingletonType, cls).__call__(*args, **kwargs)
        elif args or kwargs:
            raise ValueError("A singleton instance has already been instantiated.")
        return cls.instance


class AbstractSingletonType(SingletonType, ABCMeta):
    """This will create singleton base classes, that need to be subclassed and implemented."""

    pass


class Factory(ABCMeta):
    """Instantiate appropriate wrapper for the infrastructure based on input
    argument, ``of_type``.

    Attributes
    ----------
    types : list of subclasses of ``cls.__base__``
       Updated to contain all possible implementations currently. Check out code.
    typenames : list of str
       Names of implemented wrapper classes, correspond to possible ``of_type``
       values.

    """

    def __init__(cls, names, bases, dictionary):
        """Search in directory for attribute names subclassing `bases[0]`"""
        super(Factory, cls).__init__(names, bases, dictionary)

        cls.modules = []
        base = import_module(cls.__base__.__module__)

        if hasattr(cls, "implementation_module"):
            import_module(cls.implementation_module)

        try:
            py_files = glob(os.path.abspath(os.path.join(base.__path__[0] + '/**/',
                                                         '[A-Za-z]*.py')), recursive=True)

            def _f(path):
                name = 'orion' + path.split('orion')[-1]
                return name.replace('/', '.')[:-3]

            py_mods = map(_f, py_files)

            for py_mod in py_mods:
                cls.modules.append(import_module(py_mod))
        except AttributeError:
            # This means that base class and implementations reside in a module
            # itself and not a subpackage.
            pass

        # Get types advertised through entry points!
        for entry_point in pkg_resources.iter_entry_points(cls.__base__.__name__):
            entry_point.load()
            log.debug("Found a %s %s from distribution: %s=%s",
                      entry_point.name, cls.__name__,
                      entry_point.dist.project_name, entry_point.dist.version)

        # Get types visible from base module or package, but internal
        def get_all_subclasses(parent):
            """Get set of subclasses recursively"""
            subclasses = set()
            for subclass in parent.__subclasses__():
                subclasses.add(subclass)
                subclasses |= get_all_subclasses(subclass)

            return subclasses

        cls.types = list(get_all_subclasses(cls.__base__))
        cls.types = [class_ for class_ in cls.types if class_.__name__ != cls.__name__]
        cls.typenames = list(map(lambda x: get_qualified_name(x.__module__,
                                                              x.__name__).lower(), cls.types))
        log.debug("Implementations found: %s", cls.typenames)

    def __call__(cls, of_type, *args, **kwargs):
        """Create an object, instance of ``cls.__base__``, on first call.

        :param of_type: Name of class, subclass of ``cls.__base__``, wrapper
           of a database framework that will be instantiated on the first call.
        :param args: positional arguments to initialize ``cls.__base__``'s instance (if any)
        :param kwargs: keyword arguments to initialize ``cls.__base__``'s instance (if any)

        .. seealso::
           `Factory.typenames` for values of argument `of_type`.

        .. seealso::
           Attributes of ``cls.__base__`` and ``cls.__base__.__init__`` for
           values of `args` and `kwargs`.

        .. note:: New object is saved as `Factory`'s internal state.

        :return: The object which was created on the first call.
        """
        module, name = of_type
        qualified_name = get_qualified_name(module, name).lower()

        for inherited_class in cls.types:
            inherited_qualified_name = get_qualified_name(inherited_class.__module__,
                                                          inherited_class.__name__).lower()
            if inherited_qualified_name == qualified_name or \
               module == inherited_qualified_name:
                return inherited_class.__call__(*args, **kwargs)

        error = "Could not find implementation of {0}, type = '{1}'".format(
            cls.__base__.__name__, qualified_name)
        error += "\nCurrently, there is an implementation for types:\n"
        error += str(cls.typenames)
        raise NotImplementedError(error)


class SingletonFactory(AbstractSingletonType, Factory):
    """Wrapping `Factory` with `SingletonType`. Keep compatibility with `AbstractSingletonType`."""

    pass


class Concept(object):  # pylint: disable=too-few-public-methods
    """Provide a base class for an abstract Concept (like an Algorithm or a DataAnalyser)."""

    # pylint: disable=unused-argument
    def __init__(self, *args, **kwargs):
        """Initialize the object and instanciate any parameters inside the configuration dictionary
        to the correct type using the custom factory for this particular Concept.
        """
        # Get base class information
        self.base_class = type(self).__base__
        self.name = self.base_class.name  # Descriptor name for log outputs

        log.debug("Creating %s object of %s type with parameters:\n%s",
                  self.name, type(self).__name__, kwargs)

        for varname, param in kwargs.items():
            setattr(self, varname, param)


class Wrapper(object):
    """Provide a base class for wrappers to act as proxy for the wrapped object"""

    def __init__(self, *args, **kwargs):
        """Initialize the wrapped object by creating a Factory for the wrapped object's type
        then instantiating the object with the config passed.
        """
        self.instance = None
        self.module = self._get_module()
        self.factory = self.factory_type('Factory', (self.wraps,), {})

        for key, item in kwargs.items():

            if isinstance(item, dict) and item:
                item = self._instantiate_dict(item, *args)

            elif isinstance(item, str):
                item = self._instantiate_str(item, *args)

            setattr(self, key, item)

    @property
    def factory_type(self):
        """Return the type of factory to be use for the wrapped type"""
        return Factory

    @property
    def wraps(self):
        """Return the type of object this wrapper wraps"""
        raise NotImplementedError

    @property
    def __class__(self):
        """Change the type check behavior of the wrapper to check has the
        underlying instance type
        """
        return self.instance.__class__

    def __instancecheck__(self, instance):
        """Proxy the instance check to the database object"""
        return isinstance(self.instance, instance)

    def __subclasscheck__(self, instance):
        """Proxy the subclass check to the database"""
        return issubclass(self.instance, instance)

    def __getattr__(self, name):
        """Proxy the call to `__getattr__` to the underlying object. If the
        wrapped instance does not have this attribute, return it from the wrapper
        instead
        """
        if name == 'instance':
            raise AttributeError()

        return getattr(self.instance, name)

    def _get_module(self):
        # Implementation module might be redefined at concept or wrapper level
        module = getattr(self, "implementation_module", self.__module__)

        return module

    def _instantiate_dict(self, item, *args):
        has_instantiated = False
        for subkey, subitem in item.items():
            try:
                # Check for dictionary, because Factory needs a mapping
                if isinstance(subitem, dict):
                    # Build the module's name for the object
                    qualified_name = get_qualified_name(self.module, subkey)

                    item = self.factory((qualified_name, subkey), *args, **subitem)

                    # We can't instantiate more than one object of the wrapped type
                    if has_instantiated:
                        raise RuntimeError("Can only instantiate once")

                    has_instantiated = True
                else:
                    # If we don't have a dictionary, this means its an attribute
                    setattr(self, subkey, subitem)

            except NotImplementedError:
                # If we couldn't instantiate it, this means its an attribute
                setattr(self, subkey, subitem)

        if not has_instantiated:
            raise NotImplementedError("No implementation detected for type {}".format(self.wraps))

        return item

    def _instantiate_str(self, item, *args):
        qualified_tuple = (get_qualified_name(self.module, item), item)
        qualified_name = get_qualified_name(*qualified_tuple)

        if qualified_name in self.factory.typenames:
            item = self.factory(qualified_tuple, *args)

        return item
