import datetime
import inspect
from dataclasses import Field, asdict, dataclass, field, fields, is_dataclass, replace
from typing import (
    AbstractSet,
    Any,
    Dict,
    ItemsView,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from orion.core.utils.flatten import flatten

ConfigType = TypeVar("ConfigType", bound="Config")


class SerializableMixin:
    """ Mixin for both the `Config` and `ImmutableConfig` dataclasses below. """

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """ Returns the item / attribute with the given name if present, else `default`.
        """
        return getattr(self, key, default)

    def items(self) -> Iterable[Tuple[str, Any]]:
        """ Returns an iterator over the items/attributes in this object. """
        for field in fields(self):
            yield field.name, getattr(self, field.name)

    def asdict(self) -> Dict[str, Any]:
        """ Returns a dict version of this dataclass. """
        return asdict(self)

    def replace(self, **kwargs):
        """ Replaces one or more values in this dataclass. """
        return replace(self, **kwargs)

    def flatten(self) -> Dict:
        """Returns a serialized, flattened version of this dataclass.

        [extended_summary]

        Returns
        -------
        Dict
            [description]
        """
        return flatten(self.asdict())

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """ Create an instance of this class from the given dictionary. """
        # TODO: Unflatten the dict?
        # config_dict = unflatten(config_dict)
        for field in fields(cls):
            if (
                field.name in config_dict
                and is_dataclass(field.type)
                and inspect.isclass(field.type)
                and issubclass(field.type, Config)
            ):
                # The field is itself a dataclass type, so we recursively call from_dict
                # passing in the corresponding value.
                field_config_dict: Dict[str, Any] = config_dict[field.name]
                field_type: Type[SerializableMixin] = field.type
                field_value = field_type.from_dict(field_config_dict)

                # Replace the dict for that field with the deserialized value:
                config_dict[field.name] = field_value
        return cls(**config_dict)  # type: ignore


@dataclass(unsafe_hash=True)
class Config(SerializableMixin, MutableMapping):
    """ Base class for the various Config objects: StorageConfig, ExperimentConfig, etc.
    """


@dataclass(frozen=True, unsafe_hash=True)
class ImmutableConfig(SerializableMixin, Mapping):
    """ Base class for the various Config objects: StorageConfig, ExperimentConfig, etc.
    """


@dataclass(unsafe_hash=True)
class DatabaseConfig(Config):
    type: str


@dataclass(unsafe_hash=True)
class StorageConfig(Config):
    type: str
    database: DatabaseConfig


@dataclass(unsafe_hash=True)
class Refers(Config):
    parent_id: Optional[str]
    root_id: int
    adapter: List


@dataclass(unsafe_hash=True)
class MetaData(Config):
    user: str
    orion_version: str
    datetime: datetime.datetime


@dataclass()
class ExperimentInfo(Config):
    """ A little dataclass used to get a typed result from querying the storage. """

    name: str
    refers: Refers
    metadata: MetaData
    pool_size: int
    max_trials: int
    max_broken: int
    version: int
    space: Dict[str, str]
    algorithms: Dict[str, Dict]
    producer: Dict[str, Union[Dict[str, Any], str]]
    working_dir: str
    _id: int
    # IDEA: Store a reference to the Storage object associated with the experiment in
    # case we want to support multi-storage in the future.
    _storage: Optional[StorageConfig] = field(default=None, hash=False, repr=False)

