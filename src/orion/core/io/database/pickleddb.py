# -*- coding: utf-8 -*-
"""
Pickled Database
================

Implement permanent version of :class:`orion.core.io.database.ephemeraldb.EphemeralDB`.

"""

import logging
import os
import pickle
from contextlib import contextmanager
from pickle import PicklingError

import psutil
from filelock import FileLock, SoftFileLock, Timeout

import orion.core
from orion.core.io.database import AbstractDB, DatabaseTimeout
from orion.core.io.database.ephemeraldb import EphemeralDB

log = logging.getLogger(__name__)

DEFAULT_HOST = os.path.join(orion.core.DIRS.user_data_dir, "orion", "orion_db.pkl")

TIMEOUT_ERROR_MESSAGE = """\
Could not acquire lock for PickledDB after {} seconds.

This is likely due to one or many of the following scenarios:

1. There is a large amount of workers and many simultaneous queries. This typically occurs
   when the task to optimize is short (few minutes). Try to reduce the amount of workers
   at least below 50.

2. The database is growing large with thousands of trials and many experiments.
   If so, you can use a different PickleDB (different file, that is, different `host`)
   for each experiment seperately to alleviate this issue.

3. The filesystem is slow. Parallel filesystems on HPC often suffer from
   large pool of users generating frequent I/O. In this case try using a separate
   partition that may be less affected.

If you cannot solve the issues listed above that are causing timeouts, you
may need to setup the MongoDB backend for better performance.
See https://orion.readthedocs.io/en/stable/install/database.html
"""


def find_unpickable_doc(dict_of_dict):
    """Look for a dictionary that cannot be pickled."""
    for name, collection in dict_of_dict.items():
        documents = collection.find()

        for doc in documents:
            try:
                pickle.dumps(doc)

            except (PicklingError, AttributeError):
                return name, doc

    return None, None


def find_unpickable_field(doc):
    """Look for a field in a dictionary that cannot be pickled"""
    if not isinstance(doc, dict):
        doc = doc.to_dict()

    for k, v in doc.items():
        try:
            pickle.dumps(v)

        except (PicklingError, AttributeError):
            return k, v

    return None, None


# pylint: disable=too-many-public-methods
class PickledDB(AbstractDB):
    """Pickled EphemeralDB to support permanancy and concurrency

    This is a very simple and inefficient implementation of a permanent database on disk for Oríon.
    The data is loaded from disk for every operation, and every operation is protected with a
    filelock.

    Parameters
    ----------
    host: str
        File path to save pickled ephemeraldb.  Default is {user data dir}/orion/orion_db.pkl ex:
        $HOME/.local/share/orion/orion_db.pkl
    timeout: int
        Maximum number of seconds to wait for the lock before raising DatabaseTimeout.
        Default is 60.

    """

    # pylint: disable=unused-argument
    def __init__(self, host=DEFAULT_HOST, timeout=60, *args, **kwargs):
        super(PickledDB, self).__init__(host)

        self.host = os.path.abspath(host)

        self.timeout = timeout

        if os.path.dirname(host):
            os.makedirs(os.path.dirname(host), exist_ok=True)

    @property
    def is_connected(self):
        """Return true, always."""
        return True

    def initiate_connection(self):
        """Do nothing"""
        pass

    def close_connection(self):
        """Do nothing"""
        pass

    def ensure_index(self, collection_name, keys, unique=False):
        """Create given indexes if they do not already exist in database.

        Indexes are only created if `unique` is True.
        """
        with self.locked_database() as database:
            database.ensure_index(collection_name, keys, unique=unique)

    def index_information(self, collection_name):
        """Return dict of names and sorting order of indexes"""
        with self.locked_database(write=False) as database:
            return database.index_information(collection_name)

    def drop_index(self, collection_name, name):
        """Remove index from the database"""
        with self.locked_database() as database:
            return database.drop_index(collection_name, name)

    def write(self, collection_name, data, query=None):
        """Write new information to a collection. Perform insert or update.

        .. seealso:: :meth:`orion.core.io.database.AbstractDB.write` for argument documentation.

        """
        with self.locked_database() as database:
            return database.write(collection_name, data, query=query)

    def read(self, collection_name, query=None, selection=None):
        """Read a collection and return a value according to the query.

        .. seealso:: :meth:`orion.core.io.database.AbstractDB.read` for argument documentation.

        """
        with self.locked_database(write=False) as database:
            return database.read(collection_name, query=query, selection=selection)

    def read_and_write(self, collection_name, query, data, selection=None):
        """Read a collection's document and update the found document.

        Returns the updated document, or None if nothing found.

        .. seealso:: :meth:`orion.core.io.database.AbstractDB.read_and_write` for
                     argument documentation.

        """
        with self.locked_database() as database:
            return database.read_and_write(
                collection_name, query=query, data=data, selection=selection
            )

    def count(self, collection_name, query=None):
        """Count the number of documents in a collection which match the `query`.

        .. seealso:: :meth:`orion.core.io.database.AbstractDB.count` for argument documentation.

        """
        with self.locked_database(write=False) as database:
            return database.count(collection_name, query=query)

    def remove(self, collection_name, query):
        """Delete from a collection document[s] which match the `query`.

        .. seealso:: :meth:`orion.core.io.database.AbstractDB.remove` for argument documentation.

        """
        with self.locked_database() as database:
            return database.remove(collection_name, query=query)

    def _get_database(self):
        """Read fresh DB state from pickled file"""
        if not os.path.exists(self.host):
            return EphemeralDB()

        with open(self.host, "rb") as f:
            data = f.read()
            if not data:
                database = EphemeralDB()
            else:
                database = pickle.loads(data)

        return database

    def _dump_database(self, database):
        """Write pickled DB on disk"""
        tmp_file = self.host + ".tmp"

        try:
            with open(tmp_file, "wb") as f:
                pickle.dump(database, f)

        except (PicklingError, AttributeError):
            # pylint: disable=protected-access
            collection, doc = find_unpickable_doc(database._db)
            log.error(
                "Document in (collection: %s) is not pickable\ndoc: %s",
                collection,
                doc.to_dict(),
            )

            key, value = find_unpickable_field(doc)
            log.error("because (value %s) in (field: %s) is not pickable", value, key)
            raise

        os.rename(tmp_file, self.host)

    @contextmanager
    def locked_database(self, write=True):
        """Lock database file during wrapped operation call."""
        lock = _create_lock(self.host + ".lock")

        try:
            with lock.acquire(timeout=self.timeout):
                database = self._get_database()

                yield database

                if write:
                    self._dump_database(database)
        except Timeout as e:
            raise DatabaseTimeout(TIMEOUT_ERROR_MESSAGE.format(self.timeout)) from e


local_file_systems = ["ext2", "ext3", "ext4", "ntfs"]


def _fs_support_globalflock(file_system):
    if file_system.fstype == "lustre":
        return ("flock" in file_system.opts) and ("localflock" not in file_system.opts)

    elif file_system.fstype == "beegfs":
        return "tuneUseGlobalFileLocks" in file_system.opts

    elif file_system.fstype == "gpfs":
        return True

    elif file_system.fstype == "nfs":
        return False

    return file_system.fstype in local_file_systems


def _find_mount_point(path):
    """Finds the mount point used to access `path`."""
    path = os.path.abspath(path)
    while not os.path.ismount(path):
        path = os.path.dirname(path)

    return path


def _get_fs(path):
    """Gets info about the filesystem on which `path` lives."""
    mount = _find_mount_point(path)

    for file_system in psutil.disk_partitions(True):
        if file_system.mountpoint == mount:
            return file_system

    return None


def _create_lock(path):
    """Create lock based on file system capabilities

    Determine if we can rely on the fcntl module for locking files.
    Otherwise, fallback on using the directory creation atomicity as a locking mechanism.
    """
    file_system = _get_fs(path)

    if _fs_support_globalflock(file_system):
        log.debug("Using flock.")
        return FileLock(path)
    else:
        log.debug("Cluster does not support flock. Falling back to softfilelock.")
        return SoftFileLock(path)
