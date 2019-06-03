# -*- coding: utf-8 -*-
"""
:mod:`orion.core.io.database.mongodb` -- Wrapper for MongoDB
============================================================

.. module:: database
   :platform: Unix
   :synopsis: Implement :class:`orion.core.io.database.AbstractDB` for MongoDB.

"""
import functools

import pymongo

from orion.core.io.database import (
    AbstractDB, DatabaseError, DuplicateKeyError)


AUTH_FAILED_MESSAGES = [
    "auth failed",
    "Authentication failed."]

DUPLICATE_KEY_MESSAGES = [
    "duplicate key error"]


def mongodb_exception_wrapper(method):
    """Convert pymongo exceptions to generic exception types defined in src.core.io.database.

    Current exception types converted:
    pymongo.errors.DuplicateKeyError -> DuplicateKeyError
    pymongo.errors.BulkWriteError[DUPLICATE_KEY_MESSAGES] -> DuplicateKeyError
    pymongo.errors.ConnectionFailure -> DatabaseError
    pymongo.errors.OperationFailure(AUTH_FAILED_MESSAGES) -> DatabaseError

    """
    @functools.wraps(method)
    def _decorator(self, *args, **kwargs):

        try:
            rval = method(self, *args, **kwargs)
        except pymongo.errors.DuplicateKeyError as e:
            raise DuplicateKeyError(str(e)) from e
        except pymongo.errors.BulkWriteError as e:
            for error in e.details['writeErrors']:
                if any(m in error["errmsg"] for m in DUPLICATE_KEY_MESSAGES):
                    raise DuplicateKeyError(error["errmsg"]) from e

            raise
        except pymongo.errors.ConnectionFailure as e:
            raise DatabaseError("Connection Failure: database not found on "
                                "specified uri") from e
        except pymongo.errors.OperationFailure as e:
            if any(m in str(e) for m in AUTH_FAILED_MESSAGES):
                raise DatabaseError("Authentication Failure: bad credentials") from e

            raise

        return rval

    return _decorator


class MongoDB(AbstractDB):
    """Wrap MongoDB with three primary methods `read`, `write`, `remove`.

    Attributes
    ----------
    host : str
       Hostname or MongoDB compliant full credentials+address+database
       specification.

    Information on MongoDB `connection string
    <https://docs.mongodb.com/manual/reference/connection-string/>`_.

    .. seealso:: :class:`orion.core.io.database.AbstractDB` for more on attributes.

    """

    def __init__(self, host='localhost', name=None,
                 port=None, username=None, password=None):
        """Init method, see attributes of :class:`AbstractDB`."""
        self.options = {'authSource': name}
        self.uri = None

        super(MongoDB, self).__init__(host=host, name=name, port=port,
                                      username=username, password=password)

    @mongodb_exception_wrapper
    def initiate_connection(self):
        """Connect to database, unless MongoDB `is_connected`.

        :raises :exc:`DatabaseError`: if connection or authentication fails

        """
        if self.is_connected:
            return

        self._sanitize_attrs()

        self._conn = pymongo.MongoClient(self.uri if self.uri else self.host,
                                         port=self.port,
                                         username=self.username,
                                         password=self.password,
                                         **self.options)
        self._db = self._conn[self.name]
        self._db.command('ismaster')  # .. seealso:: :meth:`is_connected`

    @property
    def is_connected(self):
        """True, if practical connection has been achieved.

        .. note:: MongoDB does not do this automatically when creating the client.
        """
        try:
            self._db.command('ismaster')
        except (pymongo.errors.ConnectionFailure,
                pymongo.errors.OperationFailure,
                TypeError, AttributeError):
            _is_connected = False
        else:
            _is_connected = True
        return _is_connected

    def close_connection(self):
        """Disconnect from database.

        .. note:: Doesn't really do anything because MongoDB reopens connection,
           when a client or client-derived object is accessed.
        """
        self._conn.close()

    def ensure_index(self, collection_name, keys, unique=False):
        """Create given indexes if they do not already exist in database.

        .. seealso:: :meth:`AbstractDB.ensure_index` for argument documentation.

        """
        # MongoDB's `create_index()` is idempotent, which means it will only
        # create new indexes if they do not already exists. That's why we do
        # not need to verify if indexes already exists.
        dbcollection = self._db[collection_name]

        keys = self._convert_index_keys(keys)

        dbcollection.create_index(keys, unique=unique, background=True)

    def _convert_index_keys(self, keys):
        """Convert index keys to MongoDB ones."""
        if not isinstance(keys, (list, tuple)):
            keys = [(keys, self.ASCENDING)]

        converted_keys = []
        for key, sort_order in keys:
            converted_keys.append((key, self._convert_sort_order(sort_order)))

        return converted_keys

    def _convert_sort_order(self, sort_order):
        """Convert generic `AbstractDB` sort orders to MongoDB ones."""
        if sort_order is self.ASCENDING:
            return pymongo.ASCENDING
        elif sort_order is self.DESCENDING:
            return pymongo.DESCENDING
        else:
            raise RuntimeError("Invalid database sort order %s" %
                               str(sort_order))

    @mongodb_exception_wrapper
    def write(self, collection_name, data, query=None):
        """Write new information to a collection. Perform insert or update.

        .. seealso:: :meth:`AbstractDB.write` for argument documentation.

        """
        dbcollection = self._db[collection_name]

        if query is None:
            # We can assume that we do not want to update.
            # So we do insert_many instead.
            if type(data) not in (list, tuple):
                data = [data]
            result = dbcollection.insert_many(documents=data)
            return result.acknowledged

        update_data = {'$set': data}

        result = dbcollection.update_many(filter=query,
                                          update=update_data,
                                          upsert=True)
        return result.acknowledged

    def read(self, collection_name, query=None, selection=None):
        """Read a collection and return a value according to the query.

        .. seealso:: :meth:`AbstractDB.read` for argument documentation.

        """
        dbcollection = self._db[collection_name]

        cursor = dbcollection.find(query, selection)
        dbdocs = list(cursor)

        return dbdocs

    @mongodb_exception_wrapper
    def read_and_write(self, collection_name, query, data, selection=None):
        """Read a collection's document and update the found document.

        Returns the updated document, or None if nothing found.

        .. seealso:: :meth:`AbstractDB.read_and_write` for
                     argument documentation.

        """
        dbcollection = self._db[collection_name]

        update_data = {'$set': data}

        dbdoc = dbcollection.find_one_and_update(
            query, update_data, projection=selection,
            return_document=pymongo.ReturnDocument.AFTER)

        return dbdoc

    def count(self, collection_name, query=None):
        """Count the number of documents in a collection which match the `query`.

        .. seealso:: :meth:`AbstractDB.count` for argument documentation.

        """
        dbcollection = self._db[collection_name]
        if hasattr(dbcollection, 'count_documents'):
            return dbcollection.count_documents(filter=query if query else {})

        return dbcollection.count(filter=query)

    def remove(self, collection_name, query):
        """Delete from a collection document[s] which match the `query`.

        .. seealso:: :meth:`AbstractDB.remove` for argument documentation.

        """
        dbcollection = self._db[collection_name]

        result = dbcollection.delete_many(filter=query)
        return result.acknowledged

    # pylint: disable=access-member-before-definition, attribute-defined-outside-init
    def _sanitize_attrs(self):
        """Sanitize attributes using MongoDB's 'uri_parser' module."""
        try:
            # Host can be a valid MongoDB URI
            settings = pymongo.uri_parser.parse_uri(self.host)
        except pymongo.errors.InvalidURI:  # host argument was a hostname
            if self.port is None:
                self.port = pymongo.MongoClient.PORT
        else:  # host argument was a URI
            # Arguments in MongoClient overwrite elements from URI

            self.uri = self.host
            self.host, _port = settings['nodelist'][0]
            if settings['database'] is not None:
                self.name = settings['database']
            if _port is not None:
                self.port = _port
            if settings['username'] is not None:
                self.username = settings['username']
            if settings['password'] is not None:
                self.password = settings['password']

            # Use new self.name if authSource not specified in URI
            self.options['authSource'] = settings['options'].get('authsource', self.name)
