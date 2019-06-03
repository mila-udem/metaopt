#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage and tests for :mod:`orion.core.io.database`."""

import pytest

from orion.core.io.database import Database, ReadOnlyDB
from orion.core.io.database.mongodb import MongoDB


@pytest.mark.usefixtures("null_db_instances")
class TestDatabaseFactory(object):
    """Test the creation of a determinate `Database` type, by a complete spefication
    of a database by-itself (this on which every `Database` acts on as part
    of its being, attributes of an `AbstractDB`) and for-itself (what essentially
    differentiates one concrete `Database` from one other).

    """

    def test_empty_first_call(self):
        """Should not be able to make first call without any arguments.

        Hegelian Ontology Primer
        ------------------------

        Type indeterminate <-> type abstracted from its property <-> No type
        """
        with pytest.raises(NotImplementedError) as exc_info:
            Database()
        assert 'No' in str(exc_info.value)

    def test_notfound_type_first_call(self):
        """Raise when supplying not implemented wrapper name."""
        with pytest.raises(NotImplementedError) as exc_info:
            Database('notfound')
        assert 'AbstractDB' in str(exc_info.value)

    def test_instantiation_and_singleton(self):
        """Test create just one object, that object persists between calls."""
        database = Database('MongoDB', name='orion_test',
                            username='user', password='pass')

        assert isinstance(database, MongoDB)
        assert database.instance is MongoDB()
        assert database is Database()
        with pytest.raises(ValueError) as exc_info:
            Database('fire', [], {'it_matters': 'it\'s singleton'})
        assert 'singleton' in str(exc_info.value)


@pytest.mark.usefixtures("null_db_instances", "clean_db")
class TestReadOnlyDatabase(object):
    """Test coherence of read-only database and its wrapped database."""

    def test_valid_attributes(self, create_db_instance):
        """Test attributes are coherent from view and wrapped database."""
        database = create_db_instance
        readonly_database = ReadOnlyDB(database)

        assert readonly_database.is_connected == database.is_connected
        assert readonly_database.host == database.host
        assert readonly_database.port == database.port
        assert readonly_database.username == database.username
        assert readonly_database.password == database.password

    def test_read(self, create_db_instance):
        """Test read is coherent from view and wrapped database."""
        database = create_db_instance
        readonly_database = ReadOnlyDB(database)

        args = {"collection_name": "trials", "query": {"experiment": "supernaedo2"}}
        readonly_result = readonly_database.read(**args)
        result = database.read(**args)

        assert len(result) > 0  # Otherwise the test is pointless
        assert readonly_result == result

    def test_invalid_attributes(self, create_db_instance):
        """Test that attributes for writing are not accessible."""
        database = create_db_instance
        readonly_database = ReadOnlyDB(database)

        # Test that database.ensure_index indeed exists
        database.ensure_index
        with pytest.raises(AttributeError):
            readonly_database.ensure_index

        # Test that database.write indeed exists
        database.write
        with pytest.raises(AttributeError):
            readonly_database.write

        # Test that database.read_and_write indeed exists
        database.read_and_write
        with pytest.raises(AttributeError):
            readonly_database.read_and_write

        # Test that database.remove indeed exists
        database.remove
        with pytest.raises(AttributeError):
            readonly_database.remove
