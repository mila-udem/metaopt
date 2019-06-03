# -*- coding: utf-8 -*-
"""Implementation of a test concept for tests purposes"""

from orion.wrappers import ConceptTest


class OtherConceptImpl(ConceptTest):
    """Exist just so that we can initialize two concepts inside a dictionary"""

    def __init__(self, argument=0):
        """Initialize the concept"""
        super(OtherConceptImpl, self).__init__(argument=argument)
