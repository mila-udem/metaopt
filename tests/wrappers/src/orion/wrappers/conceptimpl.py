# -*- coding: utf-8 -*-
"""Implementation of a test concept for tests purposes"""

from orion.wrappers import ConceptTest


class ConceptImpl(ConceptTest):
    """Basic implementation of a Concept"""

    def __init__(self, argument=0):
        """Initialize that concept"""
        super(ConceptImpl, self).__init__(argument=argument)
