# -*- coding: utf-8 -*-
"""Implementation of a sub-wrapper for tests purposes"""

from orion.core.utils import Wrapper
from orion.wrappers import ConceptTest


class DifferentWrapper(Wrapper):
    """Provide a wrapper that initialize concepts in another module
    through implementation_module
    """

    implementation_module = "orion.wrappers"

    def __init__(self, config):
        """Initialize the wrapper"""
        self.instance = None
        super(DifferentWrapper, self).__init__(instance=config)

    @property
    def wraps(self):
        """Wrap ConceptTest"""
        return ConceptTest
