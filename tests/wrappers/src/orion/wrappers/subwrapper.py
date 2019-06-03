# -*- coding: utf-8 -*-
"""Implementation of a sub-wrapper for tests purposes"""

from orion.core.utils import Wrapper
from orion.wrappers import ConceptTest


class SubWrapper(Wrapper):
    """Provide a wrapper the be wrapped in a two-depth wrappers configuration"""

    implementation_module = "orion.wrappers"

    def __init__(self, **config):
        """Initialize the wrapper. These kind of wrappers required to take
        arbitrary keyword arguments
        """
        self.instance = None
        super(SubWrapper, self).__init__(instance=config)

    @property
    def wraps(self):
        """Wrap ConceptTest"""
        return ConceptTest
