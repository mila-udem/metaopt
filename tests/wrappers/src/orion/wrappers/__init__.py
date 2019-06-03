# -*- coding: utf-8 -*-
"""Collection of wrappers and concepts for tests purposes"""

from orion.core.utils import (Concept, Wrapper)


class ConceptTest(Concept):
    """Basic Concept"""

    name = "Test"

    def __init__(self, **kwargs):
        """Initialize the concept"""
        super(ConceptTest, self).__init__(**kwargs)


class WrapperTest(Wrapper):
    """Basic wrapper"""

    def __init__(self, config):
        """Initialize the wrapper"""
        self.instance = None
        super(WrapperTest, self).__init__(instance=config)

    @property
    def wraps(self):
        """Wrap ConceptTest"""
        return ConceptTest


class ArgWrapperTest(Wrapper):
    """Basic wrapper but now with an argument"""

    def __init__(self, config):
        """Initiliaze the wrapper"""
        self.instance = None
        self.wrapper_arg = 0
        super(ArgWrapperTest, self).__init__(instance=config)

    @property
    def wraps(self):
        """Wrap ConceptTest"""
        return ConceptTest


class DepthWrapper(Wrapper):
    """Non-basic wrapper that wraps a wrapper"""

    def __init__(self, config):
        """Initialize the wrapper"""
        self.instance = None
        super(DepthWrapper, self).__init__(instance=config)

    @property
    def wraps(self):
        """Wrap Wrapper"""
        return Wrapper
