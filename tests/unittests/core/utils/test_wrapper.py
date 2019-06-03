#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.utils.Wrapper`."""
from copy import deepcopy

import pytest

from orion.wrappers import (ArgWrapperTest, DepthWrapper, WrapperTest)
from orion.wrappers.conceptimpl import ConceptImpl
from orion.wrappers.differentwrapper import DifferentWrapper
from orion.wrappers.subwrapper import SubWrapper


@pytest.fixture
def string_config():
    """Return a simple string as configuration"""
    return "conceptimpl"


@pytest.fixture
def dict_config_no_args():
    """Return a dictionary as configuration with no arguments"""
    return {"conceptimpl": {}}


@pytest.fixture
def dict_config_with_args():
    """Return a dictionary as configuration with an argument for the concept"""
    return {"conceptimpl": {"argument": 1}}


@pytest.fixture
def dict_config_with_wrapper_arg():
    """Return a dictionary with no arguments for the concept and one argument for the wrapper"""
    return {"conceptimpl": {}, "wrapper_arg": 1}


@pytest.fixture
def dict_config_no_impl():
    """Return a configuration with no concept to implement"""
    return {"wrapper_arg": 1}


@pytest.fixture
def dict_config_two_concepts():
    """Return a configuration with two concepts to implement"""
    return {"conceptimpl": {}, "otherconceptimpl": {}}


@pytest.fixture
def dict_depth_wrappers():
    """Return a configuration of two-depth wrappers"""
    return {"subwrapper": {"conceptimpl": {}}}


def test_string_instantiation(string_config):
    """Test if the instantiation through string works"""
    wrapper = WrapperTest(string_config)

    assert isinstance(wrapper, WrapperTest)
    assert isinstance(wrapper.instance, ConceptImpl)
    assert wrapper.instance.argument == 0


def test_no_args_dict_instantiation(dict_config_no_args):
    """Test if instantiation through dict works"""
    wrapper = WrapperTest(dict_config_no_args)

    assert isinstance(wrapper, WrapperTest)
    assert isinstance(wrapper.instance, ConceptImpl)
    assert wrapper.instance.argument == 0


def test_with_args_dict_instantiation(dict_config_with_args):
    """Test if argument setting during instantiation works"""
    wrapper = WrapperTest(dict_config_with_args)

    assert isinstance(wrapper, WrapperTest)
    assert isinstance(wrapper.instance, ConceptImpl)
    assert wrapper.instance.argument == 1


def test_dict_instantiation_with_wrapper_arg(dict_config_with_wrapper_arg):
    """Test if setting the argument of the wrapper works"""
    wrapper = ArgWrapperTest(dict_config_with_wrapper_arg)

    assert isinstance(wrapper, ArgWrapperTest)
    assert isinstance(wrapper.instance, ConceptImpl)
    assert wrapper.wrapper_arg == 1


def test_dict_instantion_with_no_impl(dict_config_no_impl):
    """Test if an error is raised when no concept has been instantiated from config"""
    with pytest.raises(NotImplementedError):
        ArgWrapperTest(dict_config_no_impl)


def test_dict_two_concepts_impl(dict_config_two_concepts):
    """Test if an error is raised when more than one concept is instantiated"""
    with pytest.raises(RuntimeError):
        WrapperTest(dict_config_two_concepts)


def test_dict_different_impl_module(dict_config_no_args):
    """Test if the instantiation of a concept in another module
    through `implementation_module` works
    """
    wrapper = DifferentWrapper(dict_config_no_args)

    assert isinstance(wrapper, DifferentWrapper)
    assert isinstance(wrapper.instance, ConceptImpl)


def test_dict_depth_wrappers(dict_depth_wrappers):
    """Test if two wrapers can be initialize in a cascading fashion"""
    wrapper = DepthWrapper(dict_depth_wrappers)

    assert isinstance(wrapper, DepthWrapper)
    assert isinstance(wrapper.instance, SubWrapper)
    assert isinstance(wrapper.instance.instance, ConceptImpl)


def test_deepcopy(string_config):
    """Test if the instantiation through string works"""
    wrapper = WrapperTest(string_config)

    deepcopy(wrapper)
