#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.evc.adapters`."""

import pytest

from orion.algo.space import Real
from orion.core.evc.adapters import (
    Adapter, AlgorithmChange, CodeChange, CompositeAdapter, DimensionAddition, DimensionDeletion,
    DimensionPriorChange, DimensionRenaming)
from orion.core.io.space_builder import DimensionBuilder
from orion.core.worker.trial import Trial


@pytest.fixture
def dummy_param():
    """Give dummy param integer param with value 1"""
    return Trial.Param(name='dummy', type='integer', value=1)


@pytest.fixture
def small_prior():
    """Give string format of small uniform distribution prior"""
    return "uniform(0, 10)"


@pytest.fixture
def large_prior():
    """Give string format of large uniform distribution prior"""
    return "uniform(0, 1000)"


@pytest.fixture
def normal_prior():
    """Give string format of normal distribution prior"""
    return "normal(0, 1)"


@pytest.fixture
def disjoint_prior():
    """Give string format of uniform distribution disjoint from small and large one"""
    return "uniform(-20, -10)"


@pytest.fixture
def integer_prior():
    """Give string format of uniform distribution casted to integers"""
    return "uniform(-20, -10, discrete=True)"


@pytest.fixture
def categorical_prior():
    """Give string format of categorical prior with floats, integers and strings"""
    return "choices([0.1, 1, 2, 'string'])"


@pytest.fixture
def multidim_prior():
    """Give string format of real distribution with multiple dimensions"""
    return "uniform(0, 10, shape=(2, 2))"


@pytest.fixture
def trials(small_prior, large_prior, normal_prior, disjoint_prior,
           integer_prior, categorical_prior, multidim_prior):
    """Trials with dimensions for all priors defined as fixtures"""
    N_TRIALS = 10

    priors = dict((name, prior) for (name, prior) in locals().items()
                  if isinstance(name, str) and name.endswith("_prior"))

    trials = []
    for _ in range(N_TRIALS):
        params = []
        for name, prior in priors.items():
            dimension = DimensionBuilder().build(name, prior)
            value = dimension.sample()[0]
            params.append(Trial.Param(name=name, type=dimension.type, value=value).to_dict())
        trials.append(Trial(params=params))

    return trials


class TestDimensionAdditionInit(object):
    """Test initialization of :class:`orion.core.evc.adapters.DimensionAddition`"""

    def test_dimension_addition_init_with_param(self, dummy_param):
        """Test initialization of :class:`orion.core.evc.adapters.DimensionAddition`
        with valid param
        """
        dimension_addition_adapter = DimensionAddition(dummy_param)

        assert dimension_addition_adapter.param is dummy_param

    def test_dimension_addition_init_with_bad_param(self):
        """Test initialization of :class:`orion.core.evc.adapters.DimensionAddition`
        with object which is not a param or a dictionary definition of it
        """
        with pytest.raises(TypeError) as exc_info:
            DimensionAddition('bad')

        assert "Invalid param argument type ('<class 'str'>')." in str(exc_info.value)

    def test_dimension_addition_init_with_dict(self, dummy_param):
        """Test initialization of :class:`orion.core.evc.adapters.DimensionAddition`
        with a dictionary definition of a param
        """
        DimensionAddition(dummy_param.to_dict())

    def test_dimension_addition_init_with_bad_dict(self):
        """Test initialization of :class:`orion.core.evc.adapters.DimensionAddition`
        with a dictionary which is a bad definition of a param
        """
        with pytest.raises(AttributeError) as exc_info:
            DimensionAddition({'bad': 'dict'})

        assert "'Param' object has no attribute 'bad'" in str(exc_info.value)


class TestDimensionDeletionInit(object):
    """Test initialization of :class:`orion.core.evc.adapters.DimensionDeletion`"""

    def test_dimension_deletion_init_with_param(self, dummy_param):
        """Test initialization of :class:`orion.core.evc.adapters.DimensionDeletion`
        with valid param
        """
        dimension_deletion_adapter = DimensionDeletion(dummy_param)

        assert dimension_deletion_adapter.param is dummy_param

    def test_dimension_deletion_init_with_bad_param(self):
        """Test initialization of :class:`orion.core.evc.adapters.DimensionDeletion`
        with object which is not a param or a dictionary definition of it
        """
        with pytest.raises(TypeError) as exc_info:
            DimensionDeletion('bad')

        assert "Invalid param argument type ('<class 'str'>')." in str(exc_info.value)

    def test_dimension_deletion_init_with_dict(self, dummy_param):
        """Test initialization of :class:`orion.core.evc.adapters.DimensionDeletion`
        with a dictionary definition of a param
        """
        DimensionDeletion(dummy_param.to_dict())

    def test_dimension_deletion_init_with_bad_dict(self):
        """Test initialization of :class:`orion.core.evc.adapters.DimensionDeletion`
        with a dictionary which is a bad definition of a param
        """
        with pytest.raises(AttributeError) as exc_info:
            print(DimensionDeletion({'bad': 'dict'}).param)

        assert "'Param' object has no attribute 'bad'" in str(exc_info.value)


class TestDimensionPriorChangeInit(object):
    """Test initialization of :class:`orion.core.evc.adapters.DimensionPriorChange`"""

    def test_dimension_prior_change_init_with_dimensions(self, large_prior, small_prior):
        """Test initialization of :class:`orion.core.evc.adapters.DimensionPriorChange`
        with valid string definitions of dimension prior
        """
        dimension_prior_change = DimensionPriorChange('dummy', large_prior, small_prior)

        assert dimension_prior_change.old_prior == large_prior
        assert dimension_prior_change.new_prior == small_prior
        assert isinstance(dimension_prior_change.old_dimension, Real)
        assert isinstance(dimension_prior_change.new_dimension, Real)

        assert dimension_prior_change.old_dimension.interval() == (0, 1000)
        assert dimension_prior_change.new_dimension.interval() == (0, 10)

    def test_dimension_prior_change_init_with_bad_dimensions(self):
        """Test initialization of :class:`orion.core.evc.adapters.DimensionPriorChange`
        with non valid string definitions of dimension prior
        """
        with pytest.raises(TypeError) as exc_info:
            DimensionPriorChange('dummy', 'bad', 'priors')

        assert "Parameter 'old': Please provide a valid form for prior" in str(exc_info.value)


class TestDimensionRenamingInit(object):
    """Test initialization of :class:`orion.core.evc.adapters.DimensionRenaming`"""

    def test_dimension_renaming_init(self):
        """Test initialization of :class:`orion.core.evc.adapters.DimensionRenaming`
        with valid names
        """
        dimension_renaming = DimensionRenaming('old_name', 'new_name')
        assert dimension_renaming.old_name == 'old_name'
        assert dimension_renaming.new_name == 'new_name'

    def test_dimension_renaming_init_bad_name(self):
        """Test initialization of :class:`orion.core.evc.adapters.DimensionRenaming`
        with invalid names which are not strings
        """
        with pytest.raises(TypeError) as exc_info:
            DimensionRenaming({'bad': 'name'}, None)

        assert "" in str(exc_info.value)


class TestAlgorithmChangeInit(object):
    """Test initialization of :class:`orion.core.evc.adapters.AlgorithmChange`"""

    def test_algorithm_change_init(self):
        """Test initialization of :class:`orion.core.evc.adapters.AlgorithmChange`"""
        AlgorithmChange()


class TestCodeChangeInit(object):
    """Test initialization of :class:`orion.core.evc.adapters.CodeChange`"""

    def test_code_change_init(self):
        """Test initialization of :class:`orion.core.evc.adapters.CodeChange`
        with valid change types
        """
        code_change_adapter = CodeChange(CodeChange.NOEFFECT)
        assert code_change_adapter.change_type == CodeChange.NOEFFECT

        code_change_adapter = CodeChange(CodeChange.BREAK)
        assert code_change_adapter.change_type == CodeChange.BREAK

    def test_code_change_init_bad_type(self):
        """Test initialization of :class:`orion.core.evc.adapters.CodeChange`
        with invalid change types
        """
        with pytest.raises(ValueError) as exc_info:
            CodeChange('bad type')

        assert "Invalid code change type 'bad type'" in str(exc_info.value)


class TestCompositeAdapterInit(object):
    """Test initialization of :class:`orion.core.evc.adapters.CompositeAdapter`"""

    def test_composite_adapter_init_emtpy(self):
        """Test initialization of :class:`orion.core.evc.adapters.CompositeAdapter`
        with no adapters
        """
        composite_adapter = CompositeAdapter()
        assert len(composite_adapter.adapters) == 0

    def test_composite_adapter_init_with_adapters(self, dummy_param):
        """Test initialization of :class:`orion.core.evc.adapters.CompositeAdapter`
        with valid adapters
        """
        dimension_addition = DimensionAddition(dummy_param)
        dimension_deletion = DimensionDeletion(dummy_param)
        composite_adapter = CompositeAdapter(dimension_addition, dimension_deletion)
        assert len(composite_adapter.adapters) == 2
        assert isinstance(composite_adapter.adapters[0], DimensionAddition)
        assert isinstance(composite_adapter.adapters[1], DimensionDeletion)

    def test_composite_adapter_init_with_bad_adapters(self):
        """Test initialization of :class:`orion.core.evc.adapters.CompositeAdapter`
        with invalid adapters
        """
        with pytest.raises(TypeError) as exc_info:
            CompositeAdapter('bad', 'adapters')

        assert ("Provided adapters must be adapter objects, not '<class 'str'>" in
                str(exc_info.value))


def test_adapter_creation(dummy_param):
    """Test initialization using :meth:`orion.core.evc.adapters.Adapter.build`"""
    adapter = Adapter.build([{
        'of_type': 'DimensionAddition',
        'param': dummy_param.to_dict()}])

    assert isinstance(adapter, CompositeAdapter)
    assert len(adapter.adapters) == 1
    assert isinstance(adapter.adapters[0], DimensionAddition)
    assert adapter.adapters[0].param.to_dict() == dummy_param.to_dict()


class TestDimensionAdditionForwardBackward(object):
    """Test :meth:`orion.core.evc.adapters.DimensionAddition.forward` and
    :meth:`orion.core.evc.adapters.DimensionAddition.backward`
    """

    def test_dimension_addition_forward(self, trials):
        """Test :meth:`orion.core.evc.adapters.DimensionAddition.forward`
        with valid param and trials
        """
        new_param = Trial.Param(name='second_normal_prior', type='integer', value=1)
        dimension_addition_adapter = DimensionAddition(new_param)

        adapted_trials = dimension_addition_adapter.forward(trials)

        assert adapted_trials[0].params[-1] == new_param
        assert adapted_trials[4].params[-1] == new_param
        assert adapted_trials[-1].params[-1] == new_param

    def test_dimension_addition_forward_already_existing(self, trials):
        """Test :meth:`orion.core.evc.adapters.DimensionAddition.forward`
        with valid param and incompatible trials because param already exists
        """
        new_param = Trial.Param(name='normal_prior', type='integer', value=1)
        dimension_addition_adapter = DimensionAddition(new_param)

        with pytest.raises(RuntimeError) as exc_info:
            dimension_addition_adapter.forward(trials)
        assert "Provided trial does not have a compatible configuration" in str(exc_info.value)

    def test_dimension_addition_backward(self, dummy_param, trials):
        """Test :meth:`orion.core.evc.adapters.DimensionAddition.backward`
        with valid param and valid trials
        """
        new_param = Trial.Param(name='second_normal_prior', type='integer', value=1)
        dimension_addition_adapter = DimensionAddition(new_param)

        sampler = DimensionBuilder().build('random', 'uniform(10, 100, discrete=True)')
        for trial in trials:
            random_param = new_param.to_dict()
            random_param['value'] = sampler.sample()
            trial.params.append(Trial.Param(**random_param))

        adapted_trials = dimension_addition_adapter.backward(trials)
        assert len(adapted_trials) == 0

        trials[0].params[-1].value = 1
        assert trials[0].params[-1] == new_param

        adapted_trials = dimension_addition_adapter.backward(trials)
        assert len(adapted_trials) == 1

        trials[4].params[-1].value = 1
        assert trials[4].params[-1] == new_param

        adapted_trials = dimension_addition_adapter.backward(trials)
        assert len(adapted_trials) == 2

        trials[-1].params[-1].value = 1
        assert trials[-1].params[-1] == new_param

        adapted_trials = dimension_addition_adapter.backward(trials)
        assert len(adapted_trials) == 3

        assert new_param not in (adapted_trials[0].params)
        assert new_param not in (adapted_trials[1].params)
        assert new_param not in (adapted_trials[2].params)

    def test_dimension_addition_backward_not_existing(self, trials):
        """Test :meth:`orion.core.evc.adapters.DimensionAddition.backward`
        with valid param and invalid trials because param does not exist
        """
        new_param = Trial.Param(name='second_normal_prior', type='integer', value=1)
        dimension_addition_adapter = DimensionAddition(new_param)

        with pytest.raises(RuntimeError) as exc_info:
            dimension_addition_adapter.backward(trials)
        assert "Provided trial does not have a compatible configuration" in str(exc_info.value)


class TestDimensionDeletionForwardBackward(object):
    """Test :meth:`orion.core.evc.adapters.DimensionDeletion.forward` and
    :meth:`orion.core.evc.adapters.DimensionDeletion.backward`
    """

    def test_dimension_deletion_forward(self, trials):
        """Test :meth:`orion.core.evc.adapters.DimensionDeletion.forward`
        with valid param and valid trials
        """
        new_param = Trial.Param(name='second_normal_prior', type='integer', value=1)
        dimension_deletion_adapter = DimensionDeletion(new_param)

        sampler = DimensionBuilder().build('random', 'uniform(10, 100, discrete=True)')
        for trial in trials:
            random_param = new_param.to_dict()
            random_param['value'] = sampler.sample()
            trial.params.append(Trial.Param(**random_param))

        adapted_trials = dimension_deletion_adapter.forward(trials)
        assert len(adapted_trials) == 0

        trials[0].params[-1].value = 1
        assert trials[0].params[-1] == new_param

        adapted_trials = dimension_deletion_adapter.forward(trials)
        assert len(adapted_trials) == 1

        trials[4].params[-1].value = 1
        assert trials[4].params[-1] == new_param

        adapted_trials = dimension_deletion_adapter.forward(trials)
        assert len(adapted_trials) == 2

        trials[-1].params[-1].value = 1
        assert trials[-1].params[-1] == new_param

        adapted_trials = dimension_deletion_adapter.forward(trials)
        assert len(adapted_trials) == 3

        assert new_param not in (adapted_trials[0].params)
        assert new_param not in (adapted_trials[1].params)
        assert new_param not in (adapted_trials[2].params)

    def test_dimension_deletion_forward_not_existing(self, trials):
        """Test :meth:`orion.core.evc.adapters.DimensionDeletion.forward`
        with valid param and invalid trials because param does not exist
        """
        new_param = Trial.Param(name='second_normal_prior', type='integer', value=1)
        dimension_deletion_adapter = DimensionDeletion(new_param)

        with pytest.raises(RuntimeError) as exc_info:
            dimension_deletion_adapter.forward(trials)
        assert "Provided trial does not have a compatible configuration" in str(exc_info.value)

    def test_dimension_deletion_backward(self, dummy_param, trials):
        """Test :meth:`orion.core.evc.adapters.DimensionDeletion.backward`
        with valid param and valid trials
        """
        new_param = Trial.Param(name='second_normal_prior', type='integer', value=1)
        dimension_deletion_adapter = DimensionDeletion(new_param)

        adapted_trials = dimension_deletion_adapter.backward(trials)

        assert adapted_trials[0].params[-1] == new_param
        assert adapted_trials[4].params[-1] == new_param
        assert adapted_trials[-1].params[-1] == new_param

    def test_dimension_deletion_backward_already_existing(self, trials):
        """Test :meth:`orion.core.evc.adapters.DimensionDeletion.backward`
        with valid param and invalid trials because param already exist
        """
        new_param = Trial.Param(name='normal_prior', type='integer', value=1)
        dimension_deletion_adapter = DimensionDeletion(new_param)

        with pytest.raises(RuntimeError) as exc_info:
            dimension_deletion_adapter.backward(trials)
        assert "Provided trial does not have a compatible configuration" in str(exc_info.value)


class TestDimensionPriorChangeForwardBackward(object):
    """Test :meth:`orion.core.evc.adapters.DimensionPriorChange.forward` and
    :meth:`orion.core.evc.adapters.DimensionPriorChange.backward`
    """

    def test_dimension_prior_change_forward(self, large_prior, small_prior, trials):
        """Test :meth:`orion.core.evc.adapters.DimensionPriorChange.forward`
        with compatible priors
        """
        dimension_prior_change_adapter = DimensionPriorChange(
            'small_prior', small_prior, large_prior)

        adapted_trials = dimension_prior_change_adapter.forward(trials)

        assert len(adapted_trials) == len(trials)

    def test_dimension_prior_change_forward_incompatible_dimensions(
            self, small_prior, disjoint_prior, trials):
        """Test :meth:`orion.core.evc.adapters.DimensionPriorChange.forward`
        with incompatible priors, such that all trials are filtered out
        """
        dimension_prior_change_adapter = DimensionPriorChange(
            'small_prior', small_prior, disjoint_prior)

        adapted_trials = dimension_prior_change_adapter.forward(trials)

        assert len(adapted_trials) == 0

    def test_dimension_prior_change_backward(self, large_prior, small_prior, trials):
        """Test :meth:`orion.core.evc.adapters.DimensionPriorChange.backward`
        with compatible priors
        """
        dimension_prior_change_adapter = DimensionPriorChange(
            'large_prior', large_prior, small_prior)

        adapted_trials = dimension_prior_change_adapter.backward(trials)

        assert len(adapted_trials) == len(trials)

    def test_dimension_prior_change_backward_incompatible_dimensions(
            self, disjoint_prior, small_prior, trials):
        """Test :meth:`orion.core.evc.adapters.DimensionPriorChange.backward`
        with incompatible priors, such that all trials are filtered out
        """
        dimension_prior_change_adapter = DimensionPriorChange(
            'small_prior', disjoint_prior, small_prior)

        adapted_trials = dimension_prior_change_adapter.backward(trials)

        assert len(adapted_trials) == 0


class TestDimensionRenamingForwardBackward(object):
    """Test :meth:`orion.core.evc.adapters.DimensionRenaming.forward` and
    :meth:`orion.core.evc.adapters.DimensionRenaming.backward`
    """

    def test_dimension_renaming_forward(self, trials):
        """Test :meth:`orion.core.evc.adapters.DimensionRenaming.forward`
        with valid names
        """
        old_name = 'small_prior'
        new_name = 'new_name'

        dimension_renaming_adapter = DimensionRenaming(old_name, new_name)

        adapted_trials = dimension_renaming_adapter.forward(trials)

        assert len(adapted_trials) == len(trials)

        assert new_name in [param.name for param in adapted_trials[0].params]
        assert old_name not in [param.name for param in adapted_trials[0].params]

        assert new_name in [param.name for param in adapted_trials[-1].params]
        assert old_name not in [param.name for param in adapted_trials[-1].params]

    def test_dimension_renaming_forward_incompatible(self, trials):
        """Test :meth:`orion.core.evc.adapters.DimensionRenaming.forward`
        with non existing old name in trials
        """
        old_name = 'bad name'
        new_name = 'small_prior'

        dimension_renaming_adapter = DimensionRenaming(old_name, new_name)

        with pytest.raises(RuntimeError) as exc_info:
            dimension_renaming_adapter.forward(trials)
        assert "Provided trial does not have a compatible configuration" in str(exc_info.value)

    def test_dimension_renaming_backward(self, trials):
        """Test :meth:`orion.core.evc.adapters.DimensionRenaming.backward`
        with valid names
        """
        old_name = 'old_name'
        new_name = 'small_prior'

        dimension_renaming_adapter = DimensionRenaming(old_name, new_name)

        adapted_trials = dimension_renaming_adapter.backward(trials)

        assert len(adapted_trials) == len(trials)

        assert old_name in [param.name for param in adapted_trials[0].params]
        assert new_name not in [param.name for param in adapted_trials[0].params]

        assert old_name in [param.name for param in adapted_trials[-1].params]
        assert new_name not in [param.name for param in adapted_trials[-1].params]

    def test_dimension_renaming_backward_incompatible(self, trials):
        """Test :meth:`orion.core.evc.adapters.DimensionRenaming.backward`
        with non existing new name in trials
        """
        old_name = 'small_prior'
        new_name = 'bad name'

        dimension_renaming_adapter = DimensionRenaming(old_name, new_name)

        with pytest.raises(RuntimeError) as exc_info:
            dimension_renaming_adapter.backward(trials)
        assert "Provided trial does not have a compatible configuration" in str(exc_info.value)


class TestAlgorithmChangeForwardBackward(object):
    """Test :meth:`orion.core.evc.adapters.AlgorithmChange.forward` and
    :meth:`orion.core.evc.adapters.AlgorithmChange.backward`
    """

    def test_algorithm_change_forward(self, trials):
        """Test :meth:`orion.core.evc.adapters.AlgorithmChange.forward`"""
        algorithm_change_adaptor = AlgorithmChange()

        adapted_trials = algorithm_change_adaptor.forward(trials)

        assert len(adapted_trials) == len(trials)

        assert adapted_trials[0] is trials[0]
        assert adapted_trials[-1] is trials[-1]

    def test_algorithm_change_backward(self, trials):
        """Test :meth:`orion.core.evc.adapters.AlgorithmChange.backward`"""
        algorithm_change_adaptor = AlgorithmChange()

        adapted_trials = algorithm_change_adaptor.backward(trials)

        assert len(adapted_trials) == len(trials)

        assert adapted_trials[0] is trials[0]
        assert adapted_trials[-1] is trials[-1]


class TestCodeChangeForwardBackward(object):
    """Test :meth:`orion.core.evc.adapters.CodeChange.forward` and
    :meth:`orion.core.evc.adapters.CodeChange.backward`
    """

    def test_code_change_forward_noeffect(self, trials):
        """Test :meth:`orion.core.evc.adapters.CodeChange.forward` with change type NOEFFECT"""
        code_change_adapter = CodeChange(CodeChange.NOEFFECT)

        adapted_trials = code_change_adapter.forward(trials)

        assert len(adapted_trials) == len(trials)

        assert adapted_trials[0] is trials[0]
        assert adapted_trials[-1] is trials[-1]

    def test_code_change_forward_unsure(self, trials):
        """Test :meth:`orion.core.evc.adapters.CodeChange.forward` with change type UNSURE"""
        code_change_adapter = CodeChange(CodeChange.UNSURE)

        adapted_trials = code_change_adapter.forward(trials)

        assert len(adapted_trials) == len(trials)

        assert adapted_trials[0] is trials[0]
        assert adapted_trials[-1] is trials[-1]

    def test_code_change_forward_break(self, trials):
        """Test :meth:`orion.core.evc.adapters.CodeChange.forward` with change type BREAK"""
        code_change_adapter = CodeChange(CodeChange.BREAK)

        adapted_trials = code_change_adapter.forward(trials)

        assert len(adapted_trials) == 0

    def test_code_change_backward_noeffect(self, trials):
        """Test :meth:`orion.core.evc.adapters.CodeChange.backward` with change type NOEFFECT"""
        code_change_adapter = CodeChange(CodeChange.NOEFFECT)

        adapted_trials = code_change_adapter.backward(trials)

        assert len(adapted_trials) == len(trials)

        assert adapted_trials[0] is trials[0]
        assert adapted_trials[-1] is trials[-1]

    def test_code_change_backward_unsure(self, trials):
        """Test :meth:`orion.core.evc.adapters.CodeChange.backward` with change type UNSURE"""
        code_change_adapter = CodeChange(CodeChange.UNSURE)

        adapted_trials = code_change_adapter.backward(trials)

        assert len(adapted_trials) == 0

    def test_code_change_backward_break(self, trials):
        """Test :meth:`orion.core.evc.adapters.CodeChange.backward` with change type BREAK"""
        code_change_adapter = CodeChange(CodeChange.BREAK)

        adapted_trials = code_change_adapter.backward(trials)

        assert len(adapted_trials) == 0


class TestCompositeAdapterForwardBackward(object):
    """Test :meth:`orion.core.evc.adapters.CompositeAdapter.forward` and
    :meth:`orion.core.evc.adapters.CompositeAdapter.backward`
    """

    def test_composite_adapter_forward_emtpy(self, trials):
        """Test :meth:`orion.core.evc.adapters.CompositeAdapter.forward` and
        :meth:`orion.core.evc.adapters.CompositeAdapter.backward` with no adapters
        """
        composite_adapter = CompositeAdapter()

        assert len(composite_adapter.forward(trials)) == len(trials)
        assert len(composite_adapter.backward(trials)) == len(trials)

    def test_composite_adapter_forward(self, dummy_param, trials):
        """Test :meth:`orion.core.evc.adapters.CompositeAdapter.forward` with two adapters"""
        new_param = Trial.Param(name='second_normal_prior', type='integer', value=1)

        dimension_addition_adapter = DimensionAddition(new_param)
        dimension_deletion_adapter = DimensionDeletion(new_param)

        composite_adapter = CompositeAdapter(dimension_addition_adapter)

        adapted_trials = composite_adapter.forward(trials)

        assert adapted_trials[0].params[-1] == new_param
        assert adapted_trials[4].params[-1] == new_param
        assert adapted_trials[-1].params[-1] == new_param

        composite_adapter = CompositeAdapter(dimension_addition_adapter, dimension_deletion_adapter)

        adapted_trials = composite_adapter.forward(trials)

        assert len(adapted_trials) == len(trials)

        assert new_param not in (adapted_trials[0].params)
        assert new_param not in (adapted_trials[4].params)
        assert new_param not in (adapted_trials[-1].params)

    def test_composite_adapter_backward(self, dummy_param, trials):
        """Test :meth:`orion.core.evc.adapters.CompositeAdapter.backward` with two adapters"""
        new_param = Trial.Param(name='second_normal_prior', type='integer', value=1)

        dimension_addition_adapter = DimensionAddition(new_param)
        dimension_deletion_adapter = DimensionDeletion(new_param)

        composite_adapter = CompositeAdapter(dimension_deletion_adapter)

        adapted_trials = composite_adapter.backward(trials)

        assert adapted_trials[0].params[-1] == new_param
        assert adapted_trials[4].params[-1] == new_param
        assert adapted_trials[-1].params[-1] == new_param

        composite_adapter = CompositeAdapter(dimension_addition_adapter, dimension_deletion_adapter)

        adapted_trials = composite_adapter.backward(trials)

        assert len(adapted_trials) == len(trials)

        assert new_param not in (adapted_trials[0].params)
        assert new_param not in (adapted_trials[4].params)
        assert new_param not in (adapted_trials[-1].params)


def test_created_adapter_has_string_of_type(dummy_param):
    """Test if the configuration dict of the newly created adapter is not a tuple"""
    dimension_addition_adapter = DimensionAddition(dummy_param)

    configuration = dimension_addition_adapter.configuration[0]

    new_adapter = Adapter.build([configuration]).adapters[0]

    assert type(new_adapter.configuration[0]['of_type']) is not tuple


def test_dimension_addition_configuration(dummy_param):
    """Test :meth:`orion.core.evc.adapters.DimensionAddition.configuration`"""
    dimension_addition_adapter = DimensionAddition(dummy_param)

    configuration = dimension_addition_adapter.configuration[0]

    assert configuration['of_type'] == "dimensionaddition"
    assert configuration['param'] == dummy_param.to_dict()

    assert Adapter.build([configuration]).adapters[0].configuration[0] == configuration


def test_dimension_deletion_configuration(dummy_param):
    """Test :meth:`orion.core.evc.adapters.DimensionDeletion.configuration`"""
    dimension_deletion_adapter = DimensionDeletion(dummy_param)

    configuration = dimension_deletion_adapter.configuration[0]

    assert configuration['of_type'] == "dimensiondeletion"
    assert configuration['param'] == dummy_param.to_dict()

    assert Adapter.build([configuration]).adapters[0].configuration[0] == configuration


def test_dimension_prior_change_configuration(small_prior, large_prior):
    """Test :meth:`orion.core.evc.adapters.DimensionPriorChange.configuration`"""
    dimension_name = 'small_prior'
    dimension_prior_change_adapter = DimensionPriorChange(dimension_name, small_prior, large_prior)

    configuration = dimension_prior_change_adapter.configuration[0]

    assert configuration['of_type'] == "dimensionpriorchange"
    assert configuration['name'] == dimension_name
    assert configuration['old_prior'] == small_prior
    assert configuration['new_prior'] == large_prior

    assert Adapter.build([configuration]).adapters[0].configuration[0] == configuration


def test_dimension_renaming_configuration():
    """Test :meth:`orion.core.evc.adapters.DimensionRenaming.configuration`"""
    old_name = 'old_name'
    new_name = 'new_name'
    dimension_renaming_adapter = DimensionRenaming(old_name, new_name)

    configuration = dimension_renaming_adapter.configuration[0]

    assert configuration['of_type'] == "dimensionrenaming"
    assert configuration['old_name'] == old_name
    assert configuration['new_name'] == new_name

    assert Adapter.build([configuration]).adapters[0].configuration[0] == configuration


def test_algorithm_change_configuration():
    """Test :meth:`orion.core.evc.adapters.AlgorithmChange.configuration`"""
    algorithm_change_adaptor = AlgorithmChange()

    configuration = algorithm_change_adaptor.configuration[0]

    assert configuration['of_type'] == "algorithmchange"

    assert Adapter.build([configuration]).adapters[0].configuration[0] == configuration


def test_code_change_configuration():
    """Test :meth:`orion.core.evc.adapters.CodeChange.configuration`"""
    code_change_adaptor = CodeChange(CodeChange.UNSURE)

    configuration = code_change_adaptor.configuration[0]

    assert configuration['of_type'] == "codechange"
    assert configuration['change_type'] == CodeChange.UNSURE

    assert Adapter.build([configuration]).adapters[0].configuration[0] == configuration


def test_composite_configuration(dummy_param):
    """Test :meth:`orion.core.evc.adapters.CompositeAdapter.configuration`"""
    new_param = Trial.Param(name='second_normal_prior', type='integer', value=1)
    dimension_addition_adapter = DimensionAddition(dummy_param)
    dimension_deletion_adapter = DimensionDeletion(new_param)

    composite_adapter = CompositeAdapter(dimension_addition_adapter, dimension_deletion_adapter)

    configuration = composite_adapter.configuration

    assert configuration[0] == dimension_addition_adapter.configuration[0]
    assert configuration[1] == dimension_deletion_adapter.configuration[0]

    assert Adapter.build(configuration).adapters[0].configuration[0] == configuration[0]
    assert Adapter.build(configuration).adapters[1].configuration[0] == configuration[1]
