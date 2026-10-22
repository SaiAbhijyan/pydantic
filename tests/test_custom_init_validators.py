"""https://github.com/pydantic/pydantic/issues/13471

A custom ``__init__`` that calls ``super().__init__`` must run once, and so must
each ``model_validator`` (``before``, ``after``, and ``wrap``). That holds for the
constructor, ``model_validate``, nested models, and subclasses.

This is distinct from https://github.com/pydantic/pydantic/issues/12730, where a
wrap handler is invoked twice while ``self_instance`` is still set.
"""

from __future__ import annotations

from typing import Any

import pytest

from pydantic import BaseModel, ValidatorFunctionWrapHandler, model_validator


def _model_with_custom_init(mode: str) -> tuple[type[BaseModel], list[str]]:
    """Build a model whose ``__init__`` and one model validator record calls.

    ``passes`` ends at 1 when the validator runs once.
    """
    calls: list[str] = []

    if mode == 'before':

        class Model(BaseModel):
            x: int
            passes: int = 0

            def __init__(self, **data: Any) -> None:
                calls.append('init')
                super().__init__(**data)

            @model_validator(mode='before')
            @classmethod
            def _validate(cls, value: Any) -> Any:
                if isinstance(value, dict):
                    value['passes'] = int(value.get('passes', 0)) + 1
                return value

        return Model, calls

    if mode == 'after':

        class Model(BaseModel):
            x: int
            passes: int = 0

            def __init__(self, **data: Any) -> None:
                calls.append('init')
                super().__init__(**data)

            @model_validator(mode='after')
            def _validate(self) -> Model:
                self.passes += 1
                return self

        return Model, calls

    if mode == 'wrap':

        class Model(BaseModel):
            x: int
            passes: int = 0

            def __init__(self, **data: Any) -> None:
                calls.append('init')
                super().__init__(**data)

            @model_validator(mode='wrap')
            @classmethod
            def _validate(cls, value: Any, handler: ValidatorFunctionWrapHandler) -> Model:
                validated = handler(value)
                validated.passes += 1
                return validated

        return Model, calls

    raise AssertionError(mode)


def _assert_ran_once(model: BaseModel, calls: list[str]) -> None:
    # ``__init__`` is recorded once, then validators run inside ``super().__init__``.
    assert (model.x, model.passes, calls) == (1, 1, ['init'])


@pytest.mark.parametrize('mode', ['before', 'after', 'wrap'])
@pytest.mark.parametrize('entrypoint', ['constructor', 'model_validate', 'model_validate_json'])
def test_root_custom_init_runs_validators_once(mode: str, entrypoint: str) -> None:
    """``__init__`` and the model validator run once for every root entrypoint."""
    Model, calls = _model_with_custom_init(mode)

    if entrypoint == 'constructor':
        model = Model(x='1')
    elif entrypoint == 'model_validate':
        model = Model.model_validate({'x': '1'})
    else:
        model = Model.model_validate_json('{"x": "1"}')

    _assert_ran_once(model, calls)


@pytest.mark.parametrize('mode', ['before', 'after', 'wrap'])
@pytest.mark.parametrize('entrypoint', ['constructor', 'model_validate', 'model_validate_json'])
def test_nested_custom_init_runs_validators_once(mode: str, entrypoint: str) -> None:
    """A custom-init model used as a field runs its ``__init__`` and validators once."""
    Child, calls = _model_with_custom_init(mode)

    class Parent(BaseModel):
        child: Child

    if entrypoint == 'constructor':
        model = Parent(child={'x': '1'}).child
    elif entrypoint == 'model_validate':
        model = Parent.model_validate({'child': {'x': '1'}}).child
    else:
        model = Parent.model_validate_json('{"child": {"x": "1"}}').child

    _assert_ran_once(model, calls)


@pytest.mark.parametrize(
    'entrypoint',
    ['constructor', 'model_validate', 'nested_constructor', 'nested_model_validate'],
)
def test_issue_13471_example_runs_init_and_validators_once(entrypoint: str) -> None:
    """The nested wrap/after example from issue #13471."""
    calls: list[str] = []

    class MyBaseClass(BaseModel):
        def __init__(self, **kwargs: Any) -> None:
            calls.append(f'{type(self).__name__}.init')
            super().__init__(**kwargs)

    class A(MyBaseClass):
        id: int

        @model_validator(mode='after')
        def a_after_validator(self) -> A:
            calls.append('A.after')
            return self

    class B(MyBaseClass):
        item: A

        @model_validator(mode='wrap')
        @classmethod
        def fault_tolerate_validate(cls, value: Any, handler: ValidatorFunctionWrapHandler) -> B:
            calls.append('B.wrap')
            return handler(value)

        @model_validator(mode='after')
        def b_after_validator(self) -> B:
            calls.append('B.after')
            return self

    def _assert_once(model: B) -> None:
        assert model.item.id == 1
        assert calls == ['B.init', 'B.wrap', 'A.init', 'A.after', 'B.after']

    if entrypoint == 'constructor':
        model = B(item={'id': 1})
    elif entrypoint == 'model_validate':
        model = B.model_validate({'item': {'id': 1}})
    else:

        class Envelope(BaseModel):
            b: B

        if entrypoint == 'nested_constructor':
            model = Envelope(b={'item': {'id': 1}}).b
        else:
            model = Envelope.model_validate({'b': {'item': {'id': 1}}}).b

    _assert_once(model)


@pytest.mark.parametrize(
    'entrypoint',
    ['constructor', 'model_validate', 'nested_constructor', 'nested_model_validate'],
)
def test_subclass_of_custom_init_runs_validators_once(entrypoint: str) -> None:
    calls: list[str] = []

    class Base(BaseModel):
        def __init__(self, **data: Any) -> None:
            calls.append('Base.init')
            super().__init__(**data)

    class Child(Base):
        x: int
        passes: int = 0

        @model_validator(mode='before')
        @classmethod
        def _before(cls, value: Any) -> Any:
            calls.append('child.before')
            return value

        @model_validator(mode='after')
        def _after(self) -> Child:
            calls.append('child.after')
            self.passes += 1
            return self

    class GrandChild(Child):
        y: int = 2

        def __init__(self, **data: Any) -> None:
            calls.append('GrandChild.init')
            super().__init__(**data)

        @model_validator(mode='wrap')
        @classmethod
        def _wrap(cls, value: Any, handler: ValidatorFunctionWrapHandler) -> GrandChild:
            calls.append('grand.wrap')
            return handler(value)

    def _assert_once(model: GrandChild) -> None:
        assert (model.x, model.y, model.passes, calls) == (
            1,
            2,
            1,
            ['GrandChild.init', 'Base.init', 'grand.wrap', 'child.before', 'child.after'],
        )

    if entrypoint == 'constructor':
        model = GrandChild(x='1')
    elif entrypoint == 'model_validate':
        model = GrandChild.model_validate({'x': '1'})
    else:

        class Parent(BaseModel):
            item: GrandChild

        if entrypoint == 'nested_constructor':
            model = Parent(item={'x': '1'}).item
        else:
            model = Parent.model_validate({'item': {'x': '1'}}).item

    _assert_once(model)
