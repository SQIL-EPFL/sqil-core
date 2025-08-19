import pytest

from sqil_core.experiment.helpers._function_override_handler import (
    FunctionOverrideHandler,
)


class ClassWithFunctions(FunctionOverrideHandler):
    public_id = 3
    _private_id = 2

    def __init__(self):
        super().__init__()
        self._default_functions = {"foo": self._default_foo, "bar": self._default_bar}
        self._functions = self._default_functions.copy()

    def _default_foo(self):
        return "default foo"

    def _default_bar(self):
        return "default bar"


def custom_foo(self):
    return "custom foo"


class TestFunctionOverrideHandler:
    @pytest.fixture
    def obj(self):
        return ClassWithFunctions()

    def test_should_permanently_override_function(self, obj):
        obj.override_function("foo", custom_foo)
        result1 = obj.call("foo")
        assert result1 == "custom foo"
        result2 = obj.call("foo")
        assert (
            result2 == "custom foo"
        ), "The 'foo' function should be overridden permanently."

    def test_should_override_and_allow_parameters(self, obj):
        def complex_foo(self, message, number=0):
            return f"{message}{number}"

        obj.override_function("foo", complex_foo)
        assert obj.call("foo", "hello", number=5) == "hello5"

    def test_overrides_should_have_access_to_instance_variables(self, obj):
        def complex_foo(self):
            return f"{self.public_id}{self._private_id}"

        obj.override_function("foo", complex_foo)
        assert obj.call("foo") == "32"

    def test_should_raise_error_if_function_to_override_does_not_exist(self, obj):
        with pytest.raises(
            AttributeError, match="Function 'non_existent' not found in the object."
        ):
            obj.override_function("non_existent", custom_foo)

    def test_should_restore_function_to_default(self, obj):
        obj.override_function("foo", custom_foo)
        result = obj.call("foo")
        assert result == "custom foo"

        obj.restore_function("foo")
        result = obj.call("foo")
        assert (
            result == "default foo"
        ), "The 'foo' function should be restored to its default implementation."

    def test_should_raise_error_if_function_to_restore_does_not_exist(self, obj):
        with pytest.raises(AttributeError):
            obj.restore_function("non_existent")

    def test_should_temporarily_override_function_within_context(self, obj):
        def temp_foo(self):
            return "temporary foo"

        with obj.temporary_override("foo", temp_foo):
            result = obj.call("foo")
            assert (
                result == "temporary foo"
            ), "The 'foo' function should be temporarily overridden."

        # After the context, it should restore to the default
        result = obj.call("foo")
        assert (
            result == "default foo"
        ), "The 'foo' function should revert back to its default after the context manager."

    def test_should_raise_error_if_function_to_temporary_override_does_not_exist(
        self, obj
    ):
        def temp_foo(self):
            return "temporary foo"

        with pytest.raises(AttributeError):
            with obj.temporary_override("non_existent", temp_foo):
                pass

    def test_should_restore_all_functions_to_defaults(self, obj):
        def custom_bar(self):
            return "custom bar"

        obj.override_function("foo", custom_foo)
        obj.override_function("bar", custom_bar)

        obj.restore_all_functions()

        # Test that all functions are restored
        result_foo = obj.call("foo")
        result_bar = obj.call("bar")

        assert (
            result_foo == "default foo"
        ), "The 'foo' function should be restored to its default implementation."
        assert (
            result_bar == "default bar"
        ), "The 'bar' function should be restored to its default implementation."

    def test_should_raise_error_if_calling_non_existent_function(self, obj):
        with pytest.raises(AttributeError):
            obj.call("non_existent")
