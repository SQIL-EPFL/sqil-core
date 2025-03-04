import pytest

from sqil_core.utils._utils import *


class TestFillGaps:

    def test_fill_gaps_basic(self):
        primary_list = [1, None, 3, None, 5]
        fallback_list = [10, 20, 30, 40, 50]
        expected_result = [1, 20, 3, 40, 5]
        assert fill_gaps(primary_list, fallback_list) == expected_result

    def test_no_gaps_in_primary(self):
        primary_list = [1, 2, 3, 4, 5]
        fallback_list = [10, 20, 30, 40, 50]
        expected_result = [1, 2, 3, 4, 5]
        assert fill_gaps(primary_list, fallback_list) == expected_result

    def test_primary_is_none(self):
        primary_list = [None, None, None]
        fallback_list = [10, 20, 30]
        expected_result = [10, 20, 30]
        assert fill_gaps(primary_list, fallback_list) == expected_result

    def test_fallback_is_empty(self):
        primary_list = [1, None, 3]
        fallback_list = []
        expected_result = [1, None, 3]  # Primary list should remain unchanged
        assert fill_gaps(primary_list, fallback_list) == expected_result

    def test_primary_is_empty(self):
        primary_list = []
        fallback_list = [10, 20, 30]
        expected_result = []  # Empty primary list should return empty result
        assert fill_gaps(primary_list, fallback_list) == expected_result

    def test_both_empty_lists(self):
        primary_list = []
        fallback_list = []
        expected_result = []  # Both lists are empty, return empty list
        assert fill_gaps(primary_list, fallback_list) == expected_result

    def test_primary_larger_than_fallback(self):
        primary_list = [1, None, 3, None, 5]
        fallback_list = [10, 20, 30]
        expected_result = [
            1,
            20,
            3,
            None,
            5,
        ]
        assert fill_gaps(primary_list, fallback_list) == expected_result

    def test_fallback_larger_than_primary(self):
        primary_list = [1, None, 3]
        fallback_list = [10, 20, 30, 40, 50]
        expected_result = [
            1,
            20,
            3,
        ]  # Only the first 3 elements from fallback should be used
        assert fill_gaps(primary_list, fallback_list) == expected_result

    def test_all_none_in_primary(self):
        primary_list = [None, None, None]
        fallback_list = [1, 2, 3]
        expected_result = [
            1,
            2,
            3,
        ]  # Should replace all None values with fallback values
        assert fill_gaps(primary_list, fallback_list) == expected_result


class TestMakeIterable:
    def test_single_integer(self):
        assert make_iterable(42) == [42]

    def test_list(self):
        assert make_iterable([1, 2, 3]) == [1, 2, 3]

    def test_tuple(self):
        assert make_iterable((4, 5, 6)) == (4, 5, 6)

    def test_set(self):
        assert make_iterable({7, 8, 9}) == {7, 8, 9}

    def test_dict(self):
        assert make_iterable({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_string(self):
        assert make_iterable("hello") == ["hello"]

    def test_none(self):
        assert make_iterable(None) == [None]

    def test_iterable_custom_object(self):
        class CustomIterable:
            def __iter__(self):
                return iter([10, 20, 30])

        obj = CustomIterable()
        assert make_iterable(obj) == obj

    def test_non_iterable_object(self):
        class NonIterable:
            pass

        obj = NonIterable()
        assert make_iterable(obj) == [obj]
