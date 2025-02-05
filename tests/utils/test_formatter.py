import pytest

from sqil_core.utils._formatter import *


class TestFormatNumber:
    def test_correctly_formats_number_in_scientific_notation(self):
        assert format_number(1, precision=3, unit="", latex=True) == "$1$"
        assert format_number(12, precision=3, unit="", latex=True) == "$12$"
        assert format_number(123, precision=3, unit="", latex=True) == "$123$"
        assert (
            format_number(1234, precision=3, unit="", latex=True) == "$1.23 x 10^{3}$"
        )
        assert (
            format_number(12345, precision=3, unit="", latex=True) == "$12.3 x 10^{3}$"
        )
        assert (
            format_number(123456, precision=3, unit="", latex=True) == "$123 x 10^{3}$"
        )
        assert (
            format_number(1234567, precision=3, unit="", latex=True)
            == "$1.23 x 10^{6}$"
        )
        assert format_number(0.1, precision=3, unit="", latex=True) == "$100 x 10^{-3}$"
        assert format_number(0.01, precision=3, unit="", latex=True) == "$10 x 10^{-3}$"
        assert format_number(0.001, precision=3, unit="", latex=True) == "$1 x 10^{-3}$"
        assert (
            format_number(0.0001, precision=3, unit="", latex=True) == "$100 x 10^{-6}$"
        )
        assert format_number(1e6, precision=3, unit="", latex=True) == "$1 x 10^{6}$"
        assert format_number(1e9, precision=3, unit="", latex=True) == "$1 x 10^{9}$"
        assert format_number(1e-6, precision=3, unit="", latex=True) == "$1 x 10^{-6}$"
        assert format_number(1e-9, precision=3, unit="", latex=True) == "$1 x 10^{-9}$"

    def test_precision_controls_the_number_of_digits(self):
        # Precision below 3 is not allowed
        assert (
            format_number(123456789, precision=0, unit="", latex=True)
            == "$123 x 10^{6}$"
        )
        assert (
            format_number(123456789, precision=1, unit="", latex=True)
            == "$123 x 10^{6}$"
        )
        assert (
            format_number(123456789, precision=2, unit="", latex=True)
            == "$123 x 10^{6}$"
        )
        assert (
            format_number(123456789, precision=3, unit="", latex=True)
            == "$123 x 10^{6}$"
        )
        # Larger precisions
        assert (
            format_number(123456789, precision=4, unit="", latex=True)
            == "$123.4 x 10^{6}$"
        )
        assert (
            format_number(123456789, precision=5, unit="", latex=True)
            == "$123.46 x 10^{6}$"
        )
        assert (
            format_number(123456789, precision=6, unit="", latex=True)
            == "$123.457 x 10^{6}$"
        )
        assert (
            format_number(123456789, precision=7, unit="", latex=True)
            == "$123.4568 x 10^{6}$"
        )
        assert (
            format_number(123456789, precision=8, unit="", latex=True)
            == "$123.45679 x 10^{6}$"
        )
        assert (
            format_number(123456789, precision=9, unit="", latex=True)
            == "$123.456789 x 10^{6}$"
        )

    def test_correctly_adds_units(self):
        assert format_number(1, unit="A", precision=3, latex=True) == "$1~A$"
        assert format_number(12, unit="A", precision=3, latex=True) == "$12~A$"
        assert format_number(123, unit="A", precision=3, latex=True) == "$123~A$"
        assert format_number(1234, unit="A", precision=3, latex=True) == "$1.23~kA$"
        assert format_number(12345, unit="A", precision=3, latex=True) == "$12.3~kA$"
        assert format_number(123456, unit="A", precision=3, latex=True) == "$123~kA$"
        assert format_number(1234567, unit="A", precision=3, latex=True) == "$1.23~MA$"
        assert format_number(0.1, unit="A", precision=3, latex=True) == "$100~mA$"
        assert format_number(0.01, unit="A", precision=3, latex=True) == "$10~mA$"
        assert format_number(0.001, unit="A", precision=3, latex=True) == "$1~mA$"
        assert format_number(0.0001, unit="A", precision=3, latex=True) == r"$100~\muA$"
        assert format_number(1e6, unit="A", precision=3, latex=True) == "$1~MA$"
        assert format_number(1e9, unit="A", precision=3, latex=True) == "$1~GA$"
        assert format_number(1e-6, unit="A", precision=3, latex=True) == r"$1~\muA$"
        assert format_number(1e-9, unit="A", precision=3, latex=True) == "$1~nA$"

    def test_adds_latex_syntax(self):
        assert format_number(1e-9, latex=True, unit="A", precision=3) == "$1~nA$"
        assert format_number(1e-9, latex=False, unit="A", precision=3) == "1 nA"
        assert format_number(1e-9, latex=True, unit="", precision=3) == "$1 x 10^{-9}$"
        assert format_number(1e-9, latex=False, unit="", precision=3) == "1 x 10^{-9}"
        assert format_number(1e9, latex=True, unit="", precision=3) == "$1 x 10^{9}$"
        assert format_number(1e9, latex=False, unit="", precision=3) == "1 x 10^{9}"

    def test_integers_remain_integers(self):
        assert format_number(1) == "$1$"
        assert format_number(2) == "$2$"
        assert format_number(1, unit="a") == "$1~a$"
        assert format_number(2, unit="a") == "$2~a$"
