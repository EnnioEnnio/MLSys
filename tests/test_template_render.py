"""Per-row template rendering handles all-present, None, and missing columns."""

from __future__ import annotations

from src.mlsys.datasets import render_template

TEMPLATE = "description: {description}, variety: {variety}, country: {country}, price: {price} USD"


def test_all_columns_present() -> None:
    row = {
        "description": "ripe and round",
        "variety": "Pinot Noir",
        "country": "France",
        "price": 42,
    }
    out = render_template(TEMPLATE, row)
    assert out == "description: ripe and round, variety: Pinot Noir, country: France, price: 42 USD"


def test_none_columns_become_unknown() -> None:
    row = {
        "description": "ripe and round",
        "variety": None,
        "country": "France",
        "price": None,
    }
    out = render_template(TEMPLATE, row)
    assert "variety: unknown" in out
    assert "price: unknown USD" in out


def test_missing_columns_become_unknown() -> None:
    row = {"description": "ripe and round"}  # variety/country/price absent
    out = render_template(TEMPLATE, row)
    assert "variety: unknown" in out
    assert "country: unknown" in out
    assert "price: unknown USD" in out


def test_template_preserves_extra_row_keys() -> None:
    row = {"description": "x", "variety": "y", "country": "z", "price": 1, "winery": "w"}
    out = render_template(TEMPLATE, row)
    # extra keys aren't injected into the template
    assert "winery" not in out
