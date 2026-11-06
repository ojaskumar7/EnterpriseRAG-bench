from __future__ import annotations


def test_app_package_imports() -> None:
    import app

    assert app is not None
