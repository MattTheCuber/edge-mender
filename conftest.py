"""Configuration for pytest."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command line options for pytest.

    References
    ----------
    .. [1] https://docs.pytest.org/en/stable/example/simple.html#control-skipping-of-tests-according-to-command-line-option
    """
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Run long-running tests",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers.

    References
    ----------
    .. [1] https://docs.pytest.org/en/stable/example/simple.html#control-skipping-of-tests-according-to-command-line-option
    """
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Modify collected test items to skip slow tests unless --slow is given.

    References
    ----------
    .. [1] https://docs.pytest.org/en/stable/example/simple.html#control-skipping-of-tests-according-to-command-line-option
    """
    if config.getoption("--slow"):
        # --slow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
