"""Shared test configuration.

Puts ``src`` on the import path so the packages inside it (utils, transcription,
audio, ...) import as top-level modules, which is how they refer to each other
at runtime.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / 'src'

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_addoption(parser):
    parser.addoption(
        '--run-network', action='store_true', default=False,
        help='Run tests that contact real network services.',
    )


def pytest_configure(config):
    config.addinivalue_line('markers', 'network: contacts a real network service')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--run-network'):
        return
    skip = pytest.mark.skip(reason='needs --run-network')
    for item in items:
        if 'network' in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def run_network(request):
    """Present so network tests fail loudly if the marker is ever removed."""
    if not request.config.getoption('--run-network'):
        pytest.skip('needs --run-network')


@pytest.fixture
def isolated_config(tmp_path):
    """
    A ConfigManager pointed at a temporary directory.

    ConfigManager is a per-file singleton, so the real instance's paths are
    redirected and then restored rather than a second instance being built.
    """
    from utils.config_manager import ConfigManager

    manager = ConfigManager()
    saved = (manager.config_dir, manager.config_file, manager.template_file, manager.config)

    manager.config_dir = str(tmp_path)
    manager.config_file = str(tmp_path / 'config.json')
    manager.template_file = str(tmp_path / 'config.template.json')
    manager.config = manager._get_default_config()

    yield manager

    manager.config_dir, manager.config_file, manager.template_file, manager.config = saved
