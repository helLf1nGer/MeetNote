"""
Connectivity check for the Colab/ngrok diarization service.

This talks to a real endpoint, so it is skipped by default. Run it explicitly
when you want to confirm the tunnel is up:

    pytest tests/test_colab_service.py --run-network
"""

import pytest
import requests

from utils.config_manager import ConfigManager

pytestmark = pytest.mark.network


def test_colab_service_is_reachable(run_network):
    colab_url = ConfigManager().get('colab_service_url')
    if not colab_url:
        pytest.skip('colab_service_url is not set in config')

    try:
        response = requests.get(f"{colab_url}/test", timeout=10)
    except requests.exceptions.ConnectionError:
        pytest.fail(f"Could not reach {colab_url} - the ngrok URL may have expired "
                    "or the Colab notebook is not running.")
    except requests.exceptions.Timeout:
        pytest.fail(f"{colab_url} timed out after 10s.")

    assert response.status_code == 200, (
        f"{colab_url}/test returned {response.status_code}: {response.text[:200]}"
    )
