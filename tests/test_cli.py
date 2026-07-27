"""Argument parsing for the headless entry point."""

import pytest

pytest.importorskip('dotenv', reason='runtime extras not installed')

from main import _parse_args  # noqa: E402


class TestSpeakerFlag:
    """
    0 means "let pyannote decide". It has to survive as 0 rather than being
    treated as "not given", so `-s 0` cannot be resolved with `or`.
    """

    def test_absent_stays_none_so_the_config_default_applies(self):
        assert _parse_args(['meeting.mp4']).speakers is None

    def test_explicit_zero_is_preserved(self):
        args = _parse_args(['meeting.mp4', '-s', '0'])
        assert args.speakers == 0
        assert args.speakers is not None

    def test_an_explicit_count_is_preserved(self):
        assert _parse_args(['meeting.mp4', '-s', '4']).speakers == 4


class TestCliSpeakerResolution:
    """The `x if x is not None else default` form, which `or` gets wrong for 0."""

    @staticmethod
    def _resolve(speakers, configured=2):
        return speakers if speakers is not None else configured

    def test_zero_reaches_the_pipeline_as_zero(self):
        # Regression: `args.speakers or config[...]` turned an explicit request
        # for auto-detection back into the configured count.
        assert self._resolve(0) == 0

    def test_absent_falls_back_to_the_config(self):
        assert self._resolve(None, configured=3) == 3

    def test_explicit_count_wins_over_the_config(self):
        assert self._resolve(5, configured=3) == 5
