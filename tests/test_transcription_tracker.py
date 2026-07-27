"""Transcription tracking and its file-hash cache."""

import pytest

from utils.transcription_tracker import TranscriptionTracker


@pytest.fixture
def tracker(tmp_path):
    """A tracker writing to a temporary file rather than the real data/ dir."""
    instance = TranscriptionTracker.__new__(TranscriptionTracker)
    instance.tracker_file = tmp_path / 'transcribed_files.json'
    instance._ensure_tracker_file()
    instance.transcribed_files = instance._load_tracker()
    instance._hash_cache = {}
    return instance


@pytest.fixture
def media_file(tmp_path):
    path = tmp_path / 'recording.wav'
    path.write_bytes(b'original audio content')
    return path


def test_hash_is_stable_for_unchanged_file(tracker, media_file):
    assert tracker._calculate_file_hash(media_file) == tracker._calculate_file_hash(media_file)


def test_hash_changes_when_file_is_modified(tracker, media_file):
    """
    Regression test.

    The hash used to be memoized with functools.lru_cache keyed on the path, so
    a re-recorded file kept its old hash and was reported as already
    transcribed. The size/mtime cache key is what makes reuse safe.
    """
    before = tracker._calculate_file_hash(media_file)

    media_file.write_bytes(b'a completely different recording')
    import os
    stats = os.stat(media_file)
    os.utime(media_file, (stats.st_atime, stats.st_mtime + 10))

    assert tracker._calculate_file_hash(media_file) != before


def test_cache_is_used_for_repeat_lookups(tracker, media_file):
    tracker._calculate_file_hash(media_file)
    assert len(tracker._hash_cache) == 1
    tracker._calculate_file_hash(media_file)
    assert len(tracker._hash_cache) == 1


def test_mark_and_query(tracker, media_file):
    assert not tracker.is_transcribed(media_file)

    assert tracker.mark_as_transcribed(media_file, '/out/recording.pdf')

    assert tracker.is_transcribed(media_file)
    history = tracker.get_transcription_history(media_file)
    assert len(history) == 1
    assert history[0]['output_path'] == '/out/recording.pdf'


def test_history_accumulates(tracker, media_file):
    tracker.mark_as_transcribed(media_file, '/out/first.pdf')
    tracker.mark_as_transcribed(media_file, '/out/second.pdf')
    assert len(tracker.get_transcription_history(media_file)) == 2


def test_modified_file_is_not_reported_as_transcribed(tracker, media_file):
    """The point of the hash fix, stated at the level users experience it."""
    tracker.mark_as_transcribed(media_file, '/out/recording.pdf')
    assert tracker.is_transcribed(media_file)

    media_file.write_bytes(b'a re-recorded meeting, same filename')
    import os
    stats = os.stat(media_file)
    os.utime(media_file, (stats.st_atime, stats.st_mtime + 10))

    assert not tracker.is_transcribed(media_file)


def test_missing_file_is_handled(tracker, tmp_path):
    assert not tracker.is_transcribed(tmp_path / 'does-not-exist.wav')
    assert tracker.get_transcription_history(tmp_path / 'does-not-exist.wav') == []


def test_corrupt_tracker_file_recovers(tracker):
    tracker.tracker_file.write_text('{ not json')
    assert tracker._load_tracker() == {}
