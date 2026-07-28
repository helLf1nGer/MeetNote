"""PDF and sidecar output, with attention to non-Latin scripts.

Skipped unless fpdf2 is installed.
"""

import builtins
import json

import pytest

pytest.importorskip('fpdf', reason='fpdf2 not installed')

from fpdf import FPDF  # noqa: E402

from utils import output_generator  # noqa: E402
from utils.config_manager import (  # noqa: E402
    DEFAULT_OUTPUT_FORMATS,
    SUPPORTED_OUTPUT_FORMATS,
)


def transcript_with_times():
    """A short two-speaker transcript with the usual combiner segment shape."""
    return [
        {'speaker': 'SPEAKER_00', 'text': 'Привет, как дела?', 'start': 0.0, 'end': 2.5},
        {'speaker': 'SPEAKER_00', 'text': 'Начнём встречу.', 'start': 2.5, 'end': 4.0},
        {'speaker': 'SPEAKER_01', 'text': 'Доброго дня!', 'start': 61.25, 'end': 63.0},
        {'speaker': 'SPEAKER_00', 'text': 'All good, thanks.', 'start': 3725.5, 'end': 3730.0},
    ]


class TestLatin1Detection:
    def test_ascii_is_latin1(self):
        assert output_generator._is_latin1('SPEAKER_00: hello there')

    def test_accented_latin_is_latin1(self):
        assert output_generator._is_latin1('café naïve façade')

    def test_cyrillic_is_not_latin1(self):
        assert not output_generator._is_latin1('СПИКЕР_00: Привет, как дела?')

    def test_ukrainian_is_not_latin1(self):
        assert not output_generator._is_latin1('Доброго дня, ласкаво просимо')


def test_unicode_font_is_bundled():
    """Cyrillic output depends on this font being present."""
    assert output_generator.find_unicode_font() is not None


def test_cyrillic_renders_with_the_bundled_font(tmp_path):
    font = output_generator.find_unicode_font()
    assert font is not None

    pdf = FPDF()
    pdf.add_page()
    pdf.add_font(font.stem, '', str(font))
    pdf.set_font(font.stem, size=12)
    pdf.multi_cell(0, 14, 'СПИКЕР_00: Привіт! Це тестова транскрипція.')

    out = tmp_path / 'cyrillic.pdf'
    pdf.output(str(out))
    assert out.stat().st_size > 0


def test_missing_font_raises_rather_than_emitting_an_empty_pdf(monkeypatch):
    """
    Regression test.

    The old code fell back to a Latin-1 core font and then swallowed the
    per-line encoding error, so a Russian transcript produced a valid but
    completely empty PDF. Failing loudly is the point.
    """
    monkeypatch.setattr(output_generator, 'find_unicode_font', lambda: None)

    pdf = FPDF()
    pdf.add_page()
    with pytest.raises(RuntimeError, match='Latin-1'):
        output_generator._apply_font(pdf, ['СПИКЕР_00: Привет'], 12)


def test_missing_font_is_fine_for_latin_text(monkeypatch):
    monkeypatch.setattr(output_generator, 'find_unicode_font', lambda: None)

    pdf = FPDF()
    pdf.add_page()
    output_generator._apply_font(pdf, ['SPEAKER_00: hello'], 12)  # must not raise


def test_create_pdf_writes_cyrillic_transcript(tmp_path, isolated_config):
    out = _configure(tmp_path, isolated_config)

    transcript = [
        {'speaker': 'SPEAKER_00', 'text': 'Привет, как дела?'},
        {'speaker': 'SPEAKER_01', 'text': 'Доброго дня!'},
        {'speaker': 'SPEAKER_00', 'text': 'All good, thanks.'},
    ]

    output = output_generator.create_pdf(transcript, '/some/where/встреча.mp4')

    assert output == str(out / 'встреча_transcription.pdf')
    from pathlib import Path
    assert Path(output).stat().st_size > 0


class TestFormatTimestamp:
    def test_plain_format(self):
        assert output_generator._format_timestamp(0) == '00:00:00'
        assert output_generator._format_timestamp(61.25) == '00:01:01'
        assert output_generator._format_timestamp(3725.5) == '01:02:05'

    def test_srt_format_keeps_milliseconds(self):
        assert output_generator._format_timestamp(0, srt=True) == '00:00:00,000'
        assert output_generator._format_timestamp(61.25, srt=True) == '00:01:01,250'
        assert output_generator._format_timestamp(3725.5, srt=True) == '01:02:05,500'

    def test_hours_past_ten_are_not_truncated(self):
        assert output_generator._format_timestamp(36000) == '10:00:00'

    def test_missing_or_negative_returns_none(self):
        # The caller decides what to do: text formats drop the stamp, SRT drops
        # the whole segment.
        assert output_generator._format_timestamp(None) is None
        assert output_generator._format_timestamp(-1) is None

    def test_unusable_values_return_none(self):
        assert output_generator._format_timestamp('not a number') is None
        assert output_generator._format_timestamp(float('nan')) is None
        # True would otherwise become 1.0 seconds via the int/bool relationship.
        assert output_generator._format_timestamp(True) is None

    def test_numeric_strings_are_accepted(self):
        # Some LLM combiners round-trip segments through JSON as strings.
        assert output_generator._format_timestamp('61.25') == '00:01:01'

    @pytest.mark.parametrize('seconds,expected', [
        (1.001, '00:00:01,001'),
        (2.002, '00:00:02,002'),
        (0.007, '00:00:00,007'),
        (4.004, '00:00:04,004'),
        (8.008, '00:00:08,008'),
        (16.016, '00:00:16,016'),
    ])
    def test_srt_milliseconds_survive_binary_float_representation(self, seconds, expected):
        """
        Regression test.

        0.001 has no exact binary form: 1.001 is stored as 1.000999..., so
        int(1.001 * 1000) is 1000 and the cue claimed a millisecond less than
        the decoder reported. word_timestamps=True makes these values routine.
        """
        assert output_generator._format_timestamp(seconds, srt=True) == expected

    def test_srt_rounding_carries_into_the_next_hour(self):
        # Rounding after the divmod chain would have produced 00:59:59,1000.
        assert output_generator._format_timestamp(3599.9996, srt=True) == '01:00:00,000'

    def test_srt_rounds_to_the_nearest_millisecond(self):
        assert output_generator._format_timestamp(1.0004, srt=True) == '00:00:01,000'
        assert output_generator._format_timestamp(1.0006, srt=True) == '00:00:01,001'

    def test_the_plain_stamp_still_truncates_at_the_second(self):
        """
        The two formats round differently on purpose: a plain stamp is something
        a reader jumps to, so it must never point past the words. SRT rounds
        because that is what subtitle consumers expect.
        """
        assert output_generator._format_timestamp(3599.9996) == '00:59:59'
        assert output_generator._format_timestamp(59.999) == '00:00:59'
        assert output_generator._format_timestamp(1.999) == '00:00:01'


class TestSegmentLine:
    def test_line_carries_the_timestamp(self):
        line = output_generator._segment_line(
            {'speaker': 'SPEAKER_00', 'text': 'hello', 'start': 61.0}
        )
        assert line == '[00:01:01] SPEAKER_00: hello'

    def test_missing_start_falls_back_to_the_plain_line(self):
        line = output_generator._segment_line({'speaker': 'SPEAKER_00', 'text': 'hello'})
        assert line == 'SPEAKER_00: hello'

    def test_timestamps_can_be_switched_off(self):
        line = output_generator._segment_line(
            {'speaker': 'SPEAKER_00', 'text': 'hello', 'start': 61.0}, with_timestamp=False
        )
        assert line == 'SPEAKER_00: hello'

    def test_missing_speaker_and_text_do_not_raise(self):
        assert output_generator._segment_line({}) == 'Unknown: '


class TestResolveFormats:
    def test_default_is_pdf_only(self):
        """
        The PDF is what almost every run is read from; writing four more files
        by default is churn, and worse when the output directory is a synced
        cloud folder. The rest are opt-in via the GUI or config.
        """
        assert output_generator._resolve_formats(None) == ['pdf']

    def test_every_supported_format_can_be_opted_into(self):
        # Regression guard: SUPPORTED_FORMATS was briefly derived from the
        # defaults, which made every opted-in format an 'unknown format'.
        requested = list(SUPPORTED_OUTPUT_FORMATS)
        assert output_generator._resolve_formats(requested) == requested

    def test_unknown_names_are_dropped(self, caplog):
        with caplog.at_level('WARNING'):
            formats = output_generator._resolve_formats(['txt', 'docx'])
        assert formats == ['pdf', 'txt']
        assert 'docx' in caplog.text

    def test_pdf_is_always_present(self):
        # It is the return value of create_pdf and what the tracker records.
        assert output_generator._resolve_formats(['srt']) == ['pdf', 'srt']

    def test_names_are_case_insensitive_and_deduplicated(self):
        assert output_generator._resolve_formats(['PDF', 'SRT', 'srt']) == ['pdf', 'srt']

    def test_a_bare_string_is_treated_as_one_format(self):
        assert output_generator._resolve_formats('srt') == ['pdf', 'srt']

    def test_a_non_list_falls_back_to_defaults(self, caplog):
        with caplog.at_level('WARNING'):
            formats = output_generator._resolve_formats({'txt': True})
        assert formats == list(DEFAULT_OUTPUT_FORMATS)


class TestSidecars:
    """create_pdf writes the sidecars alongside the PDF, same basename."""

    def _run(self, tmp_path, isolated_config, transcript=None, formats=None, source='встреча.mp4'):
        # These tests are about what each writer emits, so they opt into every
        # format explicitly rather than relying on the default - which is
        # deliberately pdf-only, and is asserted as such in TestResolveFormats.
        if formats is None:
            formats = list(SUPPORTED_OUTPUT_FORMATS)
        out_dir = _configure(tmp_path, isolated_config, formats=formats)

        transcript = transcript_with_times() if transcript is None else transcript
        pdf = output_generator.create_pdf(transcript, f'/some/where/{source}')
        return out_dir, pdf

    def test_all_sidecars_are_written(self, tmp_path, isolated_config):
        out, pdf = self._run(tmp_path, isolated_config)
        for suffix in ('.pdf', '.txt', '.json', '.srt', '.md'):
            assert (out / f'встреча_transcription{suffix}').is_file()

    def test_txt_lines_carry_timestamps(self, tmp_path, isolated_config):
        out, _ = self._run(tmp_path, isolated_config)
        lines = (out / 'встреча_transcription.txt').read_text(encoding='utf-8').splitlines()
        assert lines[0] == '[00:00:00] SPEAKER_00: Привет, как дела?'
        assert lines[2] == '[00:01:01] SPEAKER_01: Доброго дня!'
        assert lines[3] == '[01:02:05] SPEAKER_00: All good, thanks.'

    def test_txt_keeps_stampless_segments(self, tmp_path, isolated_config):
        out, _ = self._run(
            tmp_path, isolated_config,
            transcript=[{'speaker': 'SPEAKER_00', 'text': 'no times here'}],
        )
        text = (out / 'встреча_transcription.txt').read_text(encoding='utf-8')
        assert text.strip() == 'SPEAKER_00: no times here'

    def test_json_round_trips_cyrillic_and_numbers(self, tmp_path, isolated_config):
        out, _ = self._run(tmp_path, isolated_config)
        raw = (out / 'встреча_transcription.json').read_text(encoding='utf-8')

        # ensure_ascii=False: the file is readable, not С-escaped.
        assert 'Привет' in raw
        assert '\\u' not in raw

        segments = json.loads(raw)
        assert len(segments) == 4
        assert segments[0] == {
            'start': 0.0, 'end': 2.5, 'speaker': 'SPEAKER_00', 'text': 'Привет, как дела?',
        }
        assert segments[2]['start'] == 61.25

    def test_json_nulls_missing_timestamps(self, tmp_path, isolated_config):
        out, _ = self._run(
            tmp_path, isolated_config,
            transcript=[{'speaker': 'SPEAKER_00', 'text': 'no times'}],
        )
        segments = json.loads((out / 'встреча_transcription.json').read_text(encoding='utf-8'))
        assert segments[0]['start'] is None
        assert segments[0]['end'] is None

    def test_srt_is_well_formed(self, tmp_path, isolated_config):
        out, _ = self._run(tmp_path, isolated_config)
        blocks = (out / 'встреча_transcription.srt').read_text(
            encoding='utf-8'
        ).strip().split('\n\n')

        assert len(blocks) == 4
        assert blocks[0].splitlines() == [
            '1',
            '00:00:00,000 --> 00:00:02,500',
            'SPEAKER_00: Привет, как дела?',
        ]
        assert blocks[2].splitlines()[1] == '00:01:01,250 --> 00:01:03,000'
        assert blocks[3].splitlines()[1] == '01:02:05,500 --> 01:02:10,000'

    def test_srt_skips_segments_without_timestamps(self, tmp_path, isolated_config):
        out, _ = self._run(tmp_path, isolated_config, transcript=[
            {'speaker': 'SPEAKER_00', 'text': 'timed', 'start': 1.0, 'end': 2.0},
            {'speaker': 'SPEAKER_01', 'text': 'no start', 'end': 3.0},
            {'speaker': 'SPEAKER_01', 'text': 'no end', 'start': 4.0},
            {'speaker': 'SPEAKER_01', 'text': 'nothing at all'},
            {'speaker': 'SPEAKER_00', 'text': 'timed again', 'start': 5.0, 'end': 6.0},
        ])
        srt = (out / 'встреча_transcription.srt').read_text(encoding='utf-8')

        assert 'no start' not in srt
        assert 'no end' not in srt
        assert 'nothing at all' not in srt
        # Cue numbering stays contiguous over the skipped segments.
        blocks = srt.strip().split('\n\n')
        assert [b.splitlines()[0] for b in blocks] == ['1', '2']
        assert blocks[1].splitlines()[2] == 'SPEAKER_00: timed again'

    def test_srt_clamps_a_non_advancing_end(self, tmp_path, isolated_config):
        # A zero-length cue is silently dropped by most players.
        out, _ = self._run(tmp_path, isolated_config, transcript=[
            {'speaker': 'SPEAKER_00', 'text': 'instant', 'start': 10.0, 'end': 10.0},
            {'speaker': 'SPEAKER_00', 'text': 'reversed', 'start': 20.0, 'end': 19.0},
        ])
        blocks = (out / 'встреча_transcription.srt').read_text(
            encoding='utf-8'
        ).strip().split('\n\n')
        assert blocks[0].splitlines()[1] == '00:00:10,000 --> 00:00:10,500'
        assert blocks[1].splitlines()[1] == '00:00:20,000 --> 00:00:20,500'

    def test_srt_cues_keep_exact_milliseconds(self, tmp_path, isolated_config):
        # word_timestamps=True makes 0.001-step values routine, and those are
        # exactly the ones binary floats store just below the value.
        out, _ = self._run(tmp_path, isolated_config, transcript=[
            {'speaker': 'SPEAKER_00', 'text': 'first', 'start': 1.001, 'end': 2.002},
            {'speaker': 'SPEAKER_01', 'text': 'second', 'start': 4.004, 'end': 8.008},
        ])
        blocks = (out / 'встреча_transcription.srt').read_text(
            encoding='utf-8'
        ).strip().split('\n\n')

        assert blocks[0].splitlines()[1] == '00:00:01,001 --> 00:00:02,002'
        assert blocks[1].splitlines()[1] == '00:00:04,004 --> 00:00:08,008'

    def test_md_has_a_title_and_stamped_paragraphs(self, tmp_path, isolated_config):
        out, _ = self._run(tmp_path, isolated_config)
        md = (out / 'встреча_transcription.md').read_text(encoding='utf-8')

        assert md.startswith('# встреча\n')
        assert '**SPEAKER_01** _[00:01:01]_: Доброго дня!' in md

    def test_md_merges_consecutive_same_speaker_segments(self, tmp_path, isolated_config):
        out, _ = self._run(tmp_path, isolated_config)
        md = (out / 'встреча_transcription.md').read_text(encoding='utf-8')

        # The first two segments are both SPEAKER_00 and become one paragraph,
        # keeping the start of the turn.
        assert '**SPEAKER_00** _[00:00:00]_: Привет, как дела? Начнём встречу.' in md
        # Three turns for four segments.
        assert md.count('**SPEAKER_') == 3
        # The later SPEAKER_00 turn is separate - SPEAKER_01 interrupted it.
        assert '**SPEAKER_00** _[01:02:05]_: All good, thanks.' in md

    def test_md_omits_the_stamp_when_there_is_none(self, tmp_path, isolated_config):
        out, _ = self._run(
            tmp_path, isolated_config,
            transcript=[{'speaker': 'SPEAKER_00', 'text': 'no times'}],
        )
        md = (out / 'встреча_transcription.md').read_text(encoding='utf-8')
        assert '**SPEAKER_00**: no times' in md

    def test_only_configured_formats_are_written(self, tmp_path, isolated_config):
        out, _ = self._run(tmp_path, isolated_config, formats=['pdf', 'srt'])

        assert (out / 'встреча_transcription.pdf').is_file()
        assert (out / 'встреча_transcription.srt').is_file()
        assert not (out / 'встреча_transcription.txt').exists()
        assert not (out / 'встреча_transcription.json').exists()
        assert not (out / 'встреча_transcription.md').exists()

    def test_pdf_is_written_even_when_not_requested(self, tmp_path, isolated_config):
        # It is the return value and what the transcription tracker records.
        out, pdf = self._run(tmp_path, isolated_config, formats=[])
        assert (out / 'встреча_transcription.pdf').is_file()
        assert pdf.endswith('встреча_transcription.pdf')

    def test_unknown_format_is_skipped_without_failing(self, tmp_path, isolated_config, caplog):
        with caplog.at_level('WARNING'):
            out, _ = self._run(tmp_path, isolated_config, formats=['pdf', 'txt', 'docx'])

        assert (out / 'встреча_transcription.txt').is_file()
        assert not (out / 'встреча_transcription.docx').exists()
        assert 'docx' in caplog.text

    def test_a_sidecar_failure_does_not_fail_the_run(self, tmp_path, isolated_config,
                                                     monkeypatch, caplog):
        """
        The PDF is the primary artifact. A read-only disk, a locked file or a
        path-length limit hitting one sidecar must not lose the transcription.
        """
        real_open = builtins.open

        def failing_open(file, *args, **kwargs):
            # Narrow to the sidecar: config.json goes through open() too, and
            # breaking that would test the config manager instead.
            if str(file).endswith('_transcription.json'):
                raise OSError('disk on fire')
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, 'open', failing_open)

        with caplog.at_level('WARNING'):
            out, pdf = self._run(tmp_path, isolated_config)  # must not raise

        assert (out / 'встреча_transcription.pdf').is_file()
        assert not (out / 'встреча_transcription.json').exists()
        # The formats after the failing one still got written.
        assert (out / 'встреча_transcription.srt').is_file()
        assert (out / 'встреча_transcription.md').is_file()
        assert 'disk on fire' in caplog.text


class TestMalformedOutputBlock:
    """
    Regression tests.

    `"output": "bad"` is valid JSON, so _merge_defaults leaves it alone and it
    arrives here intact. Calling .get on it raised AttributeError before the PDF
    was written - losing a finished transcription at the very last step, after
    every expensive stage had already succeeded, over a config typo.
    """

    def _run_with_output_block(self, tmp_path, isolated_config, block):
        out_dir = tmp_path / 'out'
        isolated_config.config['output_directory'] = str(out_dir)
        # Assigned rather than updated: the point is that it is not a mapping.
        isolated_config.config['output'] = block
        isolated_config.save_config()

        pdf = output_generator.create_pdf(
            [{'speaker': 'SPEAKER_00', 'text': 'Привет', 'start': 0.0, 'end': 1.0}],
            '/some/where/встреча.mp4',
        )
        return out_dir, pdf

    @pytest.mark.parametrize('block', ['bad', ['pdf', 'srt'], 42, 1.5, True])
    def test_a_non_mapping_output_block_does_not_lose_the_pdf(
        self, tmp_path, isolated_config, block, caplog
    ):
        with caplog.at_level('WARNING'):
            out, pdf = self._run_with_output_block(tmp_path, isolated_config, block)

        assert (out / 'встреча_transcription.pdf').is_file()
        assert pdf.endswith('встреча_transcription.pdf')
        assert 'output' in caplog.text

    def test_it_falls_back_to_the_default_formats(self, tmp_path, isolated_config):
        out, _ = self._run_with_output_block(tmp_path, isolated_config, 'bad')
        # Exactly the defaults, as though the block had been absent.
        for fmt in DEFAULT_OUTPUT_FORMATS:
            assert (out / f'встреча_transcription.{fmt}').is_file()
        for fmt in set(SUPPORTED_OUTPUT_FORMATS) - set(DEFAULT_OUTPUT_FORMATS):
            assert not (out / f'встреча_transcription.{fmt}').exists()

    def test_timestamps_keep_their_default(self, tmp_path, isolated_config):
        # timestamps_in_pdf defaults to True, which the PDF lines reflect.
        rendered = []
        real_multi_cell = FPDF.multi_cell

        def spy(self, w, h=0, text='', *args, **kwargs):
            rendered.append(text)
            return real_multi_cell(self, w, h, text, *args, **kwargs)

        FPDF.multi_cell = spy
        try:
            self._run_with_output_block(tmp_path, isolated_config, 'bad')
        finally:
            FPDF.multi_cell = real_multi_cell

        assert rendered and rendered[0].startswith('[00:00:00] ')


class TestPdfTimestamps:
    def test_pdf_lines_include_timestamps_by_default(self, tmp_path, isolated_config, monkeypatch):
        out = _configure(tmp_path, isolated_config)
        rendered = _capture_pdf_lines(monkeypatch)

        pdf = output_generator.create_pdf(transcript_with_times(), '/some/where/встреча.mp4')

        assert rendered[0] == '[00:00:00] SPEAKER_00: Привет, как дела?'
        assert rendered[3] == '[01:02:05] SPEAKER_00: All good, thanks.'
        assert (out / 'встреча_transcription.pdf').stat().st_size > 0
        assert pdf == str(out / 'встреча_transcription.pdf')

    def test_timestamps_in_pdf_can_be_switched_off(self, tmp_path, isolated_config, monkeypatch):
        out = _configure(tmp_path, isolated_config, timestamps_in_pdf=False)
        rendered = _capture_pdf_lines(monkeypatch)

        output_generator.create_pdf(transcript_with_times(), '/some/where/встреча.mp4')

        assert rendered[0] == 'SPEAKER_00: Привет, как дела?'
        assert not any(line.startswith('[') for line in rendered)
        assert (out / 'встреча_transcription.pdf').stat().st_size > 0

    def test_the_txt_sidecar_keeps_timestamps_when_the_pdf_drops_them(
        self, tmp_path, isolated_config
    ):
        # timestamps_in_pdf is a PDF-layout preference, not a data switch.
        out = _configure(tmp_path, isolated_config,
                         formats=['pdf', 'txt'], timestamps_in_pdf=False)

        output_generator.create_pdf(transcript_with_times(), '/some/where/встреча.mp4')

        text = (out / 'встреча_transcription.txt').read_text(encoding='utf-8')
        assert text.startswith('[00:00:00] SPEAKER_00:')


def _configure(tmp_path, isolated_config, **output_options):
    """
    Point the config at a temp output directory and persist it.

    create_pdf calls load_config(), which re-reads the file from disk, so
    settings left only in the fixture's in-memory dict would be overwritten by
    the defaults before they were read.
    """
    out_dir = tmp_path / 'out'
    isolated_config.config['output_directory'] = str(out_dir)
    isolated_config.config['output'].update(output_options)
    isolated_config.save_config()
    return out_dir


def _capture_pdf_lines(monkeypatch):
    """Record the text handed to FPDF.multi_cell, still rendering it."""
    rendered = []
    real_multi_cell = FPDF.multi_cell

    def spy(self, w, h=0, text='', *args, **kwargs):
        rendered.append(text)
        return real_multi_cell(self, w, h, text, *args, **kwargs)

    monkeypatch.setattr(FPDF, 'multi_cell', spy)
    return rendered
