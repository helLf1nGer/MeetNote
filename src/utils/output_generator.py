import json
import logging
import math
import os
from pathlib import Path

from fpdf import FPDF

from .config_manager import DEFAULT_OUTPUT_FORMATS, ConfigManager

logger = logging.getLogger(__name__)
config_manager = ConfigManager()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# fpdf2's built-in fonts are Latin-1 only, so anything outside that range
# (Cyrillic, Greek, CJK, even smart quotes) needs an embedded TTF.
UNICODE_FONT_CANDIDATES = [
    PROJECT_ROOT / 'Fonts' / 'DejaVuSans.ttf',
    PROJECT_ROOT / 'dejavu-sans' / 'DejaVuSans.ttf',
]

# Every format create_pdf knows how to emit. 'pdf' is included for
# completeness, but it is written unconditionally - see _resolve_formats.
SUPPORTED_FORMATS = tuple(DEFAULT_OUTPUT_FORMATS)

# Minimum on-screen duration for a subtitle cue whose end timestamp is missing
# or not after its start; most players silently drop zero-length cues.
MIN_CUE_SECONDS = 0.5


def create_pdf(final_transcription, original_file_path):
    """
    Write the transcript to disk and return the path of the PDF.

    The PDF is the primary artifact: the pipeline returns it and the
    transcription tracker records it, so it is always produced. The sidecar
    formats (txt/json/srt/md) are best-effort - a failure there is logged and
    the run still succeeds.
    """
    config = config_manager.load_config()
    output_dir = config['output_directory']
    font_size = config['pdf_output']['font_size']
    line_spacing = config['pdf_output']['line_spacing']

    output_options = config.get('output') or {}
    if not isinstance(output_options, dict):
        # `"output": "bad"` is valid JSON and survives _merge_defaults, so it
        # reaches here intact and would raise on .get - losing a finished
        # transcript at the very last step, over a config typo.
        logger.warning("[Output] config 'output' should be an object, got %s; "
                       "using defaults.", type(output_options).__name__)
        output_options = {}

    formats = _resolve_formats(output_options.get('formats'))
    timestamps_in_pdf = bool(output_options.get('timestamps_in_pdf', True))

    logger.info("[Output] Using output directory: %s", output_dir)
    logger.info("[Output] Creating PDF document...")

    lines = [
        _segment_line(item, with_timestamp=timestamps_in_pdf)
        for item in final_transcription
    ]

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    _apply_font(pdf, lines, font_size)

    skipped = 0
    for line in lines:
        try:
            pdf.multi_cell(0, line_spacing * font_size, line, align='L', border=0)
            pdf.ln()
        except Exception as e:
            skipped += 1
            logger.error("[Output] Error adding text to PDF: %s", e)

    if skipped:
        logger.warning("[Output] %d of %d lines could not be rendered.", skipped, len(lines))

    os.makedirs(output_dir, exist_ok=True)
    pdf_file_name = os.path.join(
        output_dir,
        os.path.splitext(os.path.basename(original_file_path))[0] + '_transcription.pdf'
    )

    try:
        pdf.output(pdf_file_name)
        logger.info("[Output] Transcription PDF saved as %s", pdf_file_name)
    except Exception as e:
        logger.error("[Output] Failed to save PDF. Error: %s", e)
        raise

    _write_sidecars(final_transcription, pdf_file_name, original_file_path, formats)

    return pdf_file_name


def _resolve_formats(configured):
    """
    Normalise ``output.formats`` into a list of known format names.

    Unknown names are warned about and dropped rather than raising: a typo in
    the config should not cost the user a finished transcription.
    """
    if configured is None:
        return list(DEFAULT_OUTPUT_FORMATS)

    # A single format written as a bare string is an easy config mistake, and
    # iterating it character by character would produce nothing but warnings.
    if isinstance(configured, str):
        configured = [configured]

    if not isinstance(configured, (list, tuple)):
        logger.warning(
            "[Output] output.formats should be a list, got %s; using defaults.",
            type(configured).__name__,
        )
        return list(DEFAULT_OUTPUT_FORMATS)

    formats = []
    for name in configured:
        key = str(name).strip().lower()
        if key not in SUPPORTED_FORMATS:
            logger.warning(
                "[Output] Unknown output format %r; skipping. Known formats: %s",
                name, ', '.join(SUPPORTED_FORMATS),
            )
            continue
        if key not in formats:
            formats.append(key)

    if 'pdf' not in formats:
        formats.insert(0, 'pdf')
    return formats


def _write_sidecars(final_transcription, pdf_file_name, original_file_path, formats):
    """Write the non-PDF formats next to the PDF, one warning per failure."""
    base_path = os.path.splitext(pdf_file_name)[0]

    for fmt in formats:
        writer = _SIDECAR_WRITERS.get(fmt)
        if writer is None:
            # 'pdf' is already on disk; unknown names were dropped upstream.
            continue

        path = f"{base_path}.{fmt}"
        try:
            writer(final_transcription, path, original_file_path)
            logger.info("[Output] Wrote %s sidecar: %s", fmt, path)
        except Exception as e:
            logger.warning("[Output] Could not write %s sidecar %s: %s", fmt, path, e)


def _write_txt(final_transcription, path, original_file_path):
    """Plain text, one line per segment: ``[HH:MM:SS] SPEAKER: text``."""
    with open(path, 'w', encoding='utf-8') as f:
        for item in final_transcription:
            f.write(_segment_line(item) + '\n')


def _write_json(final_transcription, path, original_file_path):
    """
    Machine-readable segments, for re-processing without re-transcribing.

    ``start``/``end`` stay numeric (or null) so consumers can do arithmetic on
    them; formatting is left to whoever renders the data.
    """
    segments = [
        {
            'start': _numeric(item.get('start')),
            'end': _numeric(item.get('end')),
            'speaker': item.get('speaker', 'Unknown'),
            'text': _segment_text(item),
        }
        for item in final_transcription
    ]

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)


def _write_srt(final_transcription, path, original_file_path):
    """
    SubRip subtitles, for playing the recording back with captions.

    Segments without both timestamps are skipped: a cue with no time range has
    nowhere to go, and inventing one would desynchronise everything after it.
    """
    cues = []
    for item in final_transcription:
        start = _numeric(item.get('start'))
        end = _numeric(item.get('end'))
        if start is None or start < 0 or end is None:
            continue
        if end <= start:
            end = start + MIN_CUE_SECONDS
        cues.append((start, end, item))

    with open(path, 'w', encoding='utf-8') as f:
        for index, (start, end, item) in enumerate(cues, start=1):
            begin = _format_timestamp(start, srt=True)
            finish = _format_timestamp(end, srt=True)
            speaker = item.get('speaker', 'Unknown')
            f.write(f"{index}\n")
            f.write(f"{begin} --> {finish}\n")
            f.write(f"{speaker}: {_segment_text(item)}\n\n")


def _write_md(final_transcription, path, original_file_path):
    """Markdown meeting notes: one paragraph per speaker turn."""
    title = os.path.splitext(os.path.basename(original_file_path))[0]

    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"# {title}\n\n")
        for speaker, stamp, text in _merge_consecutive_speakers(final_transcription):
            prefix = f"**{speaker}**"
            if stamp:
                prefix += f" _[{stamp}]_"
            f.write(f"{prefix}: {text}\n\n")


def _merge_consecutive_speakers(final_transcription):
    """
    Group consecutive segments by speaker into ``(speaker, stamp, text)`` turns.

    Combiners emit one segment per utterance, which as markdown reads as a wall
    of one-line paragraphs. The timestamp kept is the start of the turn, since
    that is what a reader scanning the notes wants to jump to.
    """
    turns = []

    for item in final_transcription:
        speaker = item.get('speaker', 'Unknown')
        text = _segment_text(item).strip()

        if turns and turns[-1][0] == speaker:
            previous = turns[-1]
            previous[2] = ' '.join(part for part in (previous[2], text) if part)
            continue

        turns.append([speaker, _format_timestamp(item.get('start')), text])

    return turns


def _segment_line(item, with_timestamp=True):
    """Render one segment as a text line, with the timestamp when there is one."""
    speaker = item.get('speaker', 'Unknown')
    text = _segment_text(item)
    stamp = _format_timestamp(item.get('start')) if with_timestamp else None

    # Some combiners (the LLM-based ones especially) return text and speaker
    # without timestamps, so the un-stamped line stays a supported shape.
    if stamp is None:
        return f"{speaker}: {text}"
    return f"[{stamp}] {speaker}: {text}"


def _segment_text(item):
    """Segment text as a string; missing or null text becomes empty."""
    text = item.get('text')
    return '' if text is None else str(text)


def _format_timestamp(seconds, srt=False):
    """
    Format a position in the recording, or return None if there isn't one.

    Returning None rather than a placeholder lets every caller decide for
    itself: the text formats fall back to an un-stamped line, SRT skips the
    segment entirely.

    The two formats round differently, on purpose. The plain stamp truncates at
    the second so it never points past the moment the words were spoken, which
    is what a reader jumping to a position needs. SRT rounds to the nearest
    millisecond, which is what subtitle consumers expect and what keeps the
    displayed value honest: ``0.001`` increments have no exact binary form, so
    truncating ``1.001 * 1000`` (stored as 1000.9999...) would silently report
    one millisecond less than the decoder actually produced.
    """
    total = _numeric(seconds)
    if total is None or total < 0:
        return None

    if srt:
        # Rounded before the divmod chain, so a value just under an hour boundary
        # such as 3599.9996 carries cleanly into 01:00:00,000 instead of
        # overflowing into 00:59:59,1000.
        milliseconds = int(round(total * 1000))
    else:
        milliseconds = int(total) * 1000

    hours, rest = divmod(milliseconds, 3600_000)
    minutes, rest = divmod(rest, 60_000)
    whole_seconds, milliseconds = divmod(rest, 1000)

    if srt:
        return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}"


def _numeric(value):
    """Coerce a timestamp to float, or None when it is not a usable number."""
    # bool is an int subclass, and True would silently become 1.0 seconds.
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _apply_font(pdf, lines, font_size):
    """
    Select a font that can render the transcript.

    A Unicode TTF is preferred. If none is installed the built-in font is used,
    but only after confirming the text is Latin-1 - otherwise every line would
    fail to render and the PDF would come out silently empty.
    """
    font_path = find_unicode_font()

    if font_path:
        try:
            font_name = font_path.stem
            pdf.add_font(font_name, '', str(font_path))
            pdf.set_font(font_name, size=font_size)
            logger.info("[Output] Using Unicode font: %s (%s)", font_name, font_path)
            return
        except Exception as e:
            logger.error("[Output] Failed to load font %s: %s", font_path, e)

    if not _is_latin1('\n'.join(lines)):
        raise RuntimeError(
            "This transcript contains characters outside Latin-1 (for example "
            "Cyrillic), which the built-in PDF fonts cannot render. Place "
            "DejaVuSans.ttf in the 'Fonts' directory and run again. Searched: "
            + ', '.join(str(path) for path in UNICODE_FONT_CANDIDATES)
        )

    pdf.set_font('Helvetica', size=font_size)
    logger.info("[Output] Using built-in font: Helvetica")


def find_unicode_font():
    """Return the first bundled Unicode TTF that exists, or None."""
    for candidate in UNICODE_FONT_CANDIDATES:
        if candidate.is_file():
            return candidate
    logger.warning(
        "[Output] No Unicode font found in %s",
        ' or '.join(str(path.parent) for path in UNICODE_FONT_CANDIDATES),
    )
    return None


def _is_latin1(text):
    try:
        text.encode('latin-1')
        return True
    except UnicodeEncodeError:
        return False


# Declared after the writers so the names resolve at import time.
_SIDECAR_WRITERS = {
    'txt': _write_txt,
    'json': _write_json,
    'srt': _write_srt,
    'md': _write_md,
}
