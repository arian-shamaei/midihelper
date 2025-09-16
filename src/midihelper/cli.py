from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

from mido import MidiFile, merge_tracks


# -------------------------------
# Utilities
# -------------------------------

def _print_bracketed(values: Iterable) -> str:
    """
    Render a list-like sequence (supports nested lists) to a single
    bracketed line with space-separated items. Example:
    [x x [y y] x]
    """
    out_elems = []
    for v in values:
        if isinstance(v, (list, tuple)):
            out_elems.append(_print_bracketed(v))
        elif isinstance(v, str):
            out_elems.append(v)
        else:
            # Format numbers with 3 decimal places; keep ints as-is
            try:
                if isinstance(v, float):
                    out_elems.append(f"{v:.3f}")
                elif isinstance(v, int):
                    out_elems.append(str(v))
                else:
                    out_elems.append(str(v))
            except Exception:
                out_elems.append(str(v))
    return "[" + " ".join(out_elems) + "]"


def _format_sigfig(x, sigfigs: int):
    """Format a number to N significant figures without pandas dependency."""
    try:
        # Treat None/NaN as empty string to preserve count
        if x is None:
            return ""
        if isinstance(x, float):
            try:
                if math.isnan(x):
                    return ""
            except Exception:
                pass
        return f"{float(x):.{sigfigs}g}"
    except Exception:
        # Fallback to original
        return str(x)


# -------------------------------
# CSV Extract Feature
# -------------------------------

def extract_csv_column(
    input_path: Path,
    column: str | None = None,
    sigfigs: int | None = None,
    track: str | None = None,
    output_path: Path | None = None,
    non_interactive: bool = False,
) -> int:
    """
    Extract a column from a CSV.
    - Dynamically reads headers from the file.
    - If column is numeric, can round to sigfigs.
    - Optional filter by 'track' column.
    - Prints a single bracketed list, or writes to file if output_path is set.

    Returns exit code (0=success).
    """
    # CSV-based implementation (no pandas) with case-insensitive headers
    try:
        import csv as _csv
    except Exception as e:
        print(f"Error importing csv module: {e}", file=sys.stderr)
        return 1

    try:
        f = input_path.open(newline="", encoding="utf-8")
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        return 1

    with f:
        reader = _csv.DictReader(f)
        headers = reader.fieldnames or []
        if not headers:
            print("Error: CSV has no headers.", file=sys.stderr)
            return 1

        header_lut = {h.lower(): h for h in headers}

        # Column resolution with synonyms
        col_input = column
        if col_input is None and not non_interactive:
            print("Available headers:")
            for i, h in enumerate(headers, start=1):
                print(f"  {i}. {h}")
            chosen = input("Select a column (name or number): ").strip()
            if chosen.isdigit():
                idx = int(chosen) - 1
                if idx < 0 or idx >= len(headers):
                    print("Invalid choice.", file=sys.stderr)
                    return 1
                col_key = headers[idx]
            else:
                col_key = header_lut.get(chosen.lower())
                if not col_key:
                    print(f"Error: '{chosen}' not found in headers.", file=sys.stderr)
                    return 1
        else:
            # Try case-insensitive match and common aliases
            wanted = (col_input or "").lower()
            synonyms = [wanted, "midi note", "note", "notes", "time (seconds)", "time_s"]
            col_key = None
            for name in synonyms:
                col_key = header_lut.get(name)
                if col_key:
                    break
            if not col_key:
                print(f"Error: column '{column}' not found.", file=sys.stderr)
                return 1

        # Track filter (case-insensitive header)
        track_key = header_lut.get("track")
        if track_key and track is None and not non_interactive:
            ans = input("Filter by 'track'? (y/n): ").strip().lower()
            if ans == "y":
                track = input("Enter track value (compared as string): ").strip()

        # Identify a time column (for grouping simultaneous events)
        time_key = None
        for name in ("time (seconds)", "time_s", "time", "seconds"):
            if name in header_lut:
                time_key = header_lut[name]
                break

        # Collect values with optional time for grouping
        grouped_values = []  # will become a list with nested lists for equal-time groups
        current_time = None
        current_group: list = []

        def _finalize_group():
            if not current_group:
                return
            if len(current_group) == 1:
                grouped_values.append(current_group[0])
            else:
                grouped_values.append(list(current_group))

        for row in reader:
            if track_key is not None and track is not None:
                if str(row.get(track_key, "")) != str(track):
                    continue

            v = row.get(col_key, "")
            # Try to coerce to int if integer-like, else float for numeric
            if isinstance(v, str) and v != "":
                try:
                    fv = float(v)
                    iv = int(fv)
                    if fv == float(iv):
                        v = iv
                    else:
                        v = _format_sigfig(fv, sigfigs) if sigfigs is not None else fv
                except Exception:
                    pass

            tval = row.get(time_key, None) if time_key else None
            if time_key is None:
                # No time grouping available; behave as before
                grouped_values.append(v)
                continue

            if current_time is None:
                current_time = tval
                current_group = [v]
            elif tval == current_time:
                current_group.append(v)
            else:
                _finalize_group()
                current_time = tval
                current_group = [v]

        # Flush last group if grouping by time
        if time_key is not None:
            _finalize_group()
            values = grouped_values
        else:
            values = grouped_values

    out_text = _print_bracketed(values)

    if output_path is not None:
        try:
            output_path.write_text(out_text, encoding="utf-8")
            print(str(output_path))
        except Exception as e:
            print(f"Error writing output: {e}", file=sys.stderr)
            return 1
    else:
        print(out_text)

    return 0


# -------------------------------
# MIDI → CSV Feature
# -------------------------------

def build_tempo_map(mid: MidiFile) -> List[Tuple[int, int]]:
    """
    Create a global tempo map as a list of (abs_tick, tempo_us_per_beat),
    starting with default tempo unless overridden.
    """
    DEFAULT_TEMPO = 500_000  # 120 bpm
    tempo_changes: List[Tuple[int, int]] = [(0, DEFAULT_TEMPO)]

    merged = merge_tracks(mid.tracks)
    abs_tick = 0
    for msg in merged:
        abs_tick += msg.time  # delta in ticks
        if msg.type == "set_tempo":
            tempo_changes.append((abs_tick, msg.tempo))

    tempo_changes.sort(key=lambda x: x[0])

    # Deduplicate consecutive identical tempos
    dedup: List[Tuple[int, int]] = []
    for t, tempo in tempo_changes:
        if not dedup or dedup[-1][1] != tempo:
            dedup.append((t, tempo))
    return dedup


def ticks_to_seconds(abs_tick: int, tempo_map: List[Tuple[int, int]], ticks_per_beat: int) -> float:
    """
    Convert absolute ticks into seconds using the tempo map.
    """
    if abs_tick <= 0:
        return 0.0

    total_seconds = 0.0
    for idx, (t_i, tempo_i) in enumerate(tempo_map):
        t_next = tempo_map[idx + 1][0] if idx + 1 < len(tempo_map) else abs_tick
        if t_i >= abs_tick:
            break
        segment_end = min(t_next, abs_tick)
        delta_ticks = max(0, segment_end - t_i)
        if delta_ticks > 0:
            sec_per_tick = (tempo_i / 1_000_000.0) / ticks_per_beat
            total_seconds += delta_ticks * sec_per_tick
        if segment_end == abs_tick:
            break
    return total_seconds


def tempo_to_bpm(tempo_us_per_beat: int) -> float:
    if tempo_us_per_beat <= 0:
        return math.nan
    return 60_000_000.0 / tempo_us_per_beat


SIMPLE_CSV_HEADERS = [
    "Track",
    "Event Type",
    "Midi Note",
    "Note Name",
    "Velocity",
    "Duration (seconds)",
    "Time (seconds)",
]

_NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]


def _note_name(n: int) -> str:
    return f"{_NOTE_NAMES[n % 12]}{n // 12 - 1}"


def convert_midi_to_csv(input_path: Path, output_path: Path | None = None) -> Path:
    """
    Convert a MIDI file to a simplified CSV format matching the test schema:
    Track, Event Type, Midi Note, Note Name, Velocity, Duration (seconds), Time (seconds)
    - Track indices are renumbered to start at 0 for the first track that contains notes.
    - Only note_on events (with velocity>0) are emitted, paired with their note_off to compute duration.
    - Floats are formatted with 3 decimal places.
    """
    if output_path is None:
        output_path = input_path.with_suffix(".csv")

    mid = MidiFile(input_path)
    tpb = mid.ticks_per_beat
    tempo_map = build_tempo_map(mid)

    # Identify tracks that contain at least one note_on (velocity > 0)
    musical_track_ids = []
    for ti, track in enumerate(mid.tracks):
        if any(m.type == "note_on" and getattr(m, "velocity", 0) > 0 for m in track):
            musical_track_ids.append(ti)
    track_index_map = {ti: idx for idx, ti in enumerate(musical_track_ids)}

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SIMPLE_CSV_HEADERS)

        for ti, track in enumerate(mid.tracks):
            if ti not in track_index_map:
                continue
            fmt_ti = track_index_map[ti]
            abs_tick = 0
            # key: (channel, note) -> (start_tick, start_time_s, velocity)
            open_notes: dict[tuple[int, int], tuple[int, float, int]] = {}
            for msg in track:
                abs_tick += msg.time
                if msg.type == "note_on" and getattr(msg, "velocity", 0) > 0:
                    key = (getattr(msg, "channel", 0), msg.note)
                    start_s = ticks_to_seconds(abs_tick, tempo_map, tpb)
                    open_notes[key] = (abs_tick, start_s, msg.velocity)
                elif msg.type in ("note_off", "note_on") and getattr(msg, "note", None) is not None:
                    # Treat note_on with velocity 0 as note_off
                    vel = getattr(msg, "velocity", 0)
                    if msg.type == "note_on" and vel > 0:
                        continue
                    key = (getattr(msg, "channel", 0), msg.note)
                    if key in open_notes:
                        start_tick, start_s, start_vel = open_notes.pop(key)
                        end_s = ticks_to_seconds(abs_tick, tempo_map, tpb)
                        duration = end_s - start_s
                        writer.writerow([
                            fmt_ti,
                            "note_on",
                            msg.note,
                            _note_name(msg.note),
                            f"{start_vel/127.0:.3f}",
                            f"{duration:.3f}",
                            f"{start_s:.3f}",
                        ])

    return output_path


# -------------------------------
# CLI
# -------------------------------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="midihelper",
        description="Convert MIDI files to CSV and extract data from CSVs."
    )

    # Provide -help as an alias for -h/--help
    p.add_argument("-help", action="help", help="Show this help message and exit")

    sub = p.add_mutually_exclusive_group(required=True)

    # CSV Extract
    sub.add_argument(
        "-csv-extract",
        metavar="INPUT_CSV",
        help="Extract a column from a CSV to a bracketed list"
    )

    p.add_argument(
        "--column",
        help="Column name to extract (if omitted, prompts interactively)"
    )
    p.add_argument(
        "--sigfigs",
        type=int,
        help="Round numeric columns to N significant figures"
    )
    p.add_argument(
        "--track",
        help="If CSV has a 'track' header, filter rows where track==VALUE"
    )
    p.add_argument(
        "--output",
        type=Path,
        help="Write extracted output to this file (otherwise print to stdout)"
    )

    # MIDI → CSV
    sub.add_argument(
        "-midi-csv",
        nargs="+",
        metavar=("INPUT_MID", "OUTPUT_CSV"),
        help="Convert MIDI to CSV. Usage: midihelper -midi-csv input.mid [output.csv]"
    )

    return p


def main(argv: List[str] | None = None) -> None:
    parser = make_parser()
    args = parser.parse_args(argv)

    if args.csv_extract:
        # Determine non_interactive mode if flags were provided
        non_interactive = bool(args.column is not None)
        code = extract_csv_column(
            input_path=Path(args.csv_extract).expanduser().resolve(),
            column=args.column,
            sigfigs=args.sigfigs,
            track=args.track,
            output_path=args.output,
            non_interactive=non_interactive,
        )
        sys.exit(code)

    if args.midi_csv:
        if len(args.midi_csv) not in (1, 2):
            print("Error: -midi-csv expects 1 or 2 arguments: <input.mid> [output.csv]", file=sys.stderr)
            sys.exit(2)

        input_path = Path(args.midi_csv[0]).expanduser().resolve()
        if not input_path.exists():
            print(f"Error: input file not found: {input_path}", file=sys.stderr)
            sys.exit(1)

        output_path = None
        if len(args.midi_csv) == 2:
            output_path = Path(args.midi_csv[1]).expanduser().resolve()

        try:
            out = convert_midi_to_csv(input_path, output_path)
            print(str(out))
        except Exception as e:
            print(f"Conversion failed: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
