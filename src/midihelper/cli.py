from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
from mido import MidiFile, merge_tracks


# -------------------------------
# Utilities
# -------------------------------

def _print_bracketed(values: Iterable) -> str:
    """
    Render a list-like sequence to a single bracketed line:
    [v1, v2, v3]
    Strings are printed as-is; numbers use their existing formatting.
    """
    out_elems = []
    for v in values:
        if isinstance(v, str):
            out_elems.append(v)
        else:
            out_elems.append(str(v))
    return "[" + ", ".join(out_elems) + "]"


def _format_sigfig(x, sigfigs: int):
    """Format a number to N significant figures."""
    try:
        # pandas may give NaN/None; keep them as-is (string) to preserve count
        if x is None or (isinstance(x, float) and (pd.isna(x))):
            return ""
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
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        return 1

    # Dynamically list headers
    headers = list(df.columns)
    if not headers:
        print("Error: CSV has no headers.", file=sys.stderr)
        return 1

    # Choose column interactively if not provided
    if column is None:
        print("Available headers:")
        for i, h in enumerate(headers, start=1):
            print(f"  {i}. {h}")
        chosen = input("Select a column (name or number): ").strip()
        if chosen.isdigit():
            idx = int(chosen) - 1
            if idx < 0 or idx >= len(headers):
                print("Invalid choice.", file=sys.stderr)
                return 1
            column = headers[idx]
        else:
            if chosen not in headers:
                print(f"Error: '{chosen}' not found in headers.", file=sys.stderr)
                return 1
            column = chosen
    else:
        if column not in headers:
            print(f"Error: '{column}' not found in headers.", file=sys.stderr)
            return 1

    # Optional track filter (only if 'track' column exists)
    if "track" in headers:
        if track is None and not non_interactive:
            ans = input("Filter by 'track'? (y/n): ").strip().lower()
            if ans == "y":
                track = input("Enter track value (compared as string): ").strip()

        if track is not None:
            df = df[df["track"].astype(str) == str(track)]

    series = df[column]

    # Sig figs prompt for numeric columns
    if series.dtype.kind in "fc" and sigfigs is None and not non_interactive:
        raw = input("How many sig figs to round (Enter to skip): ").strip()
        if raw:
            try:
                sigfigs = int(raw)
            except ValueError:
                print("Invalid sig figs; ignoring.", file=sys.stderr)

    if sigfigs is not None and series.dtype.kind in "fc":
        values = [ _format_sigfig(v, sigfigs) for v in series.tolist() ]
    else:
        values = series.tolist()

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


CSV_FIELDS = [
    "track",
    "abs_tick",
    "time_s",
    "msg_type",
    "channel",
    "note",
    "velocity",
    "control",
    "value",
    "program",
    "tempo_us_per_beat",
    "bpm",
    "meta_type",
    "text",
]

def convert_midi_to_csv(input_path: Path, output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = input_path.with_suffix(".csv")

    mid = MidiFile(input_path)
    tpb = mid.ticks_per_beat
    tempo_map = build_tempo_map(mid)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for ti, track in enumerate(mid.tracks):
            abs_tick = 0
            for msg in track:
                abs_tick += msg.time  # per-track absolute time
                row = {field: "" for field in CSV_FIELDS}
                row["track"] = ti
                row["abs_tick"] = abs_tick
                row["time_s"] = f"{ticks_to_seconds(abs_tick, tempo_map, tpb):.9f}"
                row["msg_type"] = msg.type

                if hasattr(msg, "channel"):
                    row["channel"] = msg.channel

                if msg.type in ("note_on", "note_off"):
                    row["note"] = getattr(msg, "note", "")
                    row["velocity"] = getattr(msg, "velocity", "")

                if msg.type == "control_change":
                    row["control"] = getattr(msg, "control", "")
                    row["value"] = getattr(msg, "value", "")

                if msg.type == "program_change":
                    row["program"] = getattr(msg, "program", "")

                if msg.type == "set_tempo":
                    tempo_us = getattr(msg, "tempo", "")
                    row["tempo_us_per_beat"] = tempo_us
                    if tempo_us != "":
                        row["bpm"] = f"{tempo_to_bpm(tempo_us):.6f}"

                if msg.is_meta:
                    row["meta_type"] = msg.type
                    # Common text-like fields
                    txt = ""
                    for attr in ("text", "name", "copyright", "track_name",
                                 "lyrics", "marker", "cue"):
                        if hasattr(msg, attr):
                            txt = getattr(msg, attr)
                            break
                    row["text"] = txt

                writer.writerow(row)

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