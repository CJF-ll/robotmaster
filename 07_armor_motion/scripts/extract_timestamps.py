#!/usr/bin/env python3
"""Extract ordered presentation timestamps; never estimate missing PTS from fps."""
import argparse
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import subprocess
import tempfile


def timestamps(document):
    frames = document.get("frames", [])
    if not frames:
        raise ValueError("no video frame timestamps")
    values = []
    for frame in frames:
        try:
            value = Decimal(frame["best_effort_timestamp_time"])
        except (KeyError, InvalidOperation, TypeError):
            raise ValueError("missing/invalid PTS; refusing fps substitution") from None
        if not value.is_finite() or (values and value <= values[-1]):
            raise ValueError("nonfinite, repeated or backwards PTS")
        values.append(value)
    streams = document.get("streams", [])
    declared = streams[0].get("nb_frames", "N/A") if streams else "N/A"
    if declared != "N/A" and int(declared) != len(values):
        raise ValueError("declared frame count differs from decoded timestamp count")
    return values[0], [value - values[0] for value in values]


def self_test():
    origin, result = timestamps({"frames": [{"best_effort_timestamp_time": v} for v in ["-0.1", "0", "0.04"]]})
    assert origin == Decimal("-0.1") and result == [Decimal(0), Decimal(".1"), Decimal(".14")]
    for data in [{"frames": []}, {"frames": [{}]},
                 {"frames": [{"best_effort_timestamp_time": "NaN"}]},
                 {"frames": [{"best_effort_timestamp_time": "0"}] * 2},
                 {"frames": [{"best_effort_timestamp_time": "0"}], "streams": [{"nb_frames": "2"}]}]:
        try:
            timestamps(data)
        except ValueError:
            continue
        raise AssertionError("invalid timestamp document accepted")
    print("timestamp extraction contracts passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.input or not args.output:
        parser.error("input video and new output CSV are required")
    destination = Path(args.output).absolute()
    if destination.exists():
        raise ValueError("output already exists")
    process = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                              "-show_entries", "frame=best_effort_timestamp_time:stream=nb_frames",
                              "-of", "json", str(Path(args.input).absolute())],
                             check=True, capture_output=True, text=True)
    origin, values = timestamps(json.loads(process.stdout))
    # Same-filesystem atomic publication with no overwrite, even if another
    # process creates the destination after the initial check.
    fd, temporary = tempfile.mkstemp(prefix=".timestamps-", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write("frame,time_seconds\n")
            for index, value in enumerate(values):
                output.write(f"{index},{value}\n")
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, destination)
    finally:
        os.unlink(temporary)
    print(f"frames={len(values)} source_origin_seconds={origin} output={destination}")


if __name__ == "__main__":
    main()
