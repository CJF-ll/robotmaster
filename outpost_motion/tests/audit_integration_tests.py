"""Integration contracts; fixtures and outputs are confined to an owned temp directory."""
import csv
import pathlib
import subprocess
import sys
import tempfile


def run(args):
    subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def main():
    mode, executable, config = sys.argv[1:]
    with tempfile.TemporaryDirectory(prefix="outpost-audit-contract-") as directory:
        root = pathlib.Path(directory)
        if mode == "async":
            output = root / "probe"
            run([executable, config, str(output), "24"])
            rows = list(csv.DictReader((output / "summary.csv").open()))
            assert [r["capacity"] for r in rows] == ["0", "1", "2"]
            assert all(int(r["frames"]) == 24 for r in rows)
            assert all(float(r["max_corner_difference_px"]) < .001 for r in rows)
        elif mode == "center":
            original = pathlib.Path(config).read_text()
            marker = "shape_guard:\n    enabled: false"
            assert original.count(marker) == 1
            guarded = root / "guarded.yaml"
            guarded.write_text(original.replace(marker, "shape_guard:\n    enabled: true"))
            a, b = root / "baseline.csv", root / "guarded.csv"
            run([executable, config, str(a)])
            run([executable, str(guarded), str(b)])
            assert a.read_bytes() == b.read_bytes(), "shape permission changed center scoring"
            rows = list(csv.DictReader(a.open()))
            scored = [float(r["error_px"]) for r in rows if r["status"] == "scored"]
            assert len(scored) >= 557, "center forecast coverage regressed"
            assert sum(e * e for e in scored) / len(scored) < 36, "center forecast RMS exceeded 6 px"
            assert max(scored) < 21, "center forecast maximum error regression"
        else:
            raise ValueError("unknown contract mode")
    print(mode, "integration contracts passed")


if __name__ == "__main__":
    main()
