"""Original-video paired regression; detector targets are not manual ground truth."""
import csv
import math
import pathlib
import re
import subprocess
import sys
import tempfile


def rows(path):
    with path.open() as source:
        return list(csv.DictReader(source))


def run(args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)


def rms(values):
    return math.sqrt(sum(x * x for x in values) / len(values))


def jitter(records):
    history, differences = {}, []
    for r in records:
        key = r['id']
        if r['future_drawn'] != '1':
            history.pop(key, None)
            continue
        h = history.setdefault(key, [])
        frame = int(r['frame'])
        if h and h[-1][0] != frame - 1:
            h.clear()
        h.append((frame, float(r['fx']), float(r['fy'])))
        h[:] = h[-3:]
        if len(h) == 3:
            differences.append(math.hypot(h[2][1] - 2*h[1][1] + h[0][1],
                                          h[2][2] - 2*h[1][2] + h[0][2]))
    return differences


def main():
    evaluator, renderer, config = sys.argv[1:]
    text = pathlib.Path(config).read_text()
    pattern = r'(  center_forecast:\n    enabled: )(?:true|false)'
    assert len(re.findall(pattern, text)) == 1
    with tempfile.TemporaryDirectory(prefix='outpost-center-regression-') as temp:
        root = pathlib.Path(temp)
        scores, renders, confidence = [], [], []
        for name, enabled in [('base', 'false'), ('blend', 'true')]:
            cfg = root / (name + '.yaml')
            cfg.write_text(re.sub(pattern, lambda m: m[1] + enabled, text))
            score = root / (name + '.csv')
            run([evaluator, str(cfg), str(score)])
            scores.append(rows(score))
            confidence.append(rows(pathlib.Path(str(score) + '.confidence.csv')))
            output = root / name
            run([renderer, str(cfg), '--headless', '--no-video', '--output-dir', str(output)])
            renders.append(output)
        a, b = scores
        assert len(a) == len(b) == 678, 'issued prediction coverage changed'
        for before, after in zip(a, b):
            for key in ('frame', 'target_frame', 'id', 'status', 'origin_x', 'origin_y', 'observed_x', 'observed_y'):
                assert before[key] == after[key], ('changed scoring sample', key, before, after)
        matched = [(x, y) for x, y in zip(a, b) if x['status'] == 'scored']
        assert len(matched) == 557, 'scored coverage changed'
        groups = {
            'all': matched,
            'first': [(x, y) for x, y in matched if int(x['frame']) < 450],
            'second': [(x, y) for x, y in matched if int(x['frame']) >= 450],
            'moving': [(x, y) for x, y in matched if float(x['hold_error_px']) > 2],
            'low_displacement': [(x, y) for x, y in matched if float(x['hold_error_px']) <= 2],
        }
        for name, pairs in groups.items():
            old = [float(x['error_px']) for x, _ in pairs]
            new = [float(y['error_px']) for _, y in pairs]
            assert rms(new) <= rms(old) + .001, ('RMS regression', name)
            assert max(new) <= max(old) + .001, ('worst-case regression', name)
            if name == 'all':
                assert rms(new) < .985 * rms(old), 'paired improvement lost'
            print(name, 'n=', len(pairs), 'RMS', rms(old), '->', rms(new))
        for before, after in zip(*confidence):
            for key in ('future_var_x', 'future_var_y'):
                assert float(after[key]) >= float(before[key]), 'reduced uncertainty floor'
        for name in ('detection.csv', 'candidates.csv', 'slots.csv', 'tentative_observations.csv'):
            assert (renders[0] / name).read_bytes() == (renders[1] / name).read_bytes(), name
        tracking = [rows(p / 'tracking.csv') for p in renders]
        assert len(tracking[0]) == len(tracking[1])
        for before, after in zip(*tracking):
            for key in before:
                if key not in ('fx', 'fy'):
                    assert before[key] == after[key], ('tracking/visibility changed', key)
        for p in renders:
            frame_rows = rows(p / 'frames.csv')
            assert len(frame_rows) == 925
            assert sum(int(r['drawn_current']) == 0 for r in frame_rows) == 15
            assert sum(int(r['drawn_future']) for r in frame_rows) == 678
        old, new = map(jitter, tracking)
        assert len(old) == len(new) == 596
        assert rms(new) <= rms(old), 'display jitter increased'
        print('forecast second differences RMS', rms(old), '->', rms(new))
    print('paired center forecast video contracts passed')


if __name__ == '__main__':
    main()
