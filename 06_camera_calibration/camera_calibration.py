"""Collect chessboard images, calibrate, and show live undistortion."""
import argparse
import json
import time
import threading
from pathlib import Path

import cv2
import numpy as np


def detect(frame, pattern):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Locate on a smaller image, then refine on the original pixel grid.
    scale = min(1.0, 960.0 / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    found, corners = cv2.findChessboardCorners(
        small, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    )
    if found:
        if scale < 1:
            corners = (corners + 0.5) / scale - 0.5
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        )
    return found, corners


def camera(args):
    source = int(args.camera) if args.camera.isdecimal() else args.camera
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    else:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000,
        ])
    if not cap.isOpened():
        cap.release()
        raise RuntimeError('Cannot open camera. Check --camera index or URL, phone server and network connection.')
    return cap


class LatestFrames:
    """Drain the network stream continuously; expose only the newest frame."""
    def __init__(self, cap):
        self.cap = cap
        self.condition = threading.Condition()
        self.stopping = threading.Event()
        self.frame = None
        self.sequence = 0
        self.consumed = 0
        self.finished = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stopping.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    break
                with self.condition:
                    self.frame = frame
                    self.sequence += 1
                    self.condition.notify_all()
        finally:
            self.cap.release()
            with self.condition:
                self.finished = True
                self.condition.notify_all()

    def read(self):
        with self.condition:
            self.condition.wait_for(lambda: self.sequence > self.consumed or self.finished, timeout=12)
            if self.sequence == self.consumed or self.frame is None:
                return False, None
            self.consumed = self.sequence
            return True, self.frame.copy()

    def release(self):
        self.stopping.set()
        self.thread.join(timeout=12)


def collect(args):
    folder = Path(args.images)
    folder.mkdir(parents=True, exist_ok=True)
    cap = LatestFrames(camera(args))
    count = len(list(folder.glob('*.png')))
    poses = []
    if args.auto:
        for path in sorted(folder.glob('*.png')):
            old = cv2.imread(str(path))
            if old is not None:
                found, pose = detect(old, (args.cols, args.rows))
                if found:
                    poses.append(pose.reshape(-1, 2))
    previous = None
    stable_since = None
    last_saved = 0.0
    reconnects = 0
    status = 'Move board, then hold still' if args.auto else 'SPACE save | Q quit'
    print('AUTO: hold each new pose still; Q: quit.' if args.auto else
          'SPACE: save a detected board; Q: quit. Move board between captures.')
    cv2.namedWindow('Collect chessboard images', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Collect chessboard images', 1200, 675)
    def distance(a, b):
        return min(float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))),
                   float(np.sqrt(np.mean(np.sum((a - b[::-1]) ** 2, axis=1)))))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.release()
                previous = None
                stable_since = None
                for attempt in range(1, 6):
                    reconnects += 1
                    print(f'Stream disconnected; reconnect attempt {attempt}/5', flush=True)
                    notice = np.zeros((360, 960, 3), np.uint8)
                    cv2.putText(notice, f'Saved: {count} | Reconnecting {attempt}/5',
                                (25, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
                    cv2.imshow('Collect chessboard images', notice)
                    if cv2.waitKey(500) & 0xff in (ord('q'), 27):
                        return
                    try:
                        cap = LatestFrames(camera(args))
                        ok, frame = cap.read()
                    except RuntimeError:
                        ok = False
                    if ok:
                        print('Stream reconnected.', flush=True)
                        break
                    cap.release()
                if not ok:
                    raise RuntimeError('Camera disconnected after 5 retries. Saved images are preserved.')
            found, corners = detect(frame, (args.cols, args.rows))
            now = time.monotonic()
            auto_save = False
            if args.auto:
                if not found:
                    previous = None
                    stable_since = None
                    status = 'Board not detected - show all corners'
                else:
                    pose = corners.reshape(-1, 2)
                    diagonal = float(np.hypot(frame.shape[1], frame.shape[0]))
                    if previous is None or distance(pose, previous) > 0.002 * diagonal:
                        stable_since = now
                        previous = pose.copy()
                    novel = all(distance(pose, old) > 0.025 * diagonal for old in poses)
                    stable = stable_since is not None and now - stable_since >= 0.8
                    auto_save = novel and stable and now - last_saved >= 2.0
                    status = ('Hold still' if not stable else
                              'New pose ready' if novel else 'Move / tilt / change distance')
            # Always show the preview at a fixed size; saved frame is untouched.
            display = frame.copy()
            if found:
                cv2.drawChessboardCorners(display, (args.cols, args.rows), corners, found)
            cv2.putText(display, f'Saved: {count} | Board: {found} | {status}',
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 200, 0) if found else (0, 0, 255), 2)
            cv2.putText(display, f'Live frame: {cap.consumed} | {time.strftime("%H:%M:%S")} | Q quit',
                        (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
            cv2.imshow('Collect chessboard images', display)
            key = cv2.waitKey(1) & 0xff
            if key == ord('q') or key == 27:
                break
            if found and (auto_save or (key == 32 and not args.auto)):
                # Never overwrite an earlier capture when resuming collection.
                index = count + 1
                while (folder / f'board_{index:03d}.png').exists():
                    index += 1
                path = folder / f'board_{index:03d}.png'
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f'Cannot save {path}')
                count += 1
                if args.auto:
                    poses.append(corners.reshape(-1, 2).copy())
                    last_saved = now
                print(f'Saved {path}, resolution {frame.shape[1]} x {frame.shape[0]}')
                if args.auto and count >= args.target:
                    print(f'Target reached: {count}. Collection complete.')
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def calibrate(args):
    pattern = (args.cols, args.rows)
    obj = np.zeros((args.cols * args.rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square_mm
    objects, points, names = [], [], []
    size = None
    for path in sorted(Path(args.images).glob('*.png')):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f'Skip unreadable: {path}')
            continue
        current_size = (frame.shape[1], frame.shape[0])
        if size is not None and current_size != size:
            raise RuntimeError('All calibration images must have the same resolution.')
        found, corners = detect(frame, pattern)
        if not found:
            print(f'Skip undetected: {path}')
            continue
        size = current_size
        objects.append(obj.copy())
        points.append(corners)
        names.append(str(path))
    if len(points) < 10:
        raise RuntimeError(f'Only {len(points)} usable images. Collect at least 10; aim for 20-30 diverse views.')
    rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(objects, points, size, None, None)
    screening = {'method': 'none', 'initial_images': len(points), 'initial_rms_px': float(rms), 'excluded': []}
    if args.robust:
        initial_errors = []
        for object_points, image_points, rotation, translation in zip(objects, points, rotations, translations):
            projected, _ = cv2.projectPoints(object_points, rotation, translation, matrix, distortion)
            initial_errors.append(float(np.sqrt(np.mean(np.sum((image_points - projected) ** 2, axis=2)))))
        median = float(np.median(initial_errors))
        sigma = float(1.4826 * np.median(np.abs(np.asarray(initial_errors) - median)))
        cutoff = median + 3 * sigma
        keep = [i for i, error in enumerate(initial_errors) if error <= cutoff]
        screening.update(method='single_pass_median_plus_3_scaled_MAD', cutoff_px=cutoff,
                         excluded=[{'image': names[i], 'initial_rms_px': error}
                                   for i, error in enumerate(initial_errors) if error > cutoff])
        if len(keep) < 10:
            raise RuntimeError('Too few images remain after robust screening. Recollect images.')
        if len(keep) != len(points):
            objects, points, names = ([values[i] for i in keep] for values in (objects, points, names))
            rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(objects, points, size, None, None)
        print('Screening:', screening)
    errors = []
    total_squared = 0.0
    total_points = 0
    for name, object_points, image_points, rotation, translation in zip(names, objects, points, rotations, translations):
        projected, _ = cv2.projectPoints(object_points, rotation, translation, matrix, distortion)
        squared = float(np.sum((image_points.reshape(-1, 2) - projected.reshape(-1, 2)) ** 2))
        error = float(np.sqrt(squared / len(image_points)))
        errors.append({'image': name, 'rms_px': error})
        total_squared += squared
        total_points += len(image_points)
    verified_rms = float(np.sqrt(total_squared / total_points))
    result = {
        'image_size': list(size), 'inner_corners': list(pattern), 'square_size_mm': args.square_mm,
        'camera_matrix': matrix.tolist(), 'distortion_coefficients': distortion.flatten().tolist(),
        'opencv_rms_px': float(rms), 'reprojection_rms_px': verified_rms,
        'passes_assignment_threshold': verified_rms < 0.5,
        'number_of_images': len(points), 'per_image_errors': errors,
        'screening': screening,
        'rotation_vectors': [v.flatten().tolist() for v in rotations],
        'translation_vectors_mm': [v.flatten().tolist() for v in translations]
    }
    output = Path(args.params)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Images: {len(points)} | RMS: {verified_rms:.6f} px | PASS: {verified_rms < 0.5}')
    print('Camera matrix:\n', matrix, '\nDistortion:', distortion.flatten())
    print(f'Saved {output}')
    print('Largest per-image errors:')
    for entry in sorted(errors, key=lambda e: e['rms_px'], reverse=True)[:5]:
        print(f"  {entry['rms_px']:.4f} px: {entry['image']}")
    if verified_rms >= 0.5:
        print('Threshold not met. Inspect blur, reflections, board flatness and coverage; recollect as needed.')


def demo(args):
    data = json.loads(Path(args.params).read_text(encoding='utf-8'))
    matrix = np.asarray(data['camera_matrix'], np.float64)
    distortion = np.asarray(data['distortion_coefficients'], np.float64)
    args.width, args.height = data['image_size']
    size = tuple(data['image_size'])
    new_matrix, _ = cv2.getOptimalNewCameraMatrix(matrix, distortion, size, 1, size)
    map_x, map_y = cv2.initUndistortRectifyMap(matrix, distortion, None, new_matrix, size, cv2.CV_32FC1)
    cap = LatestFrames(camera(args))
    writer = None
    window = 'Original / Undistorted - Q to quit'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1440, 405)
    started = time.monotonic()
    try:
        if args.record:
            path = Path(args.record)
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'),
                                     args.record_fps, (2 * size[0], size[1]))
            if not writer.isOpened():
                raise RuntimeError('Cannot open video writer. Try a different output path.')
            print('Recording at fixed FPS; use desktop screen recording if timing must match real time.')
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.release()
                for attempt in range(5):
                    print(f'Reconnecting camera ({attempt + 1}/5)...', flush=True)
                    cv2.waitKey(100)
                    try:
                        cap = LatestFrames(camera(args))
                        ok, frame = cap.read()
                        if ok:
                            break
                        cap.release()
                    except RuntimeError:
                        continue
                if not ok:
                    raise RuntimeError('Cannot read camera frames after reconnecting.')
            if (frame.shape[1], frame.shape[0]) != size:
                raise RuntimeError(f'Camera resolution differs from calibrated {size}; recalibrate at supported resolution.')
            corrected = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
            for view, label in [(frame, 'Original'), (corrected, 'Undistorted')]:
                cv2.putText(view, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(view, f"RMS: {data['reprojection_rms_px']:.4f} px", (15, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            combined = np.hstack([frame, corrected])
            if writer is not None:
                writer.write(combined)
            cv2.imshow(window, combined)
            if cv2.waitKey(1) & 0xff in (ord('q'), 27):
                break
            if args.duration and time.monotonic() - started >= args.duration:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['collect', 'calibrate', 'demo'])
    parser.add_argument('--camera', default='0', help='Camera index or network stream URL')
    parser.add_argument('--cols', type=int, default=9, help='Inner corner columns')
    parser.add_argument('--rows', type=int, default=6, help='Inner corner rows')
    parser.add_argument('--square-mm', type=float, default=25.0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--images', default='images')
    parser.add_argument('--params', default='output/calibration.json')
    parser.add_argument('--record', help='Optional MP4 path for demo')
    parser.add_argument('--record-fps', type=float, default=20.0)
    parser.add_argument('--duration', type=float, default=0, help='Demo duration in seconds; 0 runs until Q')
    parser.add_argument('--robust', action='store_true', help='One screening pass using median + 3 scaled MAD; retains all source files and logs exclusions')
    parser.add_argument('--auto', action='store_true', help='Save stable poses that differ from previous captures')
    parser.add_argument('--target', type=int, default=25, help='Total image target for automatic collection')
    args = parser.parse_args()
    if args.cols < 2 or args.rows < 2 or args.square_mm <= 0 or args.record_fps <= 0 or args.target < 1:
        parser.error('Board dimensions, square size and recording FPS must be positive and valid.')
    try:
        {'collect': collect, 'calibrate': calibrate, 'demo': demo}[args.mode](args)
    except (RuntimeError, OSError, ValueError, cv2.error) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
