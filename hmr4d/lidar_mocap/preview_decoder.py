"""Persistent preview decoding of verified SessionIndex rows, without FPS math."""

from fractions import Fraction
import threading


class PreviewDecoder:
    def __init__(self):
        self.states = {}
        self.locks = {}
        self.guard = threading.Lock()
        self.seek_count = 0
        self.delivered_frames = 0

    def diagnostics(self):
        with self.guard:
            return dict(seeks=self.seek_count, delivered_frames=self.delivered_frames)

    def frames(self, session, camera, indices):
        if not indices:
            return
        import av

        key = (session.session_id, camera)
        with self.guard:
            lock = self.locks.setdefault(key, threading.Lock())
        with lock:
            rows = [session.camera_rows(camera)[i] for i in indices]
            state = self.states.get(key)
            first = rows[0]["media_pts_us"]
            if state is None or first <= state["last_us"] or first - state["last_us"] > 4_000_000:
                if state is not None:
                    state["container"].close()
                container = av.open(str(session.video_paths[camera]))
                stream = container.streams.video[0]
                stream.thread_count = 2
                stream.thread_type = "AUTO"
                if stream.time_base is None or stream.time_base <= 0:
                    raise ValueError("Native preview lacks time base")
                start = Fraction(int(first), 1_000_000) / stream.time_base
                container.seek(int(start), stream=stream, backward=True, any_frame=False)
                with self.guard:
                    self.seek_count += 1
                state = dict(container=container, iterator=iter(container.decode(stream)), last_us=-1)
                self.states[key] = state
            for index, row in zip(indices, rows):
                target = row["media_pts_us"]
                for frame in state["iterator"]:
                    if frame.pts is None or frame.time_base is None:
                        raise ValueError("Native preview frame lacks PTS")
                    actual = round(frame.pts * frame.time_base * 1_000_000)
                    if actual < target - 2:
                        continue
                    if abs(actual - target) > 2:
                        raise ValueError(f"Native preview PTS mismatch: {actual} vs {target}")
                    state["last_us"] = actual
                    with self.guard:
                        self.delivered_frames += 1
                    yield index, frame.to_ndarray(format="rgb24")[:, :, ::-1].copy()
                    break
                else:
                    raise ValueError("Native preview exhausted before verified target frame")
