"""Fusion layer (Phase 4): persistent room map + cross-camera person association.

TSDFGrid — a truncated signed distance field over a fixed world-space box,
KinectFusion-style in pure numpy. Each incoming depth frame is integrated by
projecting every voxel into that camera: voxels observed in front of the
surface are carved free, voxels near it accumulate a running average of signed
distance and color. Hundreds of noisy monocular-depth frames from all cameras
average into one stable surface — the room — while anything transient (a person
walking through, even imperfectly masked) is carved back out by later
observations. Surface voxels are read back with extract() as a colored cloud.

fuse_people — greedy cross-camera clustering of person detections in world
space: detections from different cameras within merge_dist of each other are
the same person; two detections from the SAME camera never merge (they are two
people by construction).

PersonTracker — SORT in 3D on top of the fused clusters: one constant-velocity
Kalman filter per person (predict where they'll be, blend in each new
measurement weighted by confidence), Hungarian assignment to match this tick's
clusters to existing tracks, and track lifecycle (a track needs min_hits
matches to be born, survives max_age seconds of missed detections by coasting
on its predicted velocity, keeps one persistent ID for its whole life).

Run `python fusion.py` to self-test on synthetic data (no camera needed).
"""
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment  # ships with ultralytics
except ImportError:
    linear_sum_assignment = None


class TSDFGrid:

    def __init__(self, bounds, voxel=0.04, trunc=None, max_weight=64.0):
        """bounds: (x0, x1, y0, y1, z0, z1) in world meters. trunc defaults to
        3 voxels; max_weight caps the running average so the map can still adapt
        slowly if the room changes."""
        b = np.asarray(bounds, np.float32).reshape(3, 2)
        self.lo, hi = b[:, 0], b[:, 1]
        self.voxel = float(voxel)
        self.trunc = float(trunc) if trunc else 3.0 * self.voxel
        self.max_weight = float(max_weight)
        self.shape = np.maximum(np.ceil((hi - self.lo) / self.voxel), 1).astype(int)
        n = int(self.shape.prod())
        if n > 30_000_000:
            raise ValueError(f'{n / 1e6:.0f}M voxels — increase --voxel or shrink --bounds')
        axes = [self.lo[i] + (np.arange(self.shape[i], dtype=np.float32) + 0.5) * self.voxel
                for i in range(3)]
        gx, gy, gz = np.meshgrid(*axes, indexing='ij')
        self.pts = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)  # voxel centers, world
        self.tsdf = np.ones(n, np.float32)       # signed distance / trunc, in [-1, 1]
        self.weight = np.zeros(n, np.float32)    # observation count (capped)
        self.color = np.zeros((n, 3), np.float32)

    def integrate(self, depth, rgb, fx, fy, cx, cy, R=None, t=None):
        """Fuse one depth frame (meters, 0 = invalid/masked) + its RGB image.
        R, t: camera->world extrinsics (None = camera frame IS the world frame)."""
        H, W = depth.shape
        if R is None:
            pc = self.pts
        else:  # world -> camera: p_c = R^T (p_w - t), as row vectors: (p - t) @ R
            pc = (self.pts - np.asarray(t, np.float32)) @ np.asarray(R, np.float32)
        z = pc[:, 2]
        idx = np.nonzero(z > 0.05)[0]  # voxels in front of the camera
        x, y, z = pc[idx, 0], pc[idx, 1], pc[idx, 2]
        u = np.rint(x / z * fx + cx).astype(np.int32)
        v = np.rint(y / z * fy + cy).astype(np.int32)
        m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx, u, v, z = idx[m], u[m], v[m], z[m]
        d = depth[v, u]
        m = d > 0.05  # skip invalid / person-masked pixels
        idx, u, v, z, d = idx[m], u[m], v[m], z[m], d[m]
        sdf = d - z  # + : voxel in front of surface (free), - : behind (occluded)
        m = sdf > -self.trunc  # occluded voxels carry no information
        idx, u, v, sdf = idx[m], u[m], v[m], sdf[m]
        obs = np.minimum(sdf / self.trunc, 1.0).astype(np.float32)

        w = self.weight[idx]
        self.tsdf[idx] = (self.tsdf[idx] * w + obs) / (w + 1.0)
        self.weight[idx] = np.minimum(w + 1.0, self.max_weight)

        near = np.abs(sdf) < self.trunc  # only actual surface observations color a voxel
        ci = idx[near]
        cw = np.minimum(self.weight[ci], 16.0)[:, None]  # colors adapt faster than geometry
        self.color[ci] = (self.color[ci] * cw + rgb[v[near], u[near]]) / (cw + 1.0)

    def extract(self, min_weight=4.0, iso=0.4):
        """Surface voxels as (points (N,3) float32, colors (N,3) uint8):
        seen at least min_weight times and within iso*trunc of the surface."""
        m = (self.weight >= min_weight) & (np.abs(self.tsdf) < iso)
        return self.pts[m], self.color[m].astype(np.uint8)


def fuse_people(dets, merge_dist=0.75):
    """Cluster person detections across cameras by world-space proximity.
    dets: iterable of (cam_id, {'center': (3,) world meters, ...}).
    Returns a list of clusters, each a list of (cam_id, det)."""
    clusters = []
    for cam, det in dets:
        best, best_d = None, merge_dist
        for cl in clusters:
            if any(c == cam for c, _ in cl):
                continue
            mean = np.mean([d['center'] for _, d in cl], axis=0)
            dist = float(np.linalg.norm(mean - det['center']))
            if dist < best_d:
                best, best_d = cl, dist
        if best is not None:
            best.append((cam, det))
        else:
            clusters.append([(cam, det)])
    return clusters


def _assign(cost, gate):
    """Match rows (tracks) to columns (measurements): globally optimal via the
    Hungarian algorithm when scipy is present, greedy nearest-pair otherwise.
    Returns (row, col) pairs with cost below gate."""
    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(cost)
        return [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] < gate]
    pairs, c = [], cost.astype(np.float64).copy()
    while c.size and c.min() < gate:
        r, col = np.unravel_index(np.argmin(c), c.shape)
        pairs.append((int(r), int(col)))
        c[r, :] = np.inf
        c[:, col] = np.inf
    return pairs


class Track:
    """One person: constant-velocity Kalman state [px py pz vx vy vz] (meters,
    meters/second, world frame)."""

    def __init__(self, tid, m, now, meas_std, accel_std):
        self.id = tid
        self.x = np.concatenate([np.asarray(m['center'], np.float64), np.zeros(3)])
        self.P = np.diag([meas_std ** 2] * 3 + [1.0] * 3)  # velocity starts unknown
        self.height = float(m['height'])
        self.n_cams = int(m.get('n_cams', 1))
        self.meas_var = meas_std ** 2
        self.accel_var = accel_std ** 2
        self.hits = 1
        self.t_updated = now

    def predict(self, dt):
        """Advance by physics: position += velocity * dt, uncertainty grows by
        how much a person could have accelerated meanwhile."""
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3)
        I3 = np.eye(3)
        Q = self.accel_var * np.block([[dt ** 4 / 4 * I3, dt ** 3 / 2 * I3],
                                       [dt ** 3 / 2 * I3, dt ** 2 * I3]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, m, now):
        """Blend the prediction with a measurement, weighted by the Kalman gain
        (trust in prediction vs. trust in measurement)."""
        y = np.asarray(m['center'], np.float64) - self.x[:3]  # innovation
        S = self.P[:3, :3] + self.meas_var * np.eye(3)
        K = self.P[:, :3] @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = self.P - K @ self.P[:3, :]
        self.height = 0.8 * self.height + 0.2 * float(m['height'])
        self.n_cams = int(m.get('n_cams', 1))
        self.hits += 1
        self.t_updated = now


class PersonTracker:
    """SORT in 3D over fused person clusters: predict all tracks, match tracks
    to measurements (Hungarian, gated), update matched, spawn tracks for
    unmatched measurements, drop tracks unseen for max_age seconds. Tracks are
    reported only after min_hits matches (kills one-frame false positives) and
    keep coasting on prediction through detection dropouts."""

    def __init__(self, meas_std=0.15, accel_std=2.0, gate=1.2, max_age=1.5, min_hits=3):
        self.meas_std, self.accel_std = meas_std, accel_std
        self.gate, self.max_age, self.min_hits = gate, max_age, min_hits
        self.tracks = []
        self.t_last = None
        self._next_id = 1

    def step(self, measurements, now):
        """measurements: list of {'center': (3,), 'height': m, 'n_cams': int}.
        Returns live tracks as dicts: id, center, vel, height, n_cams, matched."""
        dt = 0.0 if self.t_last is None else min(max(now - self.t_last, 0.0), 0.5)
        self.t_last = now
        for tr in self.tracks:
            tr.predict(dt)

        pairs = []
        if self.tracks and measurements:
            cost = np.array([[float(np.linalg.norm(tr.x[:3] - m['center']))
                              for m in measurements] for tr in self.tracks])
            pairs = _assign(cost, self.gate)
        for r, c in pairs:
            self.tracks[r].update(measurements[c], now)
        matched_m = {c for _, c in pairs}
        for j, m in enumerate(measurements):
            if j not in matched_m:
                self.tracks.append(Track(self._next_id, m, now, self.meas_std, self.accel_std))
                self._next_id += 1
        self.tracks = [tr for tr in self.tracks if now - tr.t_updated < self.max_age]

        return [dict(id=tr.id, center=tr.x[:3].copy(), vel=tr.x[3:].copy(),
                     height=tr.height, n_cams=tr.n_cams, matched=(tr.t_updated == now))
                for tr in self.tracks if tr.hits >= self.min_hits]


# --- synthetic self-test: two cameras observe a wall, a transient blob is carved ---

def _rot_y(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)


def _plane_depth(plane_z, W, H, fx, fy, cx, cy, R, t, noise, rng):
    """Analytic depth image of the plane z=plane_z (world) from camera (R, t)."""
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    dirs = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us, np.float64)], -1)
    depth = (plane_z - t[2]) / (dirs @ R[2])  # ray length s == camera-frame z here
    return (depth + noise * rng.standard_normal(depth.shape)).astype(np.float32)


if __name__ == '__main__':
    import time

    rng = np.random.default_rng(0)
    W, H, f = 320, 240, 300.0
    cx, cy = W / 2, H / 2
    grid = TSDFGrid((-1.5, 1.5, -1.2, 1.2, 0.2, 2.5), voxel=0.02)
    print(f'grid {tuple(grid.shape)} = {grid.pts.shape[0] / 1e6:.1f}M voxels')
    cams = [(np.eye(3, dtype=np.float32), np.zeros(3, np.float32)),
            (_rot_y(-15), np.array([0.5, 0.0, 0.0], np.float32))]
    rgb = np.full((H, W, 3), (200, 50, 50), np.uint8)

    # transient blob (wall at 1 m) seen 3 times — later free-space views must carve it
    R0, t0 = cams[0]
    for _ in range(3):
        grid.integrate(_plane_depth(1.0, W, H, f, f, cx, cy, R0, t0, 0.02, rng),
                       rgb, f, f, cx, cy, R0, t0)

    t_start = time.time()
    for _ in range(15):  # the real wall at 2 m, from both cameras, 2 cm depth noise
        for R, t in cams:
            grid.integrate(_plane_depth(2.0, W, H, f, f, cx, cy, R, t, 0.02, rng),
                           rgb, f, f, cx, cy, R, t)
    ms = (time.time() - t_start) / 30 * 1000
    pts, cols = grid.extract()
    wall = pts[np.abs(pts[:, 2] - 2.0) < 0.1]
    ghost = pts[np.abs(pts[:, 2] - 1.0) < 0.15]
    err = float(np.mean(np.abs(wall[:, 2] - 2.0))) if len(wall) else np.inf
    col_err = np.abs(np.median(cols, axis=0).astype(float) - (200, 50, 50)).max()
    print(f'integrate {ms:.0f} ms/frame | {len(pts)} surface pts | '
          f'wall err {err * 100:.1f} cm | ghost pts {len(ghost)} | color err {col_err:.0f}')
    assert len(wall) > 5000, f'too few wall points: {len(wall)}'
    assert err < 0.025, f'wall error {err:.3f} m exceeds one voxel'
    assert len(ghost) < 20, f'transient blob not carved: {len(ghost)} pts'
    assert col_err < 30, f'surface colors off by {col_err:.0f}'

    clusters = fuse_people([(0, {'center': np.array([0.0, 0.0, 1.0])}),
                            (1, {'center': np.array([0.2, 0.0, 1.0])}),
                            (1, {'center': np.array([3.0, 0.0, 1.0])})])
    assert len(clusters) == 2 and len(clusters[0]) == 2, clusters

    # tracker: two people walk opposite ways 1.5 m apart at 0.8 m/s, noisy
    # measurements at 20 Hz; walker A's detections drop out for 0.6 s mid-walk —
    # the track must coast through on its Kalman prediction, same ID throughout
    tracker = PersonTracker()
    ids = set()
    coast_err = 0.0
    for i in range(100):  # 5 s
        tk = i * 0.05
        pa = np.array([0.8 * tk - 2.0, 0.0, 0.9])
        pb = np.array([-0.8 * tk + 2.0, 1.5, 0.9])
        meas = [dict(center=pb + 0.08 * rng.standard_normal(3), height=1.8, n_cams=2)]
        if not 2.0 < tk < 2.6:  # A's dropout window
            meas.append(dict(center=pa + 0.08 * rng.standard_normal(3), height=1.7, n_cams=2))
        tracks = tracker.step(meas, tk)
        ids |= {tr['id'] for tr in tracks}
        if 2.0 < tk < 2.6:
            a = [tr for tr in tracks if abs(tr['center'][1]) < 0.7]  # A walks the y=0 line
            assert a, f'track A died during dropout at t={tk:.2f}'
            coast_err = max(coast_err, float(np.linalg.norm(a[0]['center'] - pa)))
    a = [tr for tr in tracks if abs(tr['center'][1]) < 0.7][0]
    final_err = float(np.linalg.norm(a['center'] - pa))
    vel_err = abs(float(a['vel'][0]) - 0.8)
    print(f'tracker: {len(ids)} IDs for 2 people | coast err {coast_err:.2f} m | '
          f'final err {final_err:.2f} m | vel err {vel_err:.2f} m/s')
    assert len(ids) == 2, f'expected 2 track IDs, got {sorted(ids)}'
    assert coast_err < 0.6, f'prediction drifted {coast_err:.2f} m during dropout'
    assert final_err < 0.3 and vel_err < 0.25
    print('self-test PASS')
