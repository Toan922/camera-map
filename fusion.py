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

Run `python fusion.py` to self-test on synthetic data (no camera needed).
"""
import numpy as np


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
    print('self-test PASS')
