import argparse, json, math, os, re, sys, pathlib
import numpy as np

try:
    from scipy.spatial import cKDTree as KDTree
except Exception:
    KDTree = None

def load_vertices_from_obj(path, max_points=20000):
    """Parse 'v x y z' lines from an OBJ. Returns (N,3) float32 in meters/OBJ units."""
    verts = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):  # ignore vn, vt, faces
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                except ValueError:
                    pass
    V = np.asarray(verts, dtype=np.float32)
    if V.shape[0] == 0:
        raise RuntimeError(f"No vertices found in {path}")
    # Deduplicate exact duplicates if any
    V = np.unique(V, axis=0)
    # Subsample if huge
    if V.shape[0] > max_points:
        idx = np.random.RandomState(0).choice(V.shape[0], size=max_points, replace=False)
        V = V[idx]
    return V

def pca_axes(V):
    """Return eigenvectors sorted by descending variance: (evecs[0]=max, evecs[2]=min)."""
    C = V - V.mean(0, keepdims=True)
    cov = (C.T @ C) / max(1, C.shape[0] - 1)
    w, U = np.linalg.eigh(cov)  # w ascending, U columns are eigenvectors
    order = np.argsort(w)[::-1]
    evecs = U[:, order]  # columns: max, mid, min
    evals = w[order]
    return evecs, evals

def rodrigues_rotate(P, axis, angle_rad, center=None):
    """Rotate point set P (N,3) about 'axis' (3,) by angle around 'center' (3,) using Rodrigues."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    if center is None:
        center = np.zeros(3, dtype=np.float64)
    C = P - center
    kx, ky, kz = a
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], dtype=np.float64)
    R = np.eye(3) * math.cos(angle_rad) + math.sin(angle_rad) * K + (1 - math.cos(angle_rad)) * np.outer(a, a)
    return (C @ R.T) + center

def nearest_stats(A, B, eps):
    """
    Given point sets A,B (N,3), return:
      - mean Chamfer (symmetric)
      - match ratio: fraction of points within eps (both directions averaged)
    """
    if KDTree is not None:
        tA = KDTree(B); dA, _ = tA.query(A, k=1, workers=-1)
        tB = KDTree(A); dB, _ = tB.query(B, k=1, workers=-1)
    else:
        # vectorised brute-force (O(N^2)) — fine for a few thousand points
        dA = np.sqrt(((A[:,None,:]-B[None,:,:])**2).sum(-1)).min(1)
        dB = np.sqrt(((B[:,None,:]-A[None,:,:])**2).sum(-1)).min(1)
    chamfer = float(dA.mean() + dB.mean())
    match_ratio = 0.5 * (float((dA <= eps).mean()) + float((dB <= eps).mean()))
    return chamfer, match_ratio

def test_cyclic_symmetry(V, axis, n, size_eps):
    """
    Test Cn symmetry about 'axis' by rotating V by k*2π/n (k=1..n-1) and checking overlap.
    Returns best (mean_chamfer, mean_match_ratio).
    """
    ctr = V.mean(0)
    ch_list, mr_list = [], []
    for k in range(1, n):
        Vk = rodrigues_rotate(V, axis, 2.0*math.pi*k/n, center=ctr)
        ch, mr = nearest_stats(V, Vk, eps=size_eps)
        ch_list.append(ch); mr_list.append(mr)
    return float(np.mean(ch_list)), float(np.mean(mr_list))

def axis_to_quats_wxyz(axis, n):
    """Return list of wxyz quats for rotations {0, 2π/n, ..., 2π(n-1)/n} around axis."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    quats = []
    for k in range(n):
        ang = 2.0*math.pi*k/n
        w = math.cos(ang/2.0)
        s = math.sin(ang/2.0)
        quats.append([w, a[0]*s, a[1]*s, a[2]*s])
    return quats

def decide_symmetry(V, p_axes, diag):
    """
    Try candidate axes (largest-variance and smallest-variance PCA axes) for n in {4,2}.
    We accept the first passing (n,axis) by priority C4->C2 with thresholds:
       - match_ratio >= 0.90 and chamfer <= 0.01 * diag
    Returns dict or ("none", ...).
    """
    size_eps = 0.01 * diag   # 1% of bbox diagonal for nearest-neighbour tolerance
    thr_mr   = 0.90
    thr_ch   = 0.01 * diag

    candidates = [
        ("principal", p_axes[:,0]),  # long axis -> beam C2/C4 candidate
        ("minor",     p_axes[:,2]),  # normal axis -> plate C2 candidate
    ]
    for name, axis in candidates:
        # Try C4 first, then C2
        for n in (4, 2):
            ch, mr = test_cyclic_symmetry(V, axis, n, size_eps=size_eps)
            if mr >= thr_mr and ch <= thr_ch:
                return {"type":"Cn", "n": n, "axis": axis.tolist(), "stats": {"match_ratio": mr, "chamfer": ch}}
    return {"type":"none", "n": 1, "axis": [0.0,0.0,1.0], "stats": {"match_ratio": 0.0, "chamfer": float("inf")}}

def parse_id_and_name(path):
    """From '6_partial_assembly.obj' -> (6, 'partial_assembly')."""
    stem = pathlib.Path(path).stem
    m = re.match(r"^\s*(\d+)[\-_ ]?(.*)$", stem)
    if m:
        cid = int(m.group(1))
        name = m.group(2) if m.group(2) else f"class_{cid}"
        return cid, name
    return None, stem

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objs", nargs="+", help="List of OBJ files, or a directory to glob *.obj", required=True)
    ap.add_argument("--out", type=str, default="canonical_files/symmetry.json")
    ap.add_argument("--max_points", type=int, default=20000)
    args = ap.parse_args()

    # Expand directory to *.obj
    inputs = []
    for p in args.objs:
        if os.path.isdir(p):
            inputs.extend(sorted(str(x) for x in pathlib.Path(p).glob("*.obj")))
        else:
            inputs.append(p)
    if not inputs:
        print("No OBJ files found.", file=sys.stderr); sys.exit(1)

    classes = []
    for obj_path in inputs:
        cid, name = parse_id_and_name(obj_path)
        V = load_vertices_from_obj(obj_path, max_points=args.max_points)
        # center & scale stats
        ctr = V.mean(0)
        Vc  = V - ctr
        mins = V.min(axis=0); maxs = V.max(axis=0)
        bbox_diag = float(np.linalg.norm(maxs - mins))
        evecs, evals = pca_axes(Vc)

        sym = decide_symmetry(Vc, evecs, diag=bbox_diag)
        if sym["type"] == "Cn":
            quats = axis_to_quats_wxyz(sym["axis"], sym["n"])
        else:
            quats = [[1.0, 0.0, 0.0, 0.0]]

        entry = {
            "id": int(cid) if cid is not None else len(classes)+1,
            "name": name,
            "mesh": obj_path,
            "n_vertices": int(V.shape[0]),
            "principal_axis": evecs[:,0].tolist(),
            "minor_axis": evecs[:,2].tolist(),
            "symmetry": {
                "type": sym["type"],
                "axis": sym["axis"],
                "n": sym.get("n", 1),
                "discrete_quats_wxyz": quats,
                "diagnostics": sym["stats"]
            }
        }
        classes.append(entry)
        print(f"[{os.path.basename(obj_path)}] → {entry['symmetry']}")

    out = {
        "version": 1,
        "convention": "wxyz",
        "note": "Auto-derived by PCA+overlap test from OBJ meshes.",
        "classes": classes
    }
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ Wrote {out_path}")

if __name__ == "__main__":
    main()
