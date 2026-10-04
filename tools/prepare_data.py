"""Build data/scene.js and data/xray.jpg for the demo from the xvr-data DeepFluoro files.

  python tools/prepare_data.py --xvr-data /path/to/xvr-data [--result parameters.pt]

--xvr-data is a local copy of https://huggingface.co/datasets/eigenvivek/xvr-data.
--result is an optional xvr registration output (parameters.pt with `final_pose`) for the same
X-ray, shown as the "xvr registration" preset. Needs the xvr environment (torch, diffdrr, xvr).

Frames: T = inv(pose @ REORIENT) maps CT points into the camera frame (source at the origin,
z along the central ray, detector at z = sdd), as in DiffDRR's perspective_projection.
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from PIL import Image
from scipy import ndimage
from skimage import measure
from torchio import LabelMap
from xvr.io import read_xray

SUBJECT, VIEW, LABELS = "subject01", "001", [1, 2, 3, 4, 7]  # labels used for registration in xvr
REORIENT = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
PA_TO_AP = np.diag([-1.0, -1.0, 1.0, 1.0])  # xvr-data commit de04f6c666 ("Swap DeepFluoro poses PA->AP")
XRAY_PX = 512


def inv(T):
    o = np.eye(4)
    o[:3, :3] = T[:3, :3].T
    o[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return o


def apply(T, x):
    return x @ T[:3, :3].T + T[:3, 3]


def packed(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def metrics(pose_gt, pose_pred, f):
    xvr = np.linalg.norm(apply(pose_pred, f) - apply(pose_gt, f), axis=1).mean()          # as in xvr
    tre = np.linalg.norm(apply(inv(pose_pred), f) - apply(inv(pose_gt), f), axis=1).mean()  # camera frame
    return float(xvr), float(tre)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xvr-data", type=Path, required=True)
    ap.add_argument("--result", type=Path)
    ap.add_argument("--pre-swap-poses", action="store_true",
                    help="the local DeepFluoro poses predate the PA->AP swap of 2026-07-20")
    args = ap.parse_args()
    root = args.xvr_data / "deepfluoro" / SUBJECT
    app = Path(__file__).resolve().parents[1]
    (app / "data").mkdir(exist_ok=True)

    # Anatomy mesh in DiffDRR world coordinates (diffdrr.data.canonicalize centres the volume).
    label = LabelMap(root / "mask.nii.gz")
    world = label.affine.copy()
    world[:3, 3] -= np.asarray(label.get_center())
    mask = np.isin(label.data[0].numpy(), LABELS)
    verts, faces, _, _ = measure.marching_cubes(ndimage.gaussian_filter(mask.astype(np.float32), 0.8), 0.5)
    poly = pv.PolyData(apply(world, verts), np.column_stack([np.full(len(faces), 3), faces]).ravel()).clean()
    mesh = poly.decimate(1 - 4000 / poly.n_cells)

    f = torch.load(root / "fiducials.pt", weights_only=False)[0].double().numpy()
    dist = np.abs(pv.PolyData(f).compute_implicit_distance(mesh)["implicit_distance"])
    print(f"fiducial-to-mesh distance: median {np.median(dist):.2f} mm")

    gt = torch.load(root / "xrays" / f"{VIEW}.pt", weights_only=False)
    pose_gt = gt["pose"][0].double().numpy()
    if args.pre_swap_poses:
        pose_gt = pose_gt @ PA_TO_AP

    # Detector geometry and image exactly as xvr reads the DICOM for registration.
    img, sdd, delx, dely, x0, y0, _ = read_xray(root / "xrays" / f"{VIEW}.dcm", crop=0, subtract_background=False, linearize=False)
    img = img[0, 0].numpy()
    lo, hi = np.percentile(img, [0.5, 99.5])
    img = np.clip((img - lo) / (hi - lo), 0, 1)
    Image.fromarray(np.rint(img * 255).astype(np.uint8)).resize((XRAY_PX, XRAY_PX), Image.LANCZOS).save(app / "data/xray.jpg", quality=88)
    detector = {"sdd": sdd, "delx": delx, "dely": dely, "x0": x0, "y0": y0, "height": img.shape[0], "width": img.shape[1]}

    scene = {"subject": SUBJECT, "view": VIEW, "fiducials": f.tolist(), "T_gt": inv(pose_gt @ REORIENT).tolist(),
             "pose_gt": pose_gt.tolist(), "detector": detector, "xray": "data/xray.jpg",
             "mesh": {"vertices": packed(mesh.points, "<f4"), "faces": packed(mesh.faces.reshape(-1, 4)[:, 1:], "<u4")}}
    if args.result:
        pose = torch.load(args.result, weights_only=False, map_location="cpu")["final_pose"][0].double().numpy()
        xvr, tre = metrics(pose_gt, pose, f)
        scene["result"] = {"T": inv(pose @ REORIENT).tolist(), "pose": pose.tolist(), "xvr_mtre": xvr, "camera_tre": tre}
        print(f"xvr registration of {VIEW}: xvr mTRE {xvr:.3f} mm, camera-frame TRE {tre:.3f} mm")
    (app / "data/scene.js").write_text("window.SCENE=" + json.dumps(scene, separators=(",", ":")) + ";")
    print("wrote data/scene.js", f"{(app / 'data/scene.js').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
