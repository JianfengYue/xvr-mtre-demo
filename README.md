# xvr mTRE: where is the error measured?

xvr computes mTRE as `pred_pose(fiducials) - true_pose(fiducials)` ([`experiments/evaluate.py`, line 49](https://github.com/eigenvivek/xvr/blob/v0.1.0/experiments/evaluate.py#L49)). In DiffDRR, `pose` maps camera coordinates to CT coordinates, so this reads the CT fiducials as camera-frame points, which lie around the X-ray source. Rotation errors are then scaled by the source-to-anatomy distance (about 840 mm here) instead of the size of the anatomy (about 80 mm).

**[Interactive demo](https://jianfengyue.github.io/xvr-mtre-demo/)** (or download the repository and open `index.html`).

![Demo](preview.png)

DeepFluoro subject01, X-ray 001:

| Error | At the anatomy, `pose.inverse()(f)` | xvr mTRE, `pose(f)` |
|---|---:|---:|
| Rotate the anatomy 1° about its centre, out of plane | 0.90 mm | 14.55 mm |
| Rotate the anatomy 1° about its centre, in plane | 1.20 mm | 1.30 mm |
| Translate the anatomy 1 mm | 1.00 mm | 1.00 mm |
| An xvr single-view registration of this X-ray | 0.49 mm | 2.90 mm |

Suggested change, using the same direction as `DRR.perspective_projection`:

```diff
-        mtre = (pred_pose(self.fiducials) - true_pose(self.fiducials)).norm(dim=-1).mean(dim=-1)
+        mtre = (pred_pose.inverse()(self.fiducials) - true_pose.inverse()(self.fiducials)).norm(dim=-1).mean(dim=-1)
```

Data: DeepFluoro (Grupp et al. 2020, CC BY-NC 4.0) from [xvr-data](https://huggingface.co/datasets/eigenvivek/xvr-data). The ground-truth pose is the current `deepfluoro/subject01/xrays/001.pt`. The mesh is extracted from mask labels 1, 2, 3, 4 and 7 and simplified, and the X-ray is rescaled for display. `tools/prepare_data.py` regenerates `data/` from a local copy of xvr-data. The code is under the MIT license; the files in `data/` remain under CC BY-NC 4.0.
