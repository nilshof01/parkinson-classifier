"""Entrypoint for the DrivenData execution container (copied into submission.zip
as main.py). Reads test NIfTIs from data/niftis/, writes submission.csv.
Dependencies: torch, nibabel, numpy, scipy, pandas only."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy import ndimage

HERE = Path(__file__).resolve().parent
DATA = Path("data")
TARGET_MM = 2.0
FRAME = (96, 112, 96)
CROP = (slice(26, 70), slice(48, 86), slice(36, 62))
SMOOTH_MM, ERODE, HEAD_MM, MAX_SHIFT = 5.0, 1, 140.0, 14
FALLBACK_P = 0.548


def otsu(values, bins=256):
    hist, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mu0 = np.divide(m0, w0, out=np.zeros_like(m0), where=w0 > 0)
    mu1 = np.divide(m0[-1] - m0, w1, out=np.zeros_like(m0), where=w1 > 0)
    return centers[int(np.argmax(w0 * w1 * (mu0 - mu1) ** 2))]


def load_resampled(path, affine_correct=False):
    img = nib.load(path)
    if affine_correct:
        from nibabel.processing import resample_to_output
        img = resample_to_output(img, voxel_sizes=TARGET_MM, order=1, cval=0.0)
        vol = np.asarray(img.dataobj, dtype=np.float32)
        return vol[..., 0] if vol.ndim == 4 else vol
    img = nib.as_closest_canonical(img)
    vol = np.asarray(img.dataobj, dtype=np.float32)
    if vol.ndim == 4:
        vol = vol[..., 0]
    zooms = np.array(img.header.get_zooms()[:3], dtype=float)
    return ndimage.zoom(vol, zooms / TARGET_MM, order=1)


def preprocess(path, affine_correct=False, head_restrict=True):
    vol = load_resampled(path, affine_correct)

    smooth = ndimage.gaussian_filter(vol, SMOOTH_MM / TARGET_MM)
    mask = smooth > otsu(smooth[smooth > 0])
    labeled, n = ndimage.label(mask)
    if n == 0:
        raise ValueError("empty mask")
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    mask = ndimage.binary_erosion(
        ndimage.binary_fill_holes(labeled == (int(np.argmax(sizes)) + 1)),
        iterations=ERODE)
    head = mask.copy()
    if head_restrict:
        z_top = int(np.where(mask.any(axis=(0, 1)))[0].max())
        head[:, :, : max(0, z_top - int(HEAD_MM / TARGET_MM))] = False
    centroid = np.array(ndimage.center_of_mass(head))

    frame = np.zeros(FRAME, dtype=np.float32)
    start = np.round(centroid).astype(int) - np.array(FRAME) // 2
    src_lo = np.maximum(start, 0)
    src_hi = np.minimum(start + FRAME, vol.shape)
    dst_lo = src_lo - start
    dst_hi = dst_lo + (src_hi - src_lo)
    masked = vol * mask
    frame[dst_lo[0]:dst_hi[0], dst_lo[1]:dst_hi[1], dst_lo[2]:dst_hi[2]] = \
        masked[src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]
    m = frame[frame > 0].mean()
    if not m > 0:
        raise ValueError("empty frame")
    return frame / m


def align(frame, reference):
    for _ in range(2):
        f = np.fft.rfftn(frame); r = np.fft.rfftn(reference)
        cross = r * np.conj(f); cross /= np.abs(cross) + 1e-9
        corr = np.fft.fftshift(np.fft.irfftn(cross, s=frame.shape, axes=(0, 1, 2)))
        c = np.array([s // 2 for s in frame.shape]); m = MAX_SHIFT
        win = corr[c[0]-m:c[0]+m+1, c[1]-m:c[1]+m+1, c[2]-m:c[2]+m+1]
        shift = tuple(int(p - m) for p in np.unravel_index(np.argmax(win), win.shape))
        if any(shift):
            frame = ndimage.shift(frame, shift, order=1, cval=0.0)
    return frame


def resize224(m):
    t = torch.from_numpy(np.ascontiguousarray(m, dtype=np.float32))[None, None]
    return torch.nn.functional.interpolate(
        t, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]


def std2d(t):
    return (t - t.mean()) / (t.std() + 1e-6)


def view_mip(vol):
    return torch.stack([std2d(resize224(vol.max(axis=a))) for a in (2, 1, 0)])


def view_mipasym(vol):
    axial, coronal = vol.max(axis=2), vol.max(axis=1)
    asym = axial - axial[::-1]
    a = resize224(asym)
    return torch.stack([std2d(resize224(axial)), std2d(resize224(coronal)),
                        a / (a.abs().std() + 1e-6)])


def view_volume(vol):
    t = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32))
    inside = t > 0
    if inside.any():
        t = (t - t[inside].mean()) / (t[inside].std() + 1e-6)
    return t[None]


VIEWS = {"mip": view_mip, "mipasym": view_mipasym, "volume3d": view_volume}
SHIFTS = [(0, 0, 0)] + [tuple(s if a == ax else 0 for a in range(3))
                        for ax in range(3) for s in (-1, 1)]


def tta_variants(vol, mode):
    vols = [vol] if mode == "flip" else [
        ndimage.shift(vol, s, order=1, cval=0.0) if any(s) else vol for s in SHIFTS]
    out = []
    for v in vols:
        out += [v, v[::-1].copy()]
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = json.loads((HERE / "assets" / "spec.json").read_text())
    mean_frame = np.load(HERE / "assets" / "mean_frame.npy")
    members = []
    for mem in spec["members"]:
        nets = [torch.jit.load(str(HERE / p), map_location=device).eval()
                for p in mem["files"]]
        members.append((mem["view"], mem["input"], mem["tta"], nets))
    temp, (clip_lo, clip_hi) = spec["temperature"], spec["clip"]

    template = np.load(HERE / "assets" / "template_crop.npy").ravel()

    def template_corr(crop):
        return float(np.corrcoef(crop.ravel(), template)[0, 1])

    fmt = pd.read_csv(DATA / "submission_format.csv")
    preds = []
    n_retry = 0
    for i, uid in enumerate(fmt["uid"].astype(str)):
        try:
            path = DATA / "niftis" / f"{uid}.nii.gz"
            frame = align(preprocess(path), mean_frame)
            # placement retry: a crop that resembles no striatal anatomy may be
            # misplaced; try alternative preprocessing paths and keep the best
            if template_corr(frame[CROP]) < 0.15:
                cands = [(template_corr(frame[CROP]), frame)]
                for kw in ({"affine_correct": True}, {"head_restrict": False}):
                    try:
                        f2 = align(preprocess(path, **kw), mean_frame)
                        cands.append((template_corr(f2[CROP]), f2))
                    except Exception:
                        pass
                best = max(cands, key=lambda t: t[0])
                if best[1] is not frame:
                    n_retry += 1
                frame = best[1]
            crop = frame[CROP]
            pooled = frame.reshape(48, 2, 56, 2, 48, 2).mean(axis=(1, 3, 5))
            with torch.no_grad():
                member_ps = []
                for view, inp, tta, nets in members:
                    if inp == "fusion":
                        # flip TTA applied to both streams together
                        cs = torch.stack([view_volume(crop),
                                          view_volume(crop[::-1].copy())]).to(device)
                        fs = torch.stack([view_volume(pooled),
                                          view_volume(pooled[::-1].copy())]).to(device)
                        ps = [torch.sigmoid(net(cs, fs).float()).mean().item()
                              for net in nets]
                    else:
                        vol = crop if inp == "crop" else pooled
                        xs = torch.stack([VIEWS[view](v)
                                          for v in tta_variants(vol, tta)]).to(device)
                        ps = [torch.sigmoid(net(xs).float()).mean().item()
                              for net in nets]
                    member_ps.append(float(np.mean(ps)))
            p = float(np.mean(member_ps))
            lo = np.log(max(p, 1e-6) / max(1 - p, 1e-6))
            p = 1 / (1 + np.exp(-lo / temp))
        except Exception as e:
            # log no test-data details (uid, file content) per competition rules
            print(f"fallback used for one scan ({type(e).__name__})")
            p = FALLBACK_P
        preds.append(min(max(p, clip_lo), clip_hi))
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(fmt)} scans done")

    out = pd.DataFrame({"uid": fmt["uid"], "is_pathologic": preds})
    out.to_csv("submission.csv", index=False)
    print(f"wrote submission.csv ({len(out)} rows); placement retries: {n_retry}")


if __name__ == "__main__":
    main()
