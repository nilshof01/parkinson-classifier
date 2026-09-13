"""NIfTI → aligned striatal crop.
Self-contained — no dependency on the training package.
Adapted from submission_main.py and analysis/preprocessor.py."""

import numpy as np
import nibabel as nib
from scipy import ndimage

TARGET_MM = 2.0
FRAME_SHAPE = (96, 112, 96)
SMOOTH_MM = 5.0
ERODE = 1
HEAD_MM = 140.0
MAX_SHIFT = 14

# Base crop slices for the 96×112×96 aligned frame (m0, 44×38×26 @ 2 mm)
BASE_CROP = (slice(26, 70), slice(48, 86), slice(36, 62))


def crop_slices(margin: int):
    """Return crop slices expanded by `margin` voxels on each side."""
    return tuple(slice(s.start - margin, s.stop + margin) for s in BASE_CROP)


def otsu(values, bins=256):
    hist, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
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


def extract_frame(vol, head_restrict=True):
    """Segment head, find centroid, extract the fixed-size frame."""
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

    frame = np.zeros(FRAME_SHAPE, dtype=np.float32)
    start = np.round(centroid).astype(int) - np.array(FRAME_SHAPE) // 2
    src_lo = np.maximum(start, 0)
    src_hi = np.minimum(start + FRAME_SHAPE, vol.shape)
    dst_lo = src_lo - start
    dst_hi = dst_lo + (src_hi - src_lo)
    masked = vol * mask
    frame[dst_lo[0]:dst_hi[0], dst_lo[1]:dst_hi[1], dst_lo[2]:dst_hi[2]] = \
        masked[src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]
    m = frame[frame > 0].mean()
    if not m > 0:
        raise ValueError("empty frame after centroid placement")
    return frame / m


def align(frame, reference):
    """Phase-correlation alignment to a reference mean frame (2 iterations)."""
    for _ in range(2):
        f = np.fft.rfftn(frame)
        r = np.fft.rfftn(reference)
        cross = r * np.conj(f)
        cross /= np.abs(cross) + 1e-9
        corr = np.fft.fftshift(np.fft.irfftn(cross, s=frame.shape, axes=(0, 1, 2)))
        c = np.array([s // 2 for s in frame.shape])
        m = MAX_SHIFT
        win = corr[c[0]-m:c[0]+m+1, c[1]-m:c[1]+m+1, c[2]-m:c[2]+m+1]
        shift = tuple(int(p - m) for p in np.unravel_index(np.argmax(win), win.shape))
        if any(shift):
            frame = ndimage.shift(frame, shift, order=1, cval=0.0)
    return frame


def template_corr(crop, template_flat):
    """Pearson correlation between crop and template — QC check for placement."""
    return float(np.corrcoef(crop.ravel(), template_flat)[0, 1])


def preprocess(path, mean_frame, template_flat, margin=4):
    """Full pipeline: NIfTI → aligned frame → striatal crop.

    Tries the default path first; if template correlation is low, retries
    with affine_correct=True and head_restrict=False and keeps the best.
    Returns the crop as float32 numpy array.
    """
    crop_sl = crop_slices(margin)

    def _try(affine_correct=False, head_restrict=True):
        vol = load_resampled(path, affine_correct)
        frame = extract_frame(vol, head_restrict)
        frame = align(frame, mean_frame)
        return frame

    frame = _try()
    crop = frame[crop_sl]

    if template_corr(crop, template_flat) < 0.15:
        cands = [(template_corr(crop, template_flat), frame)]
        for kw in ({"affine_correct": True}, {"head_restrict": False}):
            try:
                f2 = _try(**kw)
                c2 = f2[crop_sl]
                cands.append((template_corr(c2, template_flat), f2))
            except Exception:
                pass
        frame = max(cands, key=lambda t: t[0])[1]
        crop = frame[crop_sl]

    return crop.astype(np.float32)
