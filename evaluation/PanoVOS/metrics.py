import math
import numpy as np
from scipy.ndimage import distance_transform_edt


def db_eval_iou(annotation, segmentation, void_pixels=None):
    """Compute region similarity as the Jaccard Index.

    Arguments:
        annotation   (ndarray): binary annotation map.
        segmentation (ndarray): binary segmentation map.
        void_pixels  (ndarray): optional mask with void pixels.

    Return:
        jaccard (float or ndarray): region similarity.
    """
    assert annotation.shape == segmentation.shape, \
        f'Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match.'
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape, \
            f'Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match.'
        void_pixels = void_pixels.astype(bool)
        valid = np.logical_not(void_pixels)
        inters = np.sum((segmentation & annotation) & valid, axis=(-2, -1))
        union = np.sum((segmentation | annotation) & valid, axis=(-2, -1))
    else:
        inters = np.sum(segmentation & annotation, axis=(-2, -1))
        union = np.sum(segmentation | annotation, axis=(-2, -1))

    with np.errstate(divide='ignore', invalid='ignore'):
        j = inters / union
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j


def db_eval_boundary(annotation, segmentation, void_pixels=None, bound_th=0.008):
    assert annotation.shape == segmentation.shape
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape
    if annotation.ndim == 3:
        n_frames = annotation.shape[0]
        f_res = np.zeros(n_frames)
        for frame_id in range(n_frames):
            void_pixels_frame = None if void_pixels is None else void_pixels[frame_id, :, :]
            f_res[frame_id] = f_measure(
                segmentation[frame_id, :, :],
                annotation[frame_id, :, :],
                void_pixels_frame,
                bound_th=bound_th,
            )
    elif annotation.ndim == 2:
        f_res = f_measure(segmentation, annotation, void_pixels, bound_th=bound_th)
    else:
        raise ValueError(f'db_eval_boundary does not support tensors with {annotation.ndim} dimensions')
    return f_res


def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    """Compute boundary F-measure between foreground_mask and gt_mask."""
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    fg = foreground_mask.astype(bool)
    gt = gt_mask.astype(bool)
    if void_pixels is not None:
        valid = np.logical_not(void_pixels.astype(bool))
        fg = fg & valid
        gt = gt & valid

    bound_pix = bound_th if bound_th >= 1 else np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    fg_boundary = _seg2bmap(fg)
    gt_boundary = _seg2bmap(gt)

    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        union_b = fg_boundary | gt_boundary
        ys = np.flatnonzero(np.any(union_b, axis=1))
        xs = np.flatnonzero(np.any(union_b, axis=0))
        y0, y1 = ys[0], ys[-1] + 1
        x0, x1 = xs[0], xs[-1] + 1
        fg_sub = fg_boundary[y0:y1, x0:x1]
        gt_sub = gt_boundary[y0:y1, x0:x1]
        fg_dil = distance_transform_edt(~fg_sub) <= bound_pix
        gt_dil = distance_transform_edt(~gt_sub) <= bound_pix
        precision = np.sum(fg_sub & gt_dil) / float(n_fg)
        recall = np.sum(gt_sub & fg_dil) / float(n_gt)

    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F


def _seg2bmap(seg, width=None, height=None):
    """Compute a binary boundary map with 1-pixel-wide boundaries."""
    seg = seg.astype(bool)

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        (width > w) or (height > h) or (abs(ar1 - ar2) > 0.01)
    ), "Can't convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width), dtype=bool)
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = True

    return bmap
