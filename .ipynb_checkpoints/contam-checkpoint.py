import numpy as np
from .scene import Galaxy
from .core import Sleuth

def id_contam_candidate(field, MB, flux_table, limits, gid, pupil, order, pad, sz):
    beam_start_x= MB.spec['x_trace'][MB.spec['lam'] == min(MB.spec['lam'])][0]
    beam_stop_x = MB.spec['x_trace'][MB.spec['lam'] == max(MB.spec['lam'])][0]
    beam_start_y= MB.spec['y_trace'][MB.spec['lam'] == min(MB.spec['lam'])][0]
    beam_stop_y = MB.spec['y_trace'][MB.spec['lam'] == max(MB.spec['lam'])][0]
    
    x_trace, y_trace, lam = MB.meta["trace"].get_trace(MB.direct["npx"],MB.direct["npy"],order)

    # Trace endpoints in wavelength
    i0 = np.argmin(lam)
    i1 = np.argmax(lam)

    dx_trace = x_trace[i1] - x_trace[i0]
    dy_trace = y_trace[i1] - y_trace[i0]

    # Offset between source position and trace origin
    delta_x = MB.direct["npx"] - x_trace[i0]
    delta_y = MB.direct["npy"] - y_trace[i0]

    # Dispersed footprint of the source region
    x0 = beam_start_x + delta_x - dx_trace
    x1 = beam_stop_x + delta_x

    y0 = beam_start_y + delta_y - dy_trace
    y1 = beam_stop_y + delta_y 

    x_lims = np.sort([x0, x1]) + pad
    y_lims = np.sort([y0, y1]) + pad

    # Make sure short-dispersion orders have enough width
    if np.ptp(x_lims) < 2 * sz:
        xc = np.mean(x_lims)
        x_lims = [xc - sz, xc + sz]

    if np.ptp(y_lims) < 2 * sz:
        yc = np.mean(y_lims)
        y_lims = [yc - sz, yc + sz]

    seg = field.exposures[MB.meta['exposure_number']]['seg']
    
    ylo, yhi = np.floor(y_lims).astype(int)
    xlo, xhi = np.floor(x_lims).astype(int)
    
    ylo = max(0, ylo)
    xlo = max(0, xlo)
    yhi = min(seg.shape[0], yhi)
    xhi = min(seg.shape[1], xhi)
    
    ids = np.unique(seg[ylo:yhi, xlo:xhi])
    ids = ids[ids != 0]

    out_ids = []
    for cid in ids:
        # Don't contaminate yourself
        if cid == gid:
            continue

        # No flux measurement
        if cid not in flux_table[pupil]:
            continue

        # Flux threshold
        if flux_table[pupil][cid][0] >= limits[pupil][order]:
                out_ids.append(cid)

    return out_ids

def extract_contam(field, seg, cid, order):
    yy, xx = np.where(seg == cid)

    xmin, xmax = xx.min(), xx.max()
    ymin, ymax = yy.min(), yy.max()

    pad = 4

    width = xmax - xmin + 1 + 2 * pad
    height = ymax - ymin + 1 + 2 * pad

    csz = np.max([width, height]) // 2

    contam = Galaxy(field,cid,sz=csz,order=order,substitute_filter='F606W')

    contam.extract()

    cx = Sleuth(contam)

    cx.load_images('reference')
    cx.single_seg()
    cx.reseg()
    cx.prepare_foward_model()
    cx.gen_mask()

    return cx

def geometric_check(cx, pupil, bid, beam_limits, overlap_map):    
    cy1, cy2, cx1, cx2 = cx.obj.beams[pupil][bid].cutout_limits
    ty1, ty2, tx1, tx2 = beam_limits
    
    ylo = max(ty1, cy1)
    yhi = min(ty2, cy2)
    
    xlo = max(tx1, cx1)
    xhi = min(tx2, cx2)
    
    if ylo >= yhi or xlo >= xhi:
        return -99

    else:
        contam_map = np.zeros([3648,3648])
        contam_map[cy1:cy2,cx1:cx2] = cx.obj.beams[pupil][bid].mask
    
        if np.any(np.asarray(overlap_map, dtype=bool) & np.asarray(contam_map, dtype=bool)):
            return cx.obj.gid
        else:
            return -99
            