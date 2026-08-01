import numpy as np
from astropy import wcs
from drizzlepac.astrodrizzle import ablot

def blot_direct_image(ref_data, ref_wcs, out_wcs):
    out = ablot.do_blot(
        ref_data,
        ref_wcs,
        out_wcs,
        exptime=1.0,
        coeffs=True,
        interp="poly5",
        sinscl=1.0,
        stepsize=10,)

    out[np.isnan(out)] = 0
    return out

def blot_segmentation(seg_image,seg_wcs,out_wcs,out_shape,fill_value=0):
    """
    Transfer a segmentation image between WCS frames
    using nearest-neighbor mapping.

    Parameters
    ----------
    seg_image : ndarray
        Input segmentation image.

    seg_wcs : astropy.wcs.WCS
        WCS of segmentation image.

    out_wcs : astropy.wcs.WCS
        WCS of target image.

    out_shape : tuple
        Shape of output image (ny,nx).

    Returns
    -------
    out_seg : ndarray
        Segmentation mapped to output frame.
    """

    # output pixel grid
    yy, xx = np.indices(out_shape, dtype=float)
    
    ra, dec = out_wcs.all_pix2world(xx, yy, 0)
    
    xin, yin = seg_wcs.all_world2pix(ra, dec, 0)
    
    xin = np.floor(xin + 0.5).astype(np.int32)
    yin = np.floor(yin + 0.5).astype(np.int32)
    
    out_seg = np.full(out_shape, fill_value, dtype=seg_image.dtype)
    
    good = ((xin >= 0) &(yin >= 0) &(xin < seg_image.shape[1]) &(yin < seg_image.shape[0]))

    out_seg[good] = seg_image[yin[good], xin[good]]

    return out_seg