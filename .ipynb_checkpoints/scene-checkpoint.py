import numpy as np
from glob import glob
from astropy import wcs
from grismagic.traces import GrismTrace
import copy
from astropy.stats import sigma_clipped_stats
import pandas as pd

import os
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import jax.numpy as jnp
from scipy.ndimage import binary_dilation

from astropy.nddata import Cutout2D

from .blot_utils import blot_segmentation, blot_direct_image

### Need to address segmentation issue
### Add contam
### add way to save


class Field(object):
    """
    Container for an entire JWST WFSS field.

    Loads all exposures, calibration information, and the source catalog.
    """

    def __init__(self, data_dir, cal_dir, cat, seg,contam, ngimg_dir=None, pad = 800):

        self.data_dir = data_dir
        self.cal_dir = cal_dir
        self.pad = pad
        self.ngimg_dir = ngimg_dir
        

        # -------------------------
        # Catalog
        # -------------------------

        self.cat = Table.read(cat).to_pandas()

        # -------------------------
        # Segmentation (TODO)
        # -------------------------

        self.in_seg = fits.getdata(seg)
        self.seg_wcs = wcs.WCS(fits.getheader(seg))

        # -------------------------
        # Exposure list
        # -------------------------

        self.exposures = []

        # -------------------------
        # Calibration caches
        # -------------------------

        self.trace = {}
        self.sensitivity = {}

        # -------------------------
        # Read all grism exposures
        # -------------------------

        for gfile in sorted(glob(os.path.join(self.data_dir, "*"))):

            gdat = fits.open(gfile)

            if gdat[0].header["FILTER"] == "CLEAR":
                continue

            filt = gdat[0].header["FILTER"]
            pupil = gdat[0].header["PUPIL"]

            # Find associated direct image
            _, a, b, _, _ = os.path.basename(
                gdat[1].header["ASTROREF"]
            ).split("_")

            dfile = glob(
                os.path.join(
                    self.data_dir,
                    f"*_{a}_{b}*_cal.fits"
                )
            )[0]

            ddat = fits.open(dfile)
            
            mx_edge = np.array(np.shape(gdat[2].data)) + pad

            exposure = {
                "gfile": gfile,
                "dfile": dfile,

                "filter": filt,
                "pupil": pupil,
                
                "spec": np.pad(gdat[1].data, pad),
                "direct": np.pad(ddat[1].data, pad),

                "spec_err": np.pad(gdat[2].data, pad),
                "direct_err": np.pad(ddat[2].data, pad),

                "spec_dq": np.pad(gdat[3].data, pad),
                "direct_dq": np.pad(ddat[3].data, pad),
                
                "gwcs": wcs.WCS(gdat[1].header),
                "dwcs": wcs.WCS(ddat[1].header),
                
                "gheader": gdat[1].header,
                "dheader": ddat[1].header,
                
                "valid_region" : np.array([pad, mx_edge[0], pad, mx_edge[1]])}
        
                
            # print("before")
            # print(exposure['dwcs']._naxis)
            # print(exposure['dwcs'].pixel_shape)
            # print(exposure['dwcs'].array_shape)

            
            exposure['gwcs'] = pad_wcs(exposure['gwcs'], pad)

            exposure['dwcs'] = pad_wcs(exposure['dwcs'], pad)

            outseg = blot_segmentation(self.in_seg, self.seg_wcs,
                                       exposure['dwcs'],exposure['direct'].shape,fill_value=0)
                        
            # print("after")
            # print(exposure['dwcs']._naxis)
            # print(exposure['dwcs'].pixel_shape)
            # print(exposure['dwcs'].array_shape)
            
            exposure['seg'] = outseg
            
            self.exposures.append(exposure)

            # -------------------------
            # Cache sensitivity curves
            # -------------------------

            for order in ["+1", "0", "+2", "+3", "-1"]:
                order_ = order 
                if order == "0":
                    order_ = "+0"
                
                key = (filt, pupil, order)

                if key not in self.sensitivity:

                    trns = fits.open(
                        os.path.join(self.cal_dir,
                            f"NIRISS_NIS_{filt}_{pupil}_{order_}_sens_pmap0041.fits"))

                    wave = np.asarray(trns[1].data.field(0),dtype=np.float32)

                    sens = np.asarray(trns[1].data.field(1),dtype=np.float32)

                    self.sensitivity[key] = (wave, sens)

            # -------------------------
            # Cache trace objects
            # -------------------------

            key = (filt, pupil)

            if key not in self.trace:

                self.trace[key] = GrismTrace.from_file(
                    os.path.join(self.cal_dir,f"NIRISS_{pupil}_{filt}.V5.conf"))

        # -------------------------
        # NIRISS gridded images
        # -------------------------
        if self.ngimg_dir != None:
            self.NGimages = {}
                
            dat = fits.open(glob(os.path.join(self.ngimg_dir, "*"))[0])

            outseg = blot_segmentation(self.in_seg, self.seg_wcs,
                                       wcs.WCS(dat[0].header),dat[0].data.shape,fill_value=0)
            
            for ngfile in sorted(glob(os.path.join(self.ngimg_dir, "*"))):
                dat = fits.open(ngfile)
                mask = binary_dilation(outseg > 0, iterations=5)
                mask |= ~np.isfinite(dat[0].data )
                mask |= (dat[0].data == 0)
                _, med, _ = sigma_clipped_stats(dat[0].data,mask=mask,sigma=3.0,maxiters=5)
                # print(med)
                self.NGimages[dat[0].header['FILTER']] = {}
                self.NGimages[dat[0].header['FILTER']]['image'] = (dat[0].data - med) * dat[0].header['PHOTFLAM']
                self.NGimages[dat[0].header['FILTER']]['background'] = med
                self.NGimages[dat[0].header['FILTER']]['pivot'] = dat[0].header['PIVOT']
                self.NGimages[dat[0].header['FILTER']]['ofile'] = dat[0].header['OFILE']
                self.NGimages[dat[0].header['FILTER']]['photflam'] = dat[0].header['PHOTFLAM']
                self.NGimages[dat[0].header['FILTER']]['wcs'] = wcs.WCS(dat[0].header)
            
        # -------------------------
        # Contam Table gen
        # -------------------------
        valid_gals = {exp['pupil']:{ order:[] for order in ['+1', '0', '+2', '+3', '-1']} for exp in self.exposures}
        allids = {exp['pupil'] : [] for exp in self.exposures}
        
        for i, exp in enumerate(self.exposures):
        
            for order in ['+1', '0', '+2', '+3', '-1']:
            
                sh = np.shape(exp['direct'])
                    
                y1, y2, x1, x2 = get_beam_limits(sh[1]//2, sh[0]//2, self.trace[exp['filter'],exp['pupil']], sh[0]//2 - self.pad, exp['filter'][-1], order)
            
                ids = np.unique(self.exposures[0]['seg'][int(sh[0] - y2) : int(sh[0] - y1), int(sh[1] - x2) : int(sh[1] - x1) ])
            
                valid_gals[exp['pupil']][order].extend(ids[ids >0])
                allids[exp['pupil']].extend( ids[ids >0])
        
        self.contam_table = {}
        for pupil in valid_gals:
        
            setids = list(set(allids[pupil]))
            
            contam_dict = {'ids':setids,'+1':[], '0':[], '+2':[], '+3':[], '-1':[]}
            for s in setids:
                for order in ['+1', '0', '+2', '+3', '-1']:
                    contam_dict[order].append(s in valid_gals[pupil][order] )
        
            self.contam_table[pupil] = pd.DataFrame(contam_dict)

class Beam(object):

    def __init__(self):

        self.direct = {}
        self.spec = {}

        self.meta = {}

        self.cutout_limits = None

class Galaxy(object):
    def __init__(self, field, gid, sz=20, order = "+1"):
        self.gid = gid
        self.sz = sz
        self.order = order
        self.pad = field.pad
        self.field = field
        # catalog information
        source = field.cat.query(f"id == {gid}")

        self.ra = source.ra.values[0]
        self.dec = source.dec.values[0]

        self.sky = SkyCoord(ra=self.ra * u.deg,dec=self.dec * u.deg,frame="icrs")

        # beams grouped by PUPIL
        self.beams = {}


    def extract(self,):
        """
        Extract all beams for this object.

        Groups beams by JWST PUPIL
        (F115W, F150W, F200W, etc.)
        """

        # -------------------------
        # NG image cutouts
        # -------------------------
        if self.field.ngimg_dir != None:
            self.images = {}
            for filt in self.field.NGimages:
                self.images[filt] = {}

                (img,nwcs,x,y) = self.cutout_img(self.field.NGimages[filt]['wcs'],self.field.NGimages[filt]['image'])
        
                self.images[filt]["sci"] = img
                self.images[filt]["wcs"] = nwcs
                self.images[filt]["pivot"] = self.field.NGimages[filt]['pivot']
        
                self.images[filt]["x"] = x
                self.images[filt]["y"] = y
        
                self.images[filt]["npx"] = x - self.pad
                self.images[filt]["npy"] = y - self.pad

        NGwcs = copy.deepcopy(self.field.NGimages['F200W']['wcs'])
        NGwcs.pscale = 1
        
        for exp in self.field.exposures:
            # skip exposures where object is not observed
            if not self.in_image(exp["dwcs"], exp["direct"]):
                print(
                    "Object outside direct footprint:",
                    exp["filter"],
                    exp["pupil"]
                )
                continue      
                
            beam = Beam()
            # -------------------------
            # Direct cutout
            # -------------------------
            (img,err,seg,dwcs,x,y) = self.cutout_dir(exp["dwcs"],exp["direct"],exp["direct_err"],exp["seg"])
        
            mask = (np.isfinite(img) &(img != 0))
            
            fraction = mask.mean()
            
            if fraction < 0.9:            
                img = blot_direct_image(self.field.NGimages[exp['pupil']]['image'], NGwcs, dwcs)
           
            beam.direct["sci"] = img
            beam.direct["err"] = err
            beam.direct["seg"] = seg
            beam.direct["wcs"] = dwcs

            beam.direct["x"] = x
            beam.direct["y"] = y

            beam.direct["npx"] = x - self.pad
            beam.direct["npy"] = y - self.pad

            
            # -------------------------
            # Metadata
            # -------------------------
            beam.meta["filter"] = exp["filter"]
            beam.meta["pupil"] = exp["pupil"]
            beam.meta["gfile"] = exp["gfile"]
            beam.meta["dfile"] = exp["dfile"]
            
            key = (exp['filter'], exp['pupil'])
            beam.meta["trace"] = self.field.trace[key]
            
            key = (exp['filter'], exp['pupil'], self.order)
            beam.meta["sens"] = self.field.sensitivity[key]

            # -------------------------
            # Find spectral cutout limits
            # -------------------------
            limits = self.get_beam_limits(x,y,self.field.trace[(exp["filter"],exp["pupil"])],
                self.sz,exp["filter"][-1], self.order)

            beam.cutout_limits = limits
            
            if (limits[0] > exp["valid_region"][1]) or (limits[1] < exp["valid_region"][0]) or (limits[2] > exp["valid_region"][3]) or (limits[3] < exp["valid_region"][2]):
                beam.validity = 'invalid'

            elif (limits[0] > exp["valid_region"][0]) and (limits[1] < exp["valid_region"][1]) and (limits[2] > exp["valid_region"][2]) and (limits[3] < exp["valid_region"][3]):
                beam.validity = 'valid'

            else:
                beam.validity = 'partial'

            beam.valid_region = exp["valid_region"]
                
            # -------------------------
            # Spectral cutout
            # -------------------------
            spec, err, swcs = self.cutout_spec(exp["gwcs"],exp["spec"],exp["spec_err"],limits)

            beam.spec["sci"] = spec
            beam.spec["err"] = err
            beam.spec["wcs"] = swcs
            
            # -------------------------
            # Spectral prep
            # -------------------------
            beam.spec["x_trace"], beam.spec["y_trace"], beam.spec["lam"] = beam.meta["trace"].get_trace(beam.direct["npx"], beam.direct["npy"], self.order)
            beam.spec["dlam"] = np.abs(jnp.gradient(beam.spec["lam"] * 1e4))
            beam.spec["sens"] = jnp.interp(beam.spec["lam"], np.ravel(beam.meta['sens'][0]), np.ravel(beam.meta['sens'][1])) * beam.spec["dlam"]

            # -------------------------
            # Add beam
            # -------------------------

            pupil = exp["pupil"]

            if pupil not in self.beams:
                self.beams[pupil] = []

            self.beams[pupil].append(beam)


    def cutout_dir(self, parent_wcs, direct_image, err_image, seg_image):
        """
        Extract direct image cutout.
        """
        x, y = np.array(parent_wcs.world_to_pixel(self.sky)).astype(int)

        cutout = Cutout2D(data=direct_image,position=(x,y),
            size=(2*self.sz,2*self.sz),wcs=parent_wcs)

        img = cutout.data
        img[np.isnan(img)] = 0

        cutout = Cutout2D(data=err_image,position=(x,y),
            size=(2*self.sz,2*self.sz),wcs=parent_wcs)

        err = cutout.data
        err[np.isnan(err)] = 0

        cutout = Cutout2D(data=seg_image,position=(x,y),
            size=(2*self.sz,2*self.sz),wcs=parent_wcs)

        seg = cutout.data
        
        return (img.astype(np.float32),err.astype(np.float32),seg.astype(np.int32),cutout.wcs,x,y)


    def cutout_img(self, parent_wcs, direct_image):
        """
        Extract direct image cutout.
        """
        x, y = np.array(parent_wcs.world_to_pixel(self.sky)).astype(int)

        cutout = Cutout2D(data=direct_image,position=(x,y),
            size=(2*self.sz,2*self.sz),wcs=parent_wcs)

        img = cutout.data
        img[np.isnan(img)] = 0

        return (img.astype(np.float32),cutout.wcs,x,y)
        
    def in_image(self, wcs, image):
        """
        Check whether the object position falls inside an image footprint.
    
        Parameters
        ----------
        wcs : astropy.wcs.WCS
            Image WCS
        image : ndarray
            Image array
    
        Returns
        -------
        bool
            True if source position is inside image
        """
    
        x, y = wcs.world_to_pixel(self.sky)
    
        ny, nx = image.shape
    
        return (
            (x >= 0) &
            (x < nx) &
            (y >= 0) &
            (y < ny)
        )

    def get_beam_limits(self,x,y,trace,sz,orient,order):
        """
        Determine grism cutout boundaries.
        """
        x_trace, y_trace, lam = trace.get_trace(x,y,self.order)

        if orient == "C":
            return [y-sz,y+sz,
                    int(x_trace[0])-sz,int(x_trace[-1])+sz]


        elif orient == "R":
            return [int(y_trace[0])-sz,int(y_trace[-1])+sz,
                x-sz,x+sz]
        else:
            raise ValueError( "Unknown grism orientation")


    def cutout_spec(self,parent_wcs,spec_image, err_image,limits):
        y1,y2,x1,x2 = limits
        
        size = (y2-y1,x2-x1)
        
        position = ((x1+x2)//2, (y1+y2)//2)

        cutout = Cutout2D(data=spec_image,position=position,size=size,wcs=parent_wcs)

        spec = cutout.data
        spec[np.isnan(spec)] = 0

        cutout = Cutout2D(data=err_image,position=position,size=size,wcs=parent_wcs)

        err = cutout.data
        err[np.isnan(err)] = 0

        
        return (spec.astype(np.float32),err.astype(np.float32),cutout.wcs)



def get_beam_limits(x,y,trace,sz,orient,order):
    """
    Determine grism cutout boundaries.
    """
    x_trace, y_trace, lam = trace.get_trace(x,y,order)

    if orient == "C":
        return [y-sz,y+sz,
                int(x_trace[0])-sz,int(x_trace[-1])+sz]


    elif orient == "R":
        return [int(y_trace[0])-sz,int(y_trace[-1])+sz,
            x-sz,x+sz]
    else:
        raise ValueError( "Unknown grism orientation")

def pad_wcs(wcs_in, pad):
    """
    Return a copy of a WCS padded by `pad` pixels on each edge.

    Parameters
    ----------
    wcs_in : astropy.wcs.WCS
        Input WCS.

    pad : int or (ypad, xpad)
        Padding applied to each edge.

    Returns
    -------
    owcs : astropy.wcs.WCS
        Padded WCS.
    """

    if np.isscalar(pad):
        py = px = int(pad)
    else:
        py, px = map(int, pad)

    owcs = copy.deepcopy(wcs_in)

    #
    # Image dimensions
    #
    nx, ny = owcs._naxis

    nx += 2 * px
    ny += 2 * py

    owcs._naxis = [nx, ny]

    if hasattr(owcs, "_naxis1"):
        owcs._naxis1 = nx

    if hasattr(owcs, "_naxis2"):
        owcs._naxis2 = ny

    if hasattr(owcs, "naxis1"):
        owcs.naxis1 = nx

    if hasattr(owcs, "naxis2"):
        owcs.naxis2 = ny

    if hasattr(owcs, "pixel_shape"):
        owcs.pixel_shape = (nx, ny)

    if hasattr(owcs, "array_shape"):
        owcs.array_shape = (ny, nx)

    #
    # Shift reference pixel
    #
    owcs.wcs.crpix[0] += px
    owcs.wcs.crpix[1] += py

    #
    # SIP distortion
    #
    if getattr(owcs, "sip", None) is not None:
        owcs.sip.crpix[0] += px
        owcs.sip.crpix[1] += py

    #
    # Distortion lookup tables (ACS, etc.)
    #
    for ext in (owcs.cpdis1, owcs.cpdis2,
                owcs.det2im1, owcs.det2im2):
        if ext is not None:
            ext.crval[0] += px
            ext.crval[1] += py

    return owcs
    
