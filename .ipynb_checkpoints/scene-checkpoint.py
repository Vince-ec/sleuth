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

import json, pickle
import h5py

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

    def __init__(self, grism_files, ref_files, cal_dir, cat, seg,contam, img_dir=None, pad = 800):

        self.grism_files = grism_files
        self.ref_files = ref_files
        self.cal_dir = cal_dir
        self.pad = pad
        self.img_dir = img_dir
        
        # -------------------------
        # File sort
        # -------------------------
        
        self.files_info = {}
        for f in self.grism_files:
            dat = fits.open(f)
            self.files_info[f] = {'instrument': dat[0].header['INSTRUME'],
                            'filter': dat[0].header['FILTER'],
                            'pupil': dat[0].header['PUPIL']}
        
        self.reffiles_info = {}
        for f in self.ref_files:
            dat = fits.open(f)
            self.reffiles_info[dat[0].header['PUPIL']] = {'instrument': dat[0].header['INSTRUME'], 'filename' : f}

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
        
        for k, d in self.files_info.items():

            gdat = fits.open(k)

            filt = gdat[0].header["FILTER"]
            pupil = gdat[0].header["PUPIL"]

            dfile = self.reffiles_info[d['pupil']]['filename']

            ddat = fits.open(dfile)
            
            mx_edge = np.array(np.shape(gdat[2].data)) + pad

            gwcs = pad_wcs(wcs.WCS(gdat[1].header), pad)
        
            dwcs = wcs.WCS(ddat[0].header)
            dwcs.pscale = 1
            
            refimg = blot_direct_image(np.array(ddat[0].data).astype(np.float32), dwcs, gwcs)
            
            exposure = {
                "gfile": k,
                "dfile": dfile,

                "filter": filt,
                "pupil": pupil,
                
                "spec": np.pad(gdat[1].data, pad),
                "direct": refimg,

                "spec_err": np.pad(gdat[2].data, pad),
                # "direct_err": np.pad(ddat[2].data, pad),

                "spec_dq": np.pad(gdat[3].data, pad),
                # "direct_dq": np.pad(ddat[3].data, pad),
                
                "gwcs": gwcs,
                # "dwcs": wcs.WCS(ddat[1].header),
                
                "gheader": gdat[1].header,
                "dheader": ddat[0].header,
                
                "valid_region" : np.array([pad, mx_edge[0], pad, mx_edge[1]])}

            outseg = blot_segmentation(self.in_seg, self.seg_wcs,
                                       exposure['gwcs'],exposure['direct'].shape,fill_value=0)
                        
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
        # Hi-res images
        # -------------------------
        self.imgdict = {'imgs':{}, 'err':{}}
        if self.img_dir != None:
            self.imgdict['seg'] = {'file': glob(self.img_dir + '*seg*')[0]}
            
            for f in glob(self.img_dir + 'images/*'):
                dat = fits.open(f)
                
                if 'FILTER' in dat[0].header:
                    tag = dat[0].header['FILTER']
                    if tag == 'CLEAR2L':
                        tag = 'F606W'                      
                else:
                    tag = dat[0].header['FILTER2']
                    if tag == 'CLEAR2L':
                        tag = 'F606W'
                
                self.imgdict['imgs'][tag] = {'file':f}

            for f in glob(self.img_dir + 'whts/*'):
                dat = fits.open(f)
                
                if 'FILTER' in dat[0].header:
                    tag = dat[0].header['FILTER']
                    if tag == 'CLEAR2L':
                        tag = 'F606W'                      
                else:
                    tag = dat[0].header['FILTER2']
                    if tag == 'CLEAR2L':
                        tag = 'F606W'
                
                self.imgdict['err'][tag] = {'file':f}

        
            # for f in glob(self.ngimg_dir + 'whts/*'):
            #     dat = fits.open(f)
            #     self.imgdict['err'][dat[0].header['FILTER']] = {}
            #     self.imgdict['err'][dat[0].header['FILTER']]['file'] = f

            # self.NGimages = {}
                
            # # dat = fits.open(glob(os.path.join(self.ngimg_dir, "*"))[0])

            # outseg = blot_segmentation(self.in_seg, self.seg_wcs,
            #                            wcs.WCS(dat[0].header),dat[0].data.shape,fill_value=0)
            
            # # for ngfile in sorted(glob(os.path.join(self.ngimg_dir, "*"))):
            # for k in ngdict['imgs']:

            #     # dat = fits.open(ngfile)
            #     mask = binary_dilation(outseg > 0, iterations=5)
            #     mask |= ~np.isfinite(ngdict['imgs'][k]['img'] )
            #     mask |= (ngdict['imgs'][k]['img']  == 0)
            #     _, med, _ = sigma_clipped_stats(ngdict['imgs'][k]['img'] *1e20,mask=mask,sigma=3.0,maxiters=5)

            #     self.NGimages[k] = {}
            #     self.NGimages[k]['image'] = (ngdict['imgs'][k]['img']  - med*1e-20)
            #     self.NGimages[k]['err'] = ngdict['err'][k]['img'] 
            #     self.NGimages[k]['background'] = med
            #     self.NGimages[k]['pivot'] = ngdict['imgs'][k]['header']['PIVOT']
            #     self.NGimages[k]['ofile'] = ngdict['imgs'][k]['header']['OFILE']
            #     self.NGimages[k]['photflam'] = ngdict['imgs'][k]['header']['PHOTFLAM']
            #     self.NGimages[k]['wcs'] = wcs.WCS(ngdict['imgs'][k]['header'])
            
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

    def save_field(self, outdir):
        os.makedirs(outdir, exist_ok=True)
    
        # ── 1. Exposures (arrays + WCS) ─────────────────────────────────────
        array_keys = ["spec", "direct", "spec_err", "spec_dq", "seg"]
        meta_keys  = ["gfile", "dfile", "filter", "pupil", "valid_region",
                      "gheader", "dheader"]
    
        with h5py.File(os.path.join(outdir, "exposures.h5"), "w") as h5:
            for i, exp in enumerate(self.exposures):
                grp = h5.create_group(f"exp_{i:04d}")
                for k in array_keys:
                    if k in exp:
                        grp.create_dataset(k, data=exp[k], compression="gzip")
    
        # WCS and headers need pickle (FITS headers are not plain dicts)
        exp_meta = []
        exp_wcs  = []
        for i, exp in enumerate(self.exposures):
            meta = {k: exp[k] for k in ["gfile","dfile","filter","pupil"]}
            meta["valid_region"] = exp["valid_region"].tolist()
            exp_meta.append(meta)
            exp_wcs.append(exp["gwcs"])    # WCS object → pickle
    
        with open(os.path.join(outdir, "exp_meta.json"), "w") as f:
            json.dump(exp_meta, f, indent=2)
    
        with open(os.path.join(outdir, "exp_wcs.pkl"), "wb") as f:
            pickle.dump(exp_wcs, f)
    
        # ── 2. Catalogs ──────────────────────────────────────────────────────
        self.cat.to_parquet(os.path.join(outdir, "cat.parquet"))
    
        # ── 3. Contam tables ─────────────────────────────────────────────────
        for pupil, df in self.contam_table.items():
            df.to_parquet(os.path.join(outdir, f"contam_{pupil}.parquet"))
    
        # ── 4. Sensitivity curves ────────────────────────────────────────────
        sens_serializable = {
            f"{filt}__{pupil}__{order}": {"wave": wave.tolist(), "sens": sens.tolist()}
            for (filt, pupil, order), (wave, sens) in self.sensitivity.items()
        }
        with open(os.path.join(outdir, "sensitivity.json"), "w") as f:
            json.dump(sens_serializable, f)
    
        # ── 5. Trace objects ─────────────────────────────────────────────────
        # GrismTrace objects are not JSON-safe; pickle is safest
        trace_serializable = {f"{filt}__{pupil}": obj
                              for (filt, pupil), obj in self.trace.items()}
        with open(os.path.join(outdir, "trace.pkl"), "wb") as f:
            pickle.dump(trace_serializable, f)
    
        # ── 6. Top-level metadata ─────────────────────────────────────────────
        top_meta = {
            "pad":          self.pad,
            "grism_files":  self.grism_files,
            "ref_files":    self.ref_files,
            "cal_dir":      self.cal_dir,
            "img_dir":      self.img_dir,
            "files_info":   self.files_info,
            "reffiles_info": self.reffiles_info,
        }
        with open(os.path.join(outdir, "meta.json"), "w") as f:
            json.dump(top_meta, f, indent=2)
        
        # ── 7. save out images ───────────────────────────────────────────────
        with open(os.path.join(outdir, "imgdict.json"), "w") as f:
            json.dump(self.imgdict, f, indent=2)
        
        print(f"Saved to {outdir}/")

def load_field(outdir):
    field = Field.__new__(Field)   # skip __init__

    # ── metadata ──────────────────────────────────────────────────────────
    with open(os.path.join(outdir, "meta.json")) as f:
        meta = json.load(f)
    field.__dict__.update(meta)

    # ── catalogs ──────────────────────────────────────────────────────────
    field.cat = pd.read_parquet(os.path.join(outdir, "cat.parquet"))

    contam_files = glob(os.path.join(outdir, "contam_*.parquet"))
    field.contam_table = {
        os.path.basename(p).removeprefix("contam_").removesuffix(".parquet"):
        pd.read_parquet(p)
        for p in contam_files
    }

    # ── sensitivity ────────────────────────────────────────────────────────
    with open(os.path.join(outdir, "sensitivity.json")) as f:
        raw = json.load(f)
    field.sensitivity = {
        tuple(k.split("__")): (np.array(v["wave"]), np.array(v["sens"]))
        for k, v in raw.items()
    }

    # ── trace ──────────────────────────────────────────────────────────────
    with open(os.path.join(outdir, "trace.pkl"), "rb") as f:
        trace_raw = pickle.load(f)
    field.trace = {tuple(k.split("__")): v for k, v in trace_raw.items()}

    # ── WCS ────────────────────────────────────────────────────────────────
    with open(os.path.join(outdir, "exp_wcs.pkl"), "rb") as f:
        exp_wcs = pickle.load(f)

    with open(os.path.join(outdir, "exp_meta.json")) as f:
        exp_meta = json.load(f)

    # ── exposures (arrays) ─────────────────────────────────────────────────
    array_keys = ["spec", "direct", "spec_err", "spec_dq", "seg"]
    field.exposures = []
    with h5py.File(os.path.join(outdir, "exposures.h5"), "r") as h5:
        for i, (meta_i, gwcs_i) in enumerate(zip(exp_meta, exp_wcs)):
            exp = dict(meta_i)
            exp["valid_region"] = np.array(meta_i["valid_region"])
            exp["gwcs"] = gwcs_i
            grp = h5[f"exp_{i:04d}"]
            for k in array_keys:
                if k in grp:
                    exp[k] = grp[k][:]
            field.exposures.append(exp)
    # ── images ─────────────────────────────────────────────────

    with open(os.path.join(outdir, "imgdict.json")) as f:
        field.imgdict = json.load(f)
    
    # ── segmentation (stored per-exposure; top-level seg/wcs if needed) ────
    # in_seg and seg_wcs can be reloaded from original seg file if needed
    # or stored similarly; skipped here unless you need them post-load

    return field

    
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
        self.imgdict = field.imgdict
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
        if self.field.img_dir != None:

            seg = fits.open(self.field.imgdict['seg']['file'])[0].data
            seg_wcs = wcs.WCS(fits.open(self.field.imgdict['seg']['file'])[0].header)
            
            pos = np.where(seg == self.gid )
            
            ysize = pos[0].max() - pos[0].min() 
            xsize = pos[1].max() - pos[1].min() 
            size = (np.max([ysize,xsize])*1.3)//2
                                    
            self.images = {}
            
            (img,nwcs,x,y) = self.cutout_img_NC(seg_wcs, seg, size = size)
            
            self.images["seg"] = img 
            self.images["seg_wcs"] = nwcs
            
            for filt in self.field.imgdict['imgs']:
                self.images[filt] = {}
            
                dat = fits.open(self.field.imgdict['imgs'][filt]['file'])
                
                (img,nwcs,x,y) = self.cutout_img_NC(wcs.WCS(dat[0].header), dat[0].data, size = size)
            
                self.images[filt]["sci"] = img * dat[0].header['PHOTFLAM']
                self.images[filt]["wcs"] = nwcs
                self.images[filt]["pivot"] = dat[0].header['PHOTPLAM']

                dat = fits.open(self.field.imgdict['err'][filt]['file'])
                
                (img,nwcs,x,y) = self.cutout_img_NC(wcs.WCS(dat[0].header), dat[0].data, size = size)
            
                self.images[filt]["err"] = 1/np.sqrt(img) * dat[0].header['PHOTFLAM']


                
        # NGwcs = copy.deepcopy(self.field.NGimages['F200W']['wcs'])
        # NGwcs.pscale = 1
        
        for exp in self.field.exposures:
            # skip exposures where object is not observed
            if not self.in_image(exp["gwcs"], exp["direct"]):
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
            # (img,err,seg,dwcs,x,y) = self.cutout_dir(exp["gwcs"],exp["direct"],exp["direct_err"],exp["seg"])
            (img,dwcs,x,y) = self.cutout_img(exp["gwcs"],exp["direct"])
            (seg,dwcs,x,y) = self.cutout_img(exp["gwcs"],exp["seg"])
        
            mask = (np.isfinite(img) &(img != 0))
            
            fraction = mask.mean()
            
            if fraction < 0.9:            
                img = blot_direct_image(self.field.NGimages[exp['pupil']]['image'], NGwcs, dwcs)
           
            beam.direct["sci"] = img
            # beam.direct["err"] = err
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


    def cutout_dir(self, parent_wcs, direct_image,seg_image):
        """
        Extract direct image cutout.
        """
        x, y = np.array(parent_wcs.world_to_pixel(self.sky)).astype(int)

        cutout = Cutout2D(data=direct_image,position=(x,y),
            size=(2*self.sz,2*self.sz),wcs=parent_wcs)

        img = cutout.data
        img[np.isnan(img)] = 0

        # cutout = Cutout2D(data=err_image,position=(x,y),
        #     size=(2*self.sz,2*self.sz),wcs=parent_wcs)

        # err = cutout.data
        # err[np.isnan(err)] = 0

        cutout = Cutout2D(data=seg_image,position=(x,y),
            size=(2*self.sz,2*self.sz),wcs=parent_wcs)

        seg = cutout.data
        
        return (img.astype(np.float32),seg.astype(np.int32),cutout.wcs,x,y)


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

    def cutout_img_NC(self, parent_wcs, direct_image, size):
        """
        Extract direct image cutout.
        """
        x, y = np.array(parent_wcs.world_to_pixel(self.sky)).astype(int)

        cutout = Cutout2D(data=direct_image,position=(x,y),
            size=(2*size,2*size),wcs=parent_wcs)

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

    def save_galaxy(self, outdir):
        os.makedirs(outdir, exist_ok=True)
    
        # ── 1. Top-level metadata ─────────────────────────────────────────────
        meta = {
            "gid":   self.gid,
            "sz":    self.sz,
            "order": self.order,
            "pad":   self.pad,
            "ra":    self.ra,
            "dec":   self.dec,
        }
        with open(os.path.join(outdir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    
        # ── 2. Hi-res images ──────────────────────────────────────────────────
        if hasattr(self, "images") and self.images:
            img_wcs = {}
            with h5py.File(os.path.join(outdir, "images.h5"), "w") as h5:
                for filt, val in self.images.items():
                    if filt == "seg_wcs":
                        continue
                    if filt == "seg":
                        h5.create_dataset("seg", data=self.images["seg"],
                                          compression="gzip", compression_opts=4)
                        img_wcs["seg"] = self.images["seg_wcs"].to_header_string()
                    else:
                        grp = h5.create_group(filt)
                        for k, v in val.items():
                            if k == "wcs":
                                img_wcs[filt] = v.to_header_string()
                            elif np.isscalar(v):
                                grp.attrs[k] = v
                            else:
                                grp.create_dataset(k, data=v,
                                                   compression="gzip", compression_opts=4)
            with open(os.path.join(outdir, "images_wcs.pkl"), "wb") as f:
                pickle.dump(img_wcs, f)
    
        # ── 3. Beams ──────────────────────────────────────────────────────────
        beam_meta      = {}
        beam_wcs       = {}
        beam_trace     = {}   # full GrismTrace objects, not keys
        beam_sens      = {}   # full sensitivity arrays
    
        with h5py.File(os.path.join(outdir, "beams.h5"), "w") as h5:
            for pupil, beam_list in self.beams.items():
                beam_meta[pupil]  = []
                beam_wcs[pupil]   = []
                beam_trace[pupil] = []
                beam_sens[pupil]  = []
    
                for i, beam in enumerate(beam_list):
                    grp = h5.create_group(f"{pupil}/{i:04d}")
    
                    # ── arrays ────────────────────────────────────────────────
                    for section, datasets in [
                        ("direct", ["sci", "seg"]),
                        ("spec",   ["sci", "err", "x_trace", "y_trace",
                                    "lam", "dlam", "sens"]),
                    ]:
                        sgrp = grp.create_group(section)
                        for k in datasets:
                            if k in beam.__dict__[section]:
                                arr = np.asarray(beam.__dict__[section][k])
                                sgrp.create_dataset(k, data=arr,
                                                    compression="gzip", compression_opts=4)
    
                    for k in ("x", "y", "npx", "npy"):
                        if k in beam.direct:
                            grp["direct"].attrs[k] = beam.direct[k]
    
                    # ── WCS ────────────────────────────────────────────────────
                    beam_wcs[pupil].append({
                        "direct": beam.direct["wcs"],
                        "spec":   beam.spec["wcs"],
                    })
    
                    # ── metadata ──────────────────────────────────────────────
                    bm = {k: beam.meta[k]
                          for k in ("filter", "pupil", "gfile", "dfile")}
                    bm["cutout_limits"] = beam.cutout_limits
                    bm["validity"]      = beam.validity
                    bm["valid_region"]  = beam.valid_region.tolist()
                    beam_meta[pupil].append(bm)
    
                    # ── trace and sensitivity: store in full ───────────────────
                    beam_trace[pupil].append(beam.meta["trace"])
    
                    wave, sens = beam.meta["sens"]
                    beam_sens[pupil].append({
                        "wave": wave.tolist(),
                        "sens": sens.tolist(),
                    })
    
        with open(os.path.join(outdir, "beam_meta.json"), "w") as f:
            json.dump(beam_meta, f, indent=2)
    
        with open(os.path.join(outdir, "beam_wcs.pkl"), "wb") as f:
            pickle.dump(beam_wcs, f)
    
        with open(os.path.join(outdir, "beam_trace.pkl"), "wb") as f:
            pickle.dump(beam_trace, f)
    
        with open(os.path.join(outdir, "beam_sens.json"), "w") as f:
            json.dump(beam_sens, f)
    
        print(f"Galaxy {self.gid} saved to {outdir}/")

def load_galaxy(outdir):
    gal = Galaxy.__new__(Galaxy)

    # ── 1. Metadata ───────────────────────────────────────────────────────
    with open(os.path.join(outdir, "meta.json")) as f:
        meta = json.load(f)
    gal.__dict__.update(meta)
    gal.sky = SkyCoord(ra=gal.ra * u.deg, dec=gal.dec * u.deg, frame="icrs")

    # field and imgdict not available standalone; set to None
    gal.field   = None
    gal.imgdict = None

    # ── 2. Images ─────────────────────────────────────────────────────────
    gal.images = {}
    img_h5 = os.path.join(outdir, "images.h5")
    if os.path.exists(img_h5):
        with open(os.path.join(outdir, "images_wcs.pkl"), "rb") as f:
            img_wcs_raw = pickle.load(f)
        with h5py.File(img_h5, "r") as h5:
            gal.images["seg"]     = h5["seg"][:]
            gal.images["seg_wcs"] = wcs.WCS(fits.Header.fromstring(img_wcs_raw["seg"]))
            for filt in h5.keys():
                if filt == "seg":
                    continue
                gal.images[filt] = {}
                grp = h5[filt]
                for k in grp.keys():
                    gal.images[filt][k] = grp[k][:]
                for k, v in grp.attrs.items():
                    gal.images[filt][k] = v
                gal.images[filt]["wcs"] = wcs.WCS(
                    fits.Header.fromstring(img_wcs_raw[filt]))

    # ── 3. Beams ──────────────────────────────────────────────────────────
    with open(os.path.join(outdir, "beam_meta.json")) as f:
        beam_meta = json.load(f)
    with open(os.path.join(outdir, "beam_wcs.pkl"), "rb") as f:
        beam_wcs = pickle.load(f)
    with open(os.path.join(outdir, "beam_trace.pkl"), "rb") as f:
        beam_trace = pickle.load(f)
    with open(os.path.join(outdir, "beam_sens.json")) as f:
        beam_sens = json.load(f)

    gal.beams = {}
    with h5py.File(os.path.join(outdir, "beams.h5"), "r") as h5:
        for pupil in beam_meta:
            gal.beams[pupil] = []
            for i, (bm, bw, bt, bs) in enumerate(
                    zip(beam_meta[pupil], beam_wcs[pupil],
                        beam_trace[pupil], beam_sens[pupil])):

                beam = Beam()
                grp  = h5[f"{pupil}/{i:04d}"]

                # ── arrays ────────────────────────────────────────────────
                for k in ("sci", "seg"):
                    if k in grp["direct"]:
                        beam.direct[k] = grp["direct"][k][:]
                for k in ("x", "y", "npx", "npy"):
                    if k in grp["direct"].attrs:
                        beam.direct[k] = int(grp["direct"].attrs[k])
                beam.direct["wcs"] = bw["direct"]

                for k in ("sci", "err", "x_trace", "y_trace",
                          "lam", "dlam", "sens"):
                    if k in grp["spec"]:
                        beam.spec[k] = grp["spec"][k][:]
                beam.spec["wcs"] = bw["spec"]

                # ── metadata ──────────────────────────────────────────────
                beam.meta.update({k: bm[k]
                                  for k in ("filter", "pupil", "gfile", "dfile")})
                beam.meta["trace"] = bt
                beam.meta["sens"]  = (np.array(bs["wave"]), np.array(bs["sens"]))

                beam.cutout_limits = bm["cutout_limits"]
                beam.validity      = bm["validity"]
                beam.valid_region  = np.array(bm["valid_region"])

                gal.beams[pupil].append(beam)

    return gal

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
    
