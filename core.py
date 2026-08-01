import numpy as np
import copy

from sklearn.neighbors import NearestNeighbors
from astropy.wcs import wcs
from sklearn.preprocessing import StandardScaler
import scipy
from scipy.special import huber

from .blot_utils import blot_segmentation, blot_direct_image
from .photometry import build_photometry
from .templates import Grism_template
from .oned import OneDExtraction
from .linemaps import build_line_maps

import jax.numpy as jnp
import jax

class Sleuth(object):
    def __init__(self, obj, msk_min = 0.1 ):
        self.obj = obj
        self.msk_min = msk_min

        self.available_filters = set(self.obj.beams.keys())

        self.ref_beam = next(iter(self.obj.beams.values()))[0]

        self.Bkgseg = self.ref_beam.direct['seg'] == 0
    
    def clean_directs(self,):
        ref_wcs = copy.deepcopy(self.ref_beam.direct['wcs'])
        ref_wcs.pscale = 1

        for p in self.obj.beams:
            mini_img = []
            for bm in self.obj.beams[p]:
                inwcs = copy.deepcopy(bm.direct['wcs'])
                inwcs.pscale = 1
                outimg = blot_direct_image(bm.direct['sci'], inwcs, ref_wcs)
                mini_img.append(outimg)
                
            mini_img = np.array(mini_img, dtype=float)
            mini_img[mini_img == 0] = np.nan
            clean_img = np.nanmedian(mini_img, axis = 0).astype(np.float32)
            clean_img[np.isnan(clean_img)] = 0
            for bm in self.obj.beams[p]:
                inwcs = copy.deepcopy(bm.direct['wcs'])
                inwcs.pscale = 1
                outimg = blot_direct_image(clean_img, ref_wcs, inwcs)
                norm = np.max(bm.direct['sci'][bm.direct['seg'] == self.obj.gid]) / np.max(outimg[bm.direct['seg'] == self.obj.gid])
                bm.direct['sci'] = outimg*norm     

    def load_images(self, img_source, drizzle_loss_factor = 356):
        self.images = {}
        self.img_source = img_source
        
        if self.img_source == 'native':        
            for p in self.obj.beams:
                self.images[p] = {}
                
                mini_img = []
                for bm in self.obj.beams[p]:
                    mini_img.append(bm.direct['sci'])

                mini_img = np.array(mini_img, dtype=float)
                mini_img[mini_img == 0] = np.nan
                
                self.images[p]['img'] = np.nanmedian(mini_img, axis=0)       
                # add pivot
                
        elif self.img_source == 'reference':
            ref_wcs = copy.deepcopy(self.ref_beam.direct['wcs'])
            ref_wcs.pscale = 1
            
            for filt in self.obj.images:
                self.images[filt] = {}
                inwcs = copy.deepcopy(self.obj.images[filt]['wcs'])
                inwcs.pscale = 1
                outimg = blot_direct_image(self.obj.images[filt]['sci'], inwcs, ref_wcs)
                norm = np.max(self.obj.images[filt]['sci'][self.ref_beam.direct['seg'] == self.obj.gid]) /np.max(outimg[self.ref_beam.direct['seg'] == self.obj.gid])

                self.images[filt]['img'] = outimg*drizzle_loss_factor * norm
                self.images[filt]['pivot'] = self.obj.images[filt]['pivot']

    def prepare_foward_model(self, chunk_size = 20):

        for p in self.obj.beams:
            for beam in self.obj.beams[p]:
                beam.spec['disp'] = DispersionGeometry(beam.direct["sci"].shape,beam.spec['x_trace'], beam.spec['y_trace'], chunk_size=chunk_size)
                beam.spec['disp'].build_scatter_cache()
        
    def prepare_segmentation(self, sigma=3e-23, segmentation_filter=None):
        """
        Prepare the images needed for resegmentation.
    
        Parameters
        ----------
        sigma : float
            Constant noise level (temporary).
        detection_filter : str or None
            Preferred filter to use as the detection image. If None,
            the reddest available image is used.
        """
    
        # ----------------------------------------
        # Choose the detection image
        # ----------------------------------------
    
        if segmentation_filter is None:
    
            priority = ["F150W","F115W","F200W","F090W","F277W","F356W", "F410M","F444W"]
    
            for filt in priority:
                if filt in self.images:
                    segmentation_filter = filt
                    break
    
            if segmentation_filter is None:
                segmentation_filter = list(self.images.keys())[0]
    
        self.segmentation_image = self.images[segmentation_filter]['img']
    
        # ----------------------------------------
        # Temporary variance map
        # ----------------------------------------
    
        self.var_map = np.ones_like(self.segmentation_image) * sigma**2
    
        sigma = np.sqrt(self.var_map)
    
        
        # ----------------------------------------
        # luptitude images
        # ----------------------------------------
    
        self.lup_images = {}
    
        for filt in self.images:
            self.lup_images[filt] = self.flux_to_luptitude(self.images[filt]['img'], sigma)
    
        # ----------------------------------------
        # Color maps
        # ----------------------------------------
    
        self.color_maps = {}
    
        filters = list(self.lup_images.keys())
    
        for i in range(len(filters)):
            for j in range(i + 1, len(filters)):
    
                f1 = filters[i]
                f2 = filters[j]
    
                self.color_maps[(f1, f2)] = (self.lup_images[f1] - self.lup_images[f2])


    def build_features(self, mode='color', image = None):
        ref_seg = (self.ref_beam.direct["seg"] == self.obj.gid)
    
        # -------------------------------------------------
        # Pixel coordinates
        # -------------------------------------------------
    
        yy, xx = np.where(ref_seg)
    
        # -------------------------------------------------
        # Color feature space
        # -------------------------------------------------
    
        if mode == 'color':
            feature_list = []
    
            self.feature_names = []
    
            for key, cmap in self.color_maps.items():
    
                feature_list.append(cmap[ref_seg])
    
                self.feature_names.append(key)
    
            features = np.array(feature_list).T
    
    
        # -------------------------------------------------
        # Flux feature space
        # -------------------------------------------------
    
        elif mode == 'flux':
            if image is None:
                raise ValueError( "Flux mode requires an input image.")

            flux = image[galaxy_mask]

            # use log flux because the dynamic range
            # is much smaller and behaves more like colors
    
            floor = np.nanpercentile(flux[flux > 0],1) * 1e-3

            features = np.log10(flux + floor)[:,None]
    
            self.feature_names = ["log_flux"]

        else:
            raise ValueError("mode must be 'color' or 'flux'")
    
    
        # -------------------------------------------------
        # Remove bad pixels
        # -------------------------------------------------
        good = np.all(np.isfinite(features),axis=1)
    
        features = features[good]
    
        self.feature_pixels = np.column_stack([yy[good],xx[good]])

        # -------------------------------------------------
        # Standardize feature space
        # -------------------------------------------------
        self.scaler = StandardScaler()
    
        self.features = self.scaler.fit_transform(features)

    
    @staticmethod
    def flux_to_luptitude(flux, sigma):    
        b = sigma
    
        return -(2.5 / np.log(10.0)) * np.arcsinh(flux / (2.0 * sigma))    
                        
    def reset_seg(self,):
        self.Nseg = (self.ref_beam.direct["seg"] >0) * self.obj.gid
        self.seg_ids = [self.obj.gid]

    def single_seg(self,):
        self.Nseg = (self.ref_beam.direct["seg"] == self.obj.gid) * 1
        self.seg_ids = [1]
    
    def segment(self, method = 'color', limit=100, image = None):
        """
        Runs segmentation method
        for color limit refers to SNR limit; for flux, limit refers to flux limit
        """
        
        if method == "color":
            self.build_features()
        
            self.color_segment(limit)
        
        elif method == "flux":
            self.build_features(image = image)
        
            self.flux_segment(image, limit)
            
        self.seg_ids = np.unique(self.Nseg)[1:]

    def color_segment(self, snr_limit=100):
        """
        Segment galaxy into self-similar regions using
        nearest-neighbor color similarity.
        """
        features = self.features
        coords = self.feature_pixels
    
        # ----------------------------------
        # nearest neighbor tree
        # ----------------------------------
    
        nbrs = NearestNeighbors(
            n_neighbors=len(features),
            algorithm='ball_tree')
    
        nbrs.fit(features)

        # ----------------------------------
        # pixels still available
        # ----------------------------------
    
        remaining = np.ones(len(features),dtype=bool)
        regions = []
    
        # ----------------------------------
        # Grow regions
        # ----------------------------------
    
        while np.any(remaining):
            available = np.where(remaining)[0]
    
            # brightest remaining pixel
            flux = self.segmentation_image[coords[available,0],coords[available,1]]
    
            seed = available[np.argmax(flux)]
    
            # nearest pixels in color space
            dist, ind = nbrs.kneighbors(features[seed].reshape(1,-1),n_neighbors=len(features))
    
            region = []
            for idx in ind[0]:
                if not remaining[idx]:
                    continue
    
                region.append(idx)
    
                # compute S/N
    
                pix = coords[region]
    
                flux = np.sum(self.segmentation_image[pix[:,0],pix[:,1]])

                #temporary
                # noise = np.sqrt(np.sum(self.segmentation_image.var[pix[:,0],pix[:,1]]))
                noise = np.sqrt(np.sum(self.var_map[pix[:,0],pix[:,1]]))
    
                snr = flux/noise
                if snr >= snr_limit:
                    break

            # if the entire remaining galaxy is below threshold
            if len(region) == 0:
                break

            regions.append( np.array(region))
            # remove assigned pixels
            remaining[np.array(region)] = False
    
        # ----------------------------------
        # Create segmentation image
        # ----------------------------------
        Nseg = np.zeros(self.segmentation_image.shape,dtype=int)
    
        for i, region in enumerate(regions):
            pix = coords[region]
    
            Nseg[pix[:,0],pix[:,1]] = i + 1
    
        self.Nseg = Nseg
    
        self.nregions = len(regions)
                
    def flux_segment(self, image, flux_limit):
    
        """
        Segment galaxy using nearest-neighbor similarity,
        growing regions until reaching a flux threshold.
    
        Parameters
        ----------
        image : ndarray
            Image used to determine integrated flux.
    
        flux_limit : float
            Minimum integrated flux per region.
        """
        features = self.features
        coords = self.feature_pixels
    
        # ---------------------------------------
        # Build nearest neighbor tree
        # ---------------------------------------
    
        nbrs = NearestNeighbors(n_neighbors=len(features),algorithm='ball_tree')
    
        nbrs.fit(features)

        remaining = np.ones(len(features),dtype=bool)
        regions = []
    
        # ---------------------------------------
        # Region growth
        # ---------------------------------------
    
        while np.any(remaining):
            available = np.where(remaining)[0]
    
            # brightest remaining pixel
    
            flux = image[coords[available,0],coords[available,1]]
    
            seed = available[np.argmax(flux)]
    
            # nearest neighbors in feature space
    
            _, indices = nbrs.kneighbors(features[seed].reshape(1,-1),n_neighbors=len(features))
        
            region = []
        
            for idx in indices[0]:
    
                if not remaining[idx]:
                    continue
    
                region.append(idx)
    
                pix = coords[region]
    
                total_flux = np.sum(image[pix[:,0],pix[:,1]])
    
                if total_flux >= flux_limit:
                    break
    
            regions.append(np.array(region))
    
            remaining[np.array(region)] = False
    
        # ---------------------------------------
        # Create segmentation image
        # ---------------------------------------
    
        Nseg = np.zeros_like(image,dtype=int)
    
        for i, region in enumerate(regions):
            pix = coords[region]
    
            Nseg[pix[:,0],pix[:,1]] = i + 1
    
        self.Nseg = Nseg
    
        self.nregions = len(regions)

                    
    def reseg(self):
        ref_wcs = self.ref_beam.direct["wcs"]
        
        for pupil in self.obj.beams:
            for beam in self.obj.beams[pupil]:
                beam.direct["seg"] = blot_segmentation(self.Nseg,ref_wcs,beam.direct["wcs"],beam.direct["sci"].shape)
                
    # def auto_mask(self, split):
    
    #     stk_c = []
    #     stk_r = []
    #     m_c = []
    #     m_r = []    
    #     c_c = []
    #     c_r = []    

    #     for b in self.BD[split].beams:
    #         if b.grism.filter == 'GR150C':
    #             stk_c.append(b.grism.data['SCI'] - b.contam - b.model)
    #             m_c.append(b.model)
    #             c_c.append(b.contam)

    #         if b.grism.filter == 'GR150R':
    #             stk_r.append(b.grism.data['SCI'] - b.contam - b.model)
    #             m_r.append(b.model)
    #             c_r.append(b.contam)

    #     if len(stk_r) > 0:
    #         Rmsk = (np.median(stk_r, axis=0)>0.05)*(np.median(c_r, axis=0)>.1)
    #         Rmsk += (np.median(stk_r, axis=0)*(~Rmsk)>0.05)*(~(np.median(m_r, axis=0)>.01))
    #         Rmsk = binary_dilation(Rmsk,iterations=2)

    #     if len(stk_c) > 0:
    #         Cmsk = (np.median(stk_c, axis=0)>0.05)*(np.median(c_c, axis=0)>.1)
    #         Cmsk += (np.median(stk_c, axis=0)*(~Cmsk)>0.05)*(~(np.median(m_c, axis=0)>.01))
    #         Cmsk = binary_dilation(Cmsk,iterations=2)

    #     scif = []
    #     for b in self.BD[split].beams:
    #         if b.grism.filter == 'GR150C':
    #             b.grism.data['SCI'][Cmsk] = 0
    #             b.scif = np.ravel(b.grism.data['SCI'] - b.contam)

    #         if b.grism.filter == 'GR150R':
    #             b.grism.data['SCI'][Rmsk] = 0
    #             b.scif = np.ravel(b.grism.data['SCI'] - b.contam)

    #         scif.extend(b.scif)

    #     self.BD[split].scif = scif


    def gen_mask(self,):
        for pupil in self.obj.beams:
            for beam in self.obj.beams[pupil]:
                # -----------------------------------
                # Generate morphological trace mask
                # -----------------------------------                
                mdl = self.forward_model(beam, beam.direct['sci'] * (beam.direct['seg'] > 0),
                                              [np.linspace(5000, 30000), np.ones(50)])
    
                mdl /= np.nanmax(mdl)

                beam.spec["trace_mask"] = (mdl > self.msk_min)                

                # -----------------------------------
                # Detector validity mask
                # -----------------------------------
                
                d = beam.spec['sci']
    
                d[np.isnan(d)] = 0
                
                unique_values, counts = np.unique(d, return_counts=True)
                rpts = unique_values[counts > 1]
                rmask = np.ones_like(d)
    
                for r in rpts:
                    rmask[beam.spec['sci'] == r] = 0
                
                sci = beam.spec["sci"].copy()
    
                valid = np.isfinite(sci)
    
                # remove empty pixels

                valid &= rmask == 1
                valid &= sci != 0
                valid &= (beam.spec['err'] > 0)
    
                beam.spec["valid_mask"] = valid

                # -----------------------------------
                # Final extraction mask
                # -----------------------------------
                beam.mask = (beam.spec["trace_mask"] & beam.spec["valid_mask"])      
    
    def build_flats(self):
        self.flats = {}
        for pupil in self.obj.beams:
    
            flat_sci = []
            flat_err = []
            flat_mask = []
            mslices = []
            # beam_id = []

            s = 0
            for i, beam in enumerate(self.obj.beams[pupil]):
                beam.flat = {}
                # flatten beam
                beam.flat["sci"] = beam.spec["sci"].ravel()
                beam.flat["err"]= beam.spec["err"].ravel()
                beam.flat["mask"] = beam.mask.ravel()
                
                flat_sci.append(beam.flat["sci"])
                flat_err.append(beam.flat["err"])
                flat_mask.append(beam.flat["mask"])
    
                # beam_id.extend(np.ones(sci.size,dtype=int) * i)
                
                # -----------------------------------
                # generate mslices
                # -----------------------------------
                
                mslices.append(np.arange(s, s+np.sum(beam.flat["mask"] > 0)))
                s += np.sum(beam.flat["mask"] > 0)
            # concatenate all beams in pupil
    
            self.flats[pupil] = {"sci":np.concatenate(flat_sci),
                "err": np.concatenate(flat_err),
                "mask": np.concatenate(flat_mask),
                # "beam_id": np.array(beam_id),
                "mslices":mslices}

    def forward_model(self,beam, galaxy_image, spectra):   
        l = beam.cutout_limits
        inimg = jnp.zeros([l[1] - self.obj.pad, l[3] - self.obj.pad])
        
        out = disperse_obj_cached(galaxy_image * (beam.direct['seg']>0), beam.spec['disp'],  beam.spec['sens'],
                          beam.spec['lam'], spectra, inimg)

        return out[l[0] - self.obj.pad: l[1] - self.obj.pad, l[2] - self.obj.pad:l[3] - self.obj.pad]


    def standard_config(self, snr_limit):
        #load images
        self.clean_directs()
        self.load_images('reference')
        
        #initialize spectra
        self.single_seg()
        self.reseg()
        self.prepare_foward_model()
        self.gen_mask()
        self.build_flats()
        
        #color segment
        self.prepare_segmentation()
        self.reset_seg()
        self.reseg()
        self.segment(limit = snr_limit)
        self.reseg()
        
        #get phot
        self.extract_phot()
    
    def extract_phot(self):
        self.phot = build_photometry([self.images[filt]['img'] for filt in self.images],self.Nseg,self.seg_ids,
                       [filt for filt in self.images],self.Bkgseg)
    
    def Fit_Pupil(self, pupil, temp, z, return_covar=False):

        flats = self.flats[pupil]
    
        sci = flats["sci"]
        err = flats["err"]
        mask = flats["mask"]
    
        beams = self.obj.beams[pupil]
    
        Nbeam = len(beams)
        Npix = np.sum(mask)
    
        pedestal = 0.04
    
        # --------------------------------------------------
        # Background basis
        # --------------------------------------------------
    
        A_bg = np.zeros((Nbeam, Npix))
    
        for i, sl in enumerate(flats["mslices"]):
            A_bg[i, sl] = 1.0
    
        # --------------------------------------------------
        # Spectral templates
        # --------------------------------------------------
    
        rows = []
    
        for sid in self.seg_ids:    
            for name, spectrum in temp[sid].items():
    
                dispersed = []
    
                for beam in beams:
                    mdl = self.forward_model(beam,
                        beam.direct["sci"] *(beam.direct["seg"] == sid),
                        spectrum.redshift_spec(z))
    
                    dispersed.append(mdl.ravel())
    
                rows.append(np.concatenate(dispersed)[mask])
    
        A_spec = np.asarray(rows)
    
        A = np.vstack([A_bg, A_spec])
    
        if return_covar:
    
            chi2, coeffs, model, oktemp, covar = self._fit_matrix(A,sci,err,mask,return_covar=True,pedestal=pedestal,)
    
            return (chi2 / Nbeam,A_spec,coeffs,oktemp,covar,)
    
        chi2, coeffs, model, oktemp = self._fit_matrix(A,sci,err,mask,pedestal=pedestal,)
    
        return (chi2 / Nbeam,A_spec,coeffs,oktemp)
    

    def Fit_Beam(self, beam, temp, z, return_covar=False):
    
        sci = beam.flat["sci"]
        err = beam.flat["err"]
        mask = beam.flat["mask"]
    
        rows = []
    
        for sid in self.seg_ids:
    
            seg = beam.direct["seg"] == sid
    
            for name, spectrum in temp[sid].items():    
                mdl = self.forward_model(beam,
                    beam.direct["sci"] *(beam.direct["seg"] == sid),
                    spectrum.redshift_spec(z))
    
                rows.append(mdl.ravel()[mask])
    
        A = np.asarray(rows)
    
        if return_covar:
            chi2, coeffs, model, oktemp, covar = self._fit_matrix(A,sci,err,mask,return_covar=True,)
    
            return (chi2,A,coeffs,oktemp,covar,)
    
        chi2, coeffs, model, oktemp = self._fit_matrix(A,sci,err,mask,)
    
        return (chi2,A,coeffs,oktemp,)
        
        
    def _fit_matrix(self,A,sci,err,mask,return_covar=False,pedestal=0.0,huber_delta=4,):
        """
        Generic NNLS fitting routine.
    
        Parameters
        ----------
        A : ndarray
            Design matrix (Nmodel, Npix)
    
        sci : ndarray
            Flattened science array.
    
        err : ndarray
            Flattened uncertainty array.
    
        mask : ndarray (bool)
            Pixels used in the fit.
    
        Returns
        -------
        chi2
        coeffs
        model
        oktemp
        covar (optional)
        """
    
        ivar = 1.0 / err
    
        oktemp = A.sum(axis=1) != 0
    
        Ax = A[oktemp] * ivar[mask]
    
        data = (sci + pedestal)[mask] * ivar[mask]
    
        coeffs, _ = scipy.optimize.nnls(Ax.T,data)
    
        model = coeffs @ (A[oktemp] / ivar[mask])
    
        resid = sci[mask] - model
    
        norm_resid = resid * ivar[mask]
    
        chi2 = 2.0 * huber(huber_delta,norm_resid[np.isfinite(norm_resid)])
    
        if not return_covar:
            return (np.sum(chi2),coeffs,model,oktemp,)
    
        cov = safe_invert(Ax @ Ax.T)
        cov = fill_masked_covar(cov, oktemp)
        cov_err = np.sqrt(np.diag(cov))
    
        return (np.sum(chi2),coeffs,model,oktemp,cov_err,)

    def OneD(self, pupils):
        self.OneD_int, self.OneD_reg = OneDExtraction(self, pupils)    

    def ELMap_extract(self,pupil, tdict, specz, uselines, outfile, line_dir):
        build_line_maps(self, pupil, tdict, specz, uselines, outfile, line_dir)
        
    def reconstruct_model(self, beam, coeffs, oktemp, temp, z):
        """
        Reconstruct the best-fit dispersed model for a single beam.
    
        Parameters
        ----------
        beam : SleuthBeam
        coeffs : ndarray
            NNLS coefficients.
        oktemp : ndarray(bool)
            Valid template mask returned by Fit_Beam.
        temp : dict
            Template library.
        z : float
    
        Returns
        -------
        model : 2D ndarray
            Best-fit dispersed spectrum.
        """
    
        model = np.zeros_like(beam.spec["sci"], dtype=float)
    
        j = 0
        k = 0
    
        for sid in self.seg_ids:
    
            seg = (beam.direct["seg"] == sid)
    
            for name, template in temp[sid].items():
    
                if oktemp[k]:
    
                    mdl = self.forward_model(beam,
                        beam.direct["sci"] *(beam.direct["seg"] == sid),
                        spectrum.redshift_spec(z))
  
                    model += coeffs[j] * mdl
    
                    j += 1
    
                k += 1
    
        return model

class DispersionGeometry:

    def __init__(self,
                 image_shape,
                 x_trace,
                 y_trace,
                 chunk_size=128,
                 mask=None):

        Ny, Nx = image_shape

        cy = (Ny - 1) / 2.0
        cx = (Nx - 1) / 2.0

        rows, cols = np.meshgrid(
            np.arange(Ny),
            np.arange(Nx),
            indexing="ij"
        )

        if mask is None:
            keep = np.ones(rows.size, dtype=bool)
        else:
            keep = mask.ravel() != 0

        self.di = jnp.asarray((cols.ravel()[keep] - cx), dtype=jnp.float32)
        self.dj = jnp.asarray((rows.ravel()[keep] - cy), dtype=jnp.float32)
        self.pixel_index = np.where(keep)[0]

        self.chunk_size = chunk_size

        Nlam = len(x_trace)

        self.n_chunks = (Nlam + chunk_size - 1) // chunk_size
        self.pad = self.n_chunks * chunk_size - Nlam

        self.x_trace = jnp.pad(jnp.asarray(x_trace), (0, self.pad))
        self.y_trace = jnp.pad(jnp.asarray(y_trace), (0, self.pad))

    def build_scatter_cache(self):

        self.scatter = []
    
        for c in range(self.n_chunks):
    
            start = c * self.chunk_size
    
            xt = jax.lax.dynamic_slice(
                self.x_trace,
                (start,),
                (self.chunk_size,)
            )
    
            yt = jax.lax.dynamic_slice(
                self.y_trace,
                (start,),
                (self.chunk_size,)
            )
    
            x = self.di[:, None] + xt[None, :]
            y = self.dj[:, None] + yt[None, :]
    
            x0 = jnp.floor(x).astype(jnp.int32)
            y0 = jnp.floor(y).astype(jnp.int32)
    
            fx = x - x0
            fy = y - y0
    
            self.scatter.append({
    
                "x0": x0.ravel(),
                "y0": y0.ravel(),
    
                "w00": ((1-fx)*(1-fy)).ravel(),
                "w10": (fx*(1-fy)).ravel(),
                "w01": ((1-fx)*fy).ravel(),
                "w11": (fx*fy).ravel(),})

def bilinear_scatter_cached(output,
                            scatter,
                            values):

    kw = dict(
        mode="drop",
        wrap_negative_indices=False
    )

    x0 = scatter["x0"]
    y0 = scatter["y0"]

    output = output.at[
        y0,
        x0
    ].add(values*scatter["w00"], **kw)

    output = output.at[
        y0,
        x0+1
    ].add(values*scatter["w10"], **kw)

    output = output.at[
        y0+1,
        x0
    ].add(values*scatter["w01"], **kw)

    output = output.at[
        y0+1,
        x0+1
    ].add(values*scatter["w11"], **kw)

    return output

def disperse_obj_cached(image, geometry,sens, lam, spectra,output):
    
    inspc = jnp.interp(lam, (spectra[0])*1e-4, spectra[1])

    fl = image.ravel()[geometry.pixel_index]

    for c in range(geometry.n_chunks):

        start = c * geometry.chunk_size

        s = jax.lax.dynamic_slice(
            jnp.pad(sens*inspc, (0, geometry.pad)),
            (start,),
            (geometry.chunk_size,)
        )

        vals = (fl[:, None] * s[None, :]).ravel()

        output = bilinear_scatter_cached(
            output,
            geometry.scatter[c],
            vals
        )

    return output


def safe_invert(A):
    """
    Safely invert a matrix, falling back to the Moore-Penrose
    pseudoinverse if the matrix is singular.
    """
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)

def fill_masked_covar(cov, mask):
    """
    Expand a covariance matrix corresponding to the True entries
    of `mask` back to the full parameter space.

    Parameters
    ----------
    cov : (Ngood, Ngood) ndarray
        Covariance matrix for surviving parameters.

    mask : (N,) bool array
        Boolean mask indicating which parameters were fitted.

    Returns
    -------
    full_cov : (N, N) ndarray
    """
    mask = np.asarray(mask, dtype=bool)

    full = np.zeros((mask.size, mask.size), dtype=cov.dtype)

    idx = np.where(mask)[0]

    full[np.ix_(idx, idx)] = cov

    return full