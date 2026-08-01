import numpy as np
import eazy
from scipy.interpolate import interp1d

class Photometry:
    """
    Container for broadband photometry and utilities for interpolating
    spectral templates through a set of filters.

    Parameters
    ----------
    flam : (Nobj, Nfilter) array_like
        Flux densities in f_lambda units.

    eflam : (Nobj, Nfilter) array_like
        Flux uncertainties in f_lambda units.

    filters : list
        List of ``eazy.filters.FilterDefinition`` objects.

    min_err : float, optional
        Fractional systematic uncertainty added in quadrature.
    """

    C_AA = 2.99792458e18  # speed of light [Angstrom/s]

    def __init__(self,
                 flam,
                 eflam,
                 filters,
                 seg_ids,
                 min_err=0.02):
        self.seg_ids = seg_ids
        self.photom_flam = flam

        self.photom_eflam = {}
        
        for sid in self.seg_ids:
            self.photom_eflam[sid] =np.sqrt(np.asarray(eflam[sid], dtype=float)**2 +(min_err * self.photom_flam[sid])**2)

        self.filters = filters
        self.n_obj = len(self.seg_ids)
        self.n_filter = len(filters)

        self.photom_pivot = np.array(
            [filt.pivot for filt in filters]
        )

    # ============================================================
    # Template interpolation
    # ============================================================

    def interpolate_template(self, wave, flux):
        """
        Project a single template spectrum through the photometric filters.

        Parameters
        ----------
        wave : ndarray
            Wavelength array [Angstrom].

        flux : ndarray
            Flux density (f_lambda).

        Returns
        -------
        phot : ndarray
            Synthetic photometry.
        """

        phot = np.zeros(self.n_filter)

        for i, filt in enumerate(self.filters):

            phot[i] = (
                self.integrate_filter(wave, flux, filt)
                * self.C_AA
                / self.photom_pivot[i]**2
            )

        return phot

    def interpolate_templates(self, templates):
        """
        Project multiple templates through the filters.

        Parameters
        ----------
        templates : list
            List of (wave, flux) tuples.

        Returns
        -------
        A : ndarray
            Shape (Ntemplate, Nfilter)
        """

        A = np.zeros((len(templates), self.n_filter))

        for i, (wave, flux) in enumerate(templates):
            A[i] = self.interpolate_template(wave, flux)

        return A

    # ============================================================
    # Filter integration
    # ============================================================

    @staticmethod
    def integrate_filter(wave, flux, filt):
        """
        Integrate a spectrum through an EAZY filter.

        Parameters
        ----------
        wave : ndarray
            Wavelength [Angstrom].

        flux : ndarray
            Flux density (f_lambda).

        filt : eazy.filters.FilterDefinition
            Filter definition.

        Returns
        -------
        fnu : float
            Bandpass-averaged flux density.
        """

        nonzero = filt.throughput > 0

        fmin = filt.wave[nonzero].min()
        fmax = filt.wave[nonzero].max()

        if (fmin > wave.max()) or (fmax < wave.min()):
            return 0.0

        c = 2.99792458e18

        flux_fnu = flux * wave**2 / c

        template = interp1d(wave,flux_fnu,bounds_error=False, fill_value=0)(filt.wave.astype(np.float64))

        weight = filt.throughput / filt.wave

        norm = np.trapezoid(weight, filt.wave)

        return np.trapezoid(template * weight, filt.wave) / norm

    # ============================================================
    # Convenience methods
    # ============================================================

    def __len__(self):
        return self.n_obj

    def __getitem__(self, index):
        """
        Return photometry for a single object.
        """

        return (
            self.photom_flam[index],
            self.photom_eflam[index]
        )

from eazy.filters import FilterFile

def build_filter_lookup(filter_file="FILTER.RES.latest"):

    ff = FilterFile(filter_file)

    lookup = {}

    for i, filt in enumerate(ff.filters):

        name = filt.name.lower()

        # --------------------------
        # ACS
        # --------------------------

        if "acs" in name:

            if "f435w" in name: lookup["F435W"] = i+1
            if "f475w" in name: lookup["F475W"] = i+1
            if "f555w" in name: lookup["F555W"] = i+1
            if "f606w" in name: lookup["F606W"] = i+1
            if "f775w" in name: lookup["F775W"] = i+1
            if "f814w" in name: lookup["F814W"] = i+1
            if "f850lp" in name: lookup["F850LP"] = i+1

        # --------------------------
        # WFC3 IR
        # --------------------------

        if "wfc3/ir" in name:

            for f in [
                "F098M","F105W","F110W","F125W",
                "F140W","F160W",
                "F127M","F139M","F153M"
            ]:
                if f.lower() in name:
                    lookup[f] = i+1

        # --------------------------
        # WFC3 UVIS
        # --------------------------

        if "wfc3/uvis" in name:

            for f in [
                "F218W","F225W","F275W","F336W",
                "F390W","F438W","F475W",
                "F555W","F606W","F625W",
                "F775W","F814W","F850LP"
            ]:
                if f.lower() in name:
                    lookup[f] = i+1

        # --------------------------
        # JWST NIRCam
        # --------------------------

        if "nircam" in name:

            for f in [
                "F070W","F090W","F115W","F140M",
                "F150W","F150W2","F162M","F164N",
                "F182M","F187N","F200W","F210M",
                "F212N","F250M","F277W","F300M",
                "F322W2","F323N","F335M","F356W",
                "F360M","F405N","F410M","F430M",
                "F444W","F460M","F466N","F470N",
                "F480M"
            ]:
                if f.lower() in name:
                    lookup[f] = i+1

    return lookup

def build_photometry(images,
                     segmap,
                     seg_ids,
                     filters,
                     background_mask,
                     calib_error=0.05):

    # ------------------------------------
    # Build filter list once
    # ------------------------------------

    ff = eazy.filters.FilterFile('FILTER.RES.latest')

    sleuth_filters = build_filter_lookup()
    
    # for i, filt in enumerate(ff.filters):
    #     sleuth_filters[filt.name] = i
    
    pnums = [sleuth_filters[f]for f in filters]

    filt_list = [ff[p] for p in pnums]

    flux = {}
    error = {}

    
    for sid in seg_ids:
        flux[sid] = {}
        error[sid] = {}
        
        f = []
        e = []

        mask = (segmap == sid)
        npix = mask.sum()

        for img in images:

            flx = np.sum(img[mask])

            sigma = np.std(img[background_mask])

            err = np.sqrt(npix)*sigma

            f.append(max(flx,1e-22))
            e.append(err)

        f = np.asarray(f)

        e = np.sqrt(np.asarray(e)**2 +
                    (calib_error*f)**2)

        e[e<=0] = 1e-23

        # fluxes.append(f)
        # errors.append(e)
        flux[sid] = f
        error[sid] = e
        
    phot = Photometry(flam=flux, eflam=error, filters=filt_list, seg_ids=seg_ids, min_err=0.0)

    return phot