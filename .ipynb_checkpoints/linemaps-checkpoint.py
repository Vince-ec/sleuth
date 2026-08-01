import numpy as np
import h5py
import json
import copy
from scipy.interpolate import interp1d
from astropy.wcs import WCS
from astropy import wcs
from astropy.coordinates import SkyCoord
import astropy.units as u
from drizzlepac import astrodrizzle

from .templates import Grism_template, expand_templates, line_templates,line_dict
from .blot_utils import blot_segmentation

def build_model_matrix(gx, MB, template,z):
    """
    Store a beam design matrix and metadata.

    Parameters
    ----------
    A : ndarray
        Design matrix from beam fitting.
        Shape = (npix_masked, ntemplates)

    MB : beam object

    Returns
    -------
    dict
    """
    
    rows = []

    mask = np.ravel(MB.mask)
    
    for sid in gx.seg_ids:    
        for name, spectrum in template[sid].items():


            mdl = gx.forward_model(MB,
                MB.direct["sci"] *(MB.direct["seg"] == sid),
                spectrum.redshift_spec(z))


            rows.append(mdl.ravel()[mask])

    A = np.asarray(rows)

    
    return A
    
def evaluate_model_matrix(model_matrix, shape, mask, coeff):

    A = model_matrix

    model_1d = coeff @ A

    image = np.zeros(shape)

    image[mask] = model_1d

    return image
    
def mask_slice_templates(gx, beam, templates, line_wave_obs, z):

    wave = np.arange(1e4*np.min(beam.spec['lam'])*0.9, 1e4*np.min(beam.spec['lam'])*1.1,1)
    
    dlam = interp1d(beam.spec['lam']*1e4, beam.spec['dlam'])(line_wave_obs)
    
    linemask = ((wave > (line_wave_obs - dlam/2 )) & (wave < (line_wave_obs + dlam/2 )))
    
    sliced_templates = {}
    
    for sid in gx.seg_ids:
        sliced_templates[sid] = {}
        
        for i, t in enumerate(templates[sid]):
            spectrum = templates[sid][t].redshift_spec(z)
            iflux = interp1d(spectrum[0], spectrum[1])(wave)
    
            iflux[linemask]=0
    
            sliced_templates[sid][i] = Grism_template(wave, iflux)

    return sliced_templates

def set_line_to_zero(coeffs, names, line):
    CX = np.copy(coeffs)
    flag = np.zeros_like(CX)
    
    idx =0
    for n,c in zip(names, coeffs):
        if isinstance(n,str):
            if line == n.split('_')[0]:
                flag[idx] = 1
        idx+=1         
        
    CX[flag==1] = 0

    return CX

def get_wcs_pscale(wcs):
    """
    Pixel scale (arcsec/pixel) from a WCS.
    """
    if hasattr(wcs.wcs, "cd"):
        cd = wcs.wcs.cd
    else:
        cd = wcs.wcs.pc

    pscale = np.sqrt(np.abs(np.linalg.det(cd))) * 3600.0

    wcs.pscale = pscale
    return pscale

def make_output_wcs(ra, dec, size=8.0, pixscale=0.04, theta=0.0):
    """
    Generate an output tangent-plane WCS.

    Parameters
    ----------
    ra, dec : float
        Center coordinates (deg).

    size : float
        Image size (arcsec).

    pixscale : float
        Pixel scale (arcsec/pixel).

    theta : float
        Rotation angle (deg).

    Returns
    -------
    wcs : astropy.wcs.WCS
    shape : tuple
        (ny, nx)
    """

    npix = int(np.round(size / pixscale))

    cd = pixscale / 3600.0

    t = np.deg2rad(theta)
    rot = np.array([
        [ np.cos(t), -np.sin(t)],
        [ np.sin(t),  np.cos(t)]
    ])

    cdmat = rot @ np.array([
        [-cd, 0],
        [ 0, cd]
    ])

    w = WCS(naxis=2)

    w.wcs.crpix = [(npix + 1) / 2, (npix + 1) / 2]
    w.wcs.crval = [ra, dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cunit = ['deg','deg']
    
    w.wcs.cd = cdmat

    w.array_shape = (npix, npix)
    w.pixel_shape = (npix, npix)

    w.pscale = pixscale

    return w

def get_elm_wcs(gx, MB, line_wave_obs):
    
    sky = SkyCoord(ra=gx.obj.ra* u.deg,dec=gx.obj.dec * u.deg,frame="icrs")
    
    tx,ty = MB.spec['wcs'].world_to_pixel(sky)
    
    tx += MB.cutout_limits[2]
    ty += MB.cutout_limits[0]

    diffx =MB.cutout_limits[2] - tx
    diffy =MB.cutout_limits[0] - ty
    
    linex = interp1d(MB.spec['lam']*1e4,MB.spec['x_trace']+gx.obj.pad)(line_wave_obs)
    liney = interp1d(MB.spec['x_trace']+gx.obj.pad, MB.spec['y_trace']+gx.obj.pad)(linex)
    
    recenterx =MB.cutout_limits[2] - linex
    recentery =MB.cutout_limits[0] - liney
    
    bwcs = copy.deepcopy(MB.spec["wcs"])
    bwcs.wcs.crpix[0] = bwcs.wcs.crpix[0] + diffx - recenterx
    bwcs.wcs.crpix[1] = bwcs.wcs.crpix[1] + diffy - recentery
    return bwcs

class WCSMapAll:
    """Sample class to demonstrate how to define a coordinate transformation"""

    def __init__(self, input, output, origin=0):
        """
        Initialize the class.
        Parameters
        ----------
        input : `~grizli.utils.WCSObject`
            Input WCS object.

        output : `~grizli.utils.WCSObject`
            Output WCS object.

        origin : int, optional
            Origin value.

        Attributes
        ----------
        input : `~grizli.utils.WCSObject`
            Input WCS object.

        output : `~grizli.utils.WCSObject`
            Output WCS object.

        origin : int
            Origin value.

        shift : None
            Shift attribute.

        rot : None
            Rot attribute.

        scale : None
            Scale attribute.

        """
        self.checkWCS(input, "Input")
        self.checkWCS(output, "Output")
        self.input = input
        self.output = copy.deepcopy(output)
        # self.output = output
        self.origin = 1  # origin
        self.shift = None
        self.rot = None
        self.scale = None

    def checkWCS(self, obj, name):
        """
        Check if the input object is a valid WCS object.
        
        Parameters
        ----------
        obj : `~wcs.WCS`
            The input object to be checked.
        name : str
            The name of the object.
        
        """
        try:
            assert isinstance(obj, wcs.WCS)
        except AssertionError:
            print(
                name + " object needs to be an instance or subclass of a WCS object."
            )
            raise

    def forward(self, pixx, pixy):
        """
        Transform the input pixx,pixy positions in the input frame
        to pixel positions in the output frame.

        Parameters
        ----------
        pixx : array-like
            The x-coordinates of the input pixel positions.
        pixy : array-like
            The y-coordinates of the input pixel positions.

        Returns
        -------
        result : tuple
            The transformed pixel positions in the output frame.

        """
        # This matches WTRAXY results to better than 1e-4 pixels.
        skyx, skyy = self.input.all_pix2world(pixx, pixy, self.origin)
        result = self.output.all_world2pix(skyx, skyy, self.origin)
        return result

    def backward(self, pixx, pixy):
        """
        Transform pixx,pixy positions from the output frame back onto their
        original positions in the input frame.

        Parameters
        ----------
        pixx : array-like
            The x-coordinates of the output pixel positions.

        pixy : array-like
            The y-coordinates of the output pixel positions.

        Returns
        -------
        result : tuple
            The transformed pixel positions in the input frame.
        
        """
        skyx, skyy = self.output.all_pix2world(pixx, pixy, self.origin)
        result = self.input.all_world2pix(skyx, skyy, self.origin)
        return result

    def get_pix_ratio(self):
        """
        Return the ratio of plate scales between the input and output WCS.
        This is used to properly distribute the flux in each pixel in 'tdriz'.
        """
        return self.output.pscale / self.input.pscale

    def xy2rd(self, wcs, pixx, pixy):
        """
        Transform input pixel positions into sky positions in the WCS provided.

        Parameters
        ----------
        wcs : `~wcs.WCS`
            The WCS object containing the coordinate transformation.

        pixx : array-like
            The x-coordinates of the input pixel positions.

        pixy : array-like
            The y-coordinates of the input pixel positions.

        Returns
        -------
        ra : array-like
            The right ascension (RA) values in degrees.

        dec : array-like
            The declination (Dec) values in degrees.
        
        """
        return wcs.all_pix2world(pixx, pixy, 1)

    def rd2xy(self, wcs, ra, dec):
        """
        Transform input sky positions into pixel positions in the WCS provided.

        Parameters
        ----------
        wcs : `~wcs.WCS`
            The WCS object containing the coordinate transformation.

        ra : array-like
            The right ascension (RA) values in degrees.
            
        dec : array-like
            The declination (Dec) values in degrees.

        Returns
        -------
        pixx : array-like
            The x-coordinates of the transformed pixel positions.

        pixy : array-like
            The y-coordinates of the transformed pixel positions.

        """
        return wcs.all_world2pix(ra, dec, 1)

def ELM_extract(gx, beam, model, line_wave_obs, pixfrac=1.0,kernel="point", size=8.0, pixscale=0.04):
    wcsmap = WCSMapAll
    beam_wcs = get_elm_wcs(gx,beam, line_wave_obs)
    output_wcs = make_output_wcs(gx.obj.ra, gx.obj.dec, size=size, pixscale=pixscale)
        
    shape = output_wcs.array_shape

    line = np.zeros(shape, dtype=np.float32)
    weight = np.zeros(shape, dtype=np.float32)
    ctx = np.zeros(shape, dtype=np.int32)
    
    # beam_wcs = beam.spec['wcs']
    
    if not hasattr(beam_wcs, "_naxis1"):
        beam_wcs._naxis1, beam_wcs._naxis2 = beam_wcs._naxis
    
    if not hasattr(output_wcs, "_naxis1"):
        output_wcs._naxis1, output_wcs._naxis2 = output_wcs._naxis
    
    if not hasattr(beam_wcs, "pixel_shape"):
        beam_wcs.pixel_shape = (beam_wcs._naxis1, beam_wcs._naxis2)
    
    if not hasattr(output_wcs, "pixel_shape"):
        output_wcs.pixel_shape = output_wcs.array_shape[::-1]
    
    for j in [0, 1]:
        for wcs_ext in [beam_wcs.sip]:
            if wcs_ext is not None:
                wcs_ext.crpix[j] = beam_wcs.wcs.crpix[j]
    
    beam_data =beam.spec["sci"] - model
    
    # contam_weight = np.exp(-(fcontam * np.abs(beam.contam) * np.sqrt(beam.ivar)))
    
    wht = 1.0 / beam.spec["err"]**2
    wht[~np.isfinite(wht)] = 0

    sens = interp1d(beam.spec["lam"]*1e4, beam.spec["sens"])(line_wave_obs)
    dlam = interp1d(beam.spec['lam']*1e4, beam.spec['dlam'])(line_wave_obs)
    
    sens *= 1e-17 
    sens /= np.abs(dlam)
    
    beam_data /= sens
    wht *= sens**2

    beam_data = np.nan_to_num(beam_data, nan=0.0, posinf=0.0, neginf=0.0)
    wht = np.nan_to_num(wht, nan=0.0, posinf=0.0, neginf=0.0)
    
    output_wcs.pscale = 1
    
    astrodrizzle.adrizzle.do_driz(beam_data, beam_wcs, np.float32(wht), output_wcs, line, weight,ctx,1.,'cps', 
                                1, wcslin_pscale=1, uniqid=1, 
                         pixfrac= pixfrac, kernel=kernel, fillval=0, 
                         stepsize=10, wcsmap=wcsmap)

    return {"map":line, "weight":weight, "wcs" : output_wcs}

def combine_line_maps(gx, maps, line, line_wave_obs, z, pixscale,
                      kernel, sigma_clip=True):
    """
    Combine multiple beam line maps with inverse variance weighting.

    Parameters
    ----------
    gx : Sleuth object
        Parent field/object container.

    maps : dict
        Dictionary of extracted maps organized by emission line and beam.

    line : str
        Emission line name.

    line_wave_obs : float
        Observed wavelength of emission line.

    z : float
        Redshift.

    pixscale : float
        Output pixel scale (arcsec/pixel).

    kernel : str
        Drizzle kernel.

    sigma_clip : bool
        Apply beam-to-beam sigma clipping.

    Returns
    -------
    product : dict
        Combined emission line map product.
    """

    ELMs = []
    ELMs_w = []

    SLCs = []
    SLCs_w = []

    # -------------------------------------------------
    # Collect beam products
    # -------------------------------------------------

    for i, outmaps in maps[line].items():

        line_map = np.asarray(outmaps["line"]["map"])
        lwht = np.asarray(outmaps["line"]["weight"])

        bad = (
            ~np.isfinite(line_map) |
            ~np.isfinite(lwht) |
            (lwht <= 0)
        )

        ELMs.append(np.where(bad, 0, line_map))
        ELMs_w.append(np.where(bad, 0, lwht))


        slc_map = np.asarray(outmaps["slice"]["map"])
        swht = np.asarray(outmaps["slice"]["weight"])

        bad = (
            ~np.isfinite(slc_map) |
            ~np.isfinite(swht) |
            (swht <= 0)
        )

        SLCs.append(np.where(bad, 0, slc_map))
        SLCs_w.append(np.where(bad, 0, swht))


    ELMs = np.asarray(ELMs)
    ELMs_w = np.asarray(ELMs_w)

    SLCs = np.asarray(SLCs)
    SLCs_w = np.asarray(SLCs_w)


    # -------------------------------------------------
    # Sigma clip discrepant beams
    # -------------------------------------------------

    if sigma_clip:

        valid = ELMs_w > 0

        median = np.nanmedian(
            np.where(valid, ELMs, np.nan),
            axis=0
        )

        scatter = np.nanstd(
            np.where(valid, ELMs, np.nan),
            axis=0
        )

        bad = valid & (
            np.abs(ELMs - median) > 5 * scatter
        )

        ELMs_w[bad] = 0


    # -------------------------------------------------
    # Weighted combination
    # -------------------------------------------------

    weight = np.sum(ELMs_w, axis=0)

    good = weight > 0

    combined_line = np.zeros_like(weight)

    combined_line[good] = (np.sum(ELMs * ELMs_w, axis=0)[good]/ weight[good])


    # slice map
    slice_weight = np.sum(SLCs_w, axis=0)

    combined_slice = np.zeros_like(slice_weight)

    good_slice = slice_weight > 0

    combined_slice[good_slice] = (np.sum(SLCs * SLCs_w, axis=0)[good_slice]/slice_weight[good_slice])

    # -------------------------------------------------
    # Uncertainty and quality products
    # -------------------------------------------------

    error = np.zeros_like(weight)

    error[good] = 1 / np.sqrt(weight[good])

    mask = ~good

    coverage = np.sum(ELMs_w > 0, axis=0)


    # -------------------------------------------------
    # Reference WCS
    # -------------------------------------------------
    output_wcs = next(iter(maps[line].values()))["line"]["wcs"]

    # -------------------------------------------------
    # Ancillary maps
    # -------------------------------------------------

    segmap = blot_segmentation(
        gx.ref_beam.direct["seg"],
        gx.ref_beam.direct["wcs"],
        output_wcs,
        combined_line.shape,
        fill_value=0)


    directmap = blot_segmentation(
        gx.ref_beam.direct["sci"],
        gx.ref_beam.direct["wcs"],
        output_wcs,
        combined_line.shape,
        fill_value=0)

    # -------------------------------------------------
    # Metadata
    # -------------------------------------------------

    meta = {
        "line": line,
        "rest_wavelength": line_dict[line],
        "observed_wavelength": line_wave_obs,
        "redshift": z,

        "pixel_scale": pixscale,
        "shape": combined_line.shape,
        "field_size_arcsec": [
            combined_line.shape[0] * pixscale,
            combined_line.shape[1] * pixscale,],

        "n_beams": len(maps[line]),

        "units": "erg/s/cm2/pixel",
        "flux_scaling": 1e-17,

        "drizzle_pixscale": pixscale,
        "drizzle_kernel": kernel,

        "coverage_definition":"number of contributing beam maps",}


    return {"line": combined_line, "slice": combined_slice, "weight": weight, "error": error, "coverage": coverage, 
            "mask": mask, "seg": segmap, "direct": directmap, "wcs": output_wcs, "meta": meta,}



def save_sleuth_maps(filename, gx, maps, lines, z, pixscale=0.04,
                     kernel="point"):
    """
    Save Sleuth emission line maps to HDF5.

    Structure
    ---------
    object_id/
        direct
        segmentation
        metadata

        LINE/
            flux
            slice
            error
            weight
            coverage
            mask
            wcs
            metadata

    Parameters
    ----------
    filename : str
        Output HDF5 filename.

    gx : Sleuth object
        Current galaxy/object container.

    maps : dict
        Extracted line maps organized by emission line.

    lines : list
        Lines to save.

    z : float
        Object redshift.

    pixscale : float
        Output pixel scale in arcsec/pixel.

    kernel : str
        Drizzle kernel.
    """

    with h5py.File(filename, "w") as f:

        # ---------------------------------------
        # Object group
        # ---------------------------------------

        obj = f.create_group(str(gx.obj.gid))

        # ---------------------------------------
        # Object metadata
        # ---------------------------------------

        obj_meta = {"id": str(gx.obj.gid),"ra": float(gx.obj.ra),"dec": float(gx.obj.dec),}

        obj.attrs["metadata"] = json.dumps(obj_meta)

        obj.attrs["direct_wcs"] = (gx.ref_beam.direct["wcs"].to_header_string())

        # ---------------------------------------
        # Line products
        # ---------------------------------------

        for line, outmaps in maps.items():

            print(f"Saving {line}")

            line_wave_obs = line_dict[line] * (1 + z)

            grp = obj.create_group(line)


            # -----------------------------
            # Science arrays
            # -----------------------------

            datasets = ["line","slice","error","weight","coverage","mask",]

            for key in datasets:

                grp.create_dataset( key, data=outmaps[key], compression="gzip", compression_opts=4)

            # -----------------------------
            # WCS
            # -----------------------------

            grp.attrs["wcs"] = (outmaps["wcs"].to_header_string())


            # -----------------------------
            # Metadata
            # -----------------------------

            meta = outmaps["meta"]

            # add a few guaranteed fields
            meta.update({"line": line,"rest_wavelength": float(line_dict[line]),"observed_wavelength": float(line_wave_obs),
                         "redshift": float(z),"pixel_scale": float(pixscale),"drizzle_kernel": kernel,})

            grp.attrs["metadata"] = json.dumps(meta)


def build_line_maps(gx, pupil, tdict, specz, uselines, outfile, line_dir):
    ldict = line_templates(list(line_dict.keys()),line_dir)

    maps = {line: {} for line in uselines}    

        
    for beam_id,MB in enumerate(gx.obj.beams[pupil]):
        chi2, design_matrix, coeffs, active  = gx.Fit_Beam(MB, tdict, specz) 
                
        full_coeffs = np.zeros(active.size)
        full_coeffs[active] = coeffs
        
        templates, ex_coeff, OK, names = expand_templates(tdict, full_coeffs, active, ldict)
        
        template_matrix = build_model_matrix(gx, MB, templates, specz)
    
        for line in uselines:
            Cx = set_line_to_zero(ex_coeff, names, line)
            full_model = evaluate_model_matrix(template_matrix,np.shape(MB.mask), MB.mask, Cx)
            line_wave_obs = line_dict[line]*(1+specz)
    
            sliced_templates = mask_slice_templates(gx,MB,templates, line_wave_obs, specz)
            
            sliced_template_matrix = build_model_matrix(gx, MB, sliced_templates, 0)
            
            slice_model = evaluate_model_matrix(sliced_template_matrix,np.shape(MB.mask), MB.mask, Cx)
    
            maps[line][beam_id] ={"line" : ELM_extract(gx, MB, full_model, line_wave_obs),  
                "slice" : ELM_extract(gx, MB, slice_model, line_wave_obs)  }  

    MAPS = {}
    for line in uselines:
        line_wave_obs = line_dict[line]*(1+specz)
        
        MAPS[line] = combine_line_maps(gx,maps, line, line_wave_obs,specz, pixscale=0.04, kernel='point')
    
    save_sleuth_maps(outfile, gx, MAPS, uselines, specz)
            