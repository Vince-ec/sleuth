import numpy as np
from scipy.optimize import nnls
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
import os

line_dict = {'Lya':1215.24,'CIV-1549': 1549.48, 'MgII':2799.117,'NeV-3346':3346.79 ,'NeVI-3426':3426.85,
             'OII' : 3727.092, 'NeIII-3867':3867,
             'H10' :3797.904 ,'H9' : 3835.391 ,'H8' : 3889.064 ,'H7' : 3970.079 , 'Hd' :4102.89,'Hg' : 4341.68,
             'OIII-4363' :4364.436,'Hb' : 4862.68 ,'OIII' : 5008.240, 'HeI-5877':5877,'OI-6302':6302.046,
             'Ha' :6564.61,'SII':6732.67,'ArIII-7138':7138,  'OII-7325':7325,
             'SIII' : 9533.2, 'PaD':10049.368, 'HeI-1083':10830, 'PaG': 10941.1,'PaB':12820, 'PaA':18750 }

split_line_dict = {'OI-6302':['OI-6302_a', 'OI-6302_b'], 'OIII':['OIII-4959', 'OIII-5008'], 'SIII':['SIII-9068', 'SIII-9531']}

class TemplateLibrary(object):
    """
    Continuum and emission-line template library for Sleuth.

    Parameters
    ----------
    photometry : Photometry
        Photometry object used to interpolate templates.

    z : float
        Galaxy redshift.

    template_dir : str
        Directory containing continuum templates.

    line_dir : str
        Directory containing emission-line templates.
    """
    def __init__(self,
                 photometry,
                 z,
                 template_dir,
                 line_dir):



        self.REDSHIFT_GRID = np.array([0.50,0.63,0.82,1.05,1.37,1.80,2.40,3.50])
    
        self.SSFR = [-13., -12., -11., -10., -9., -8., -7.]
        self.T50  = [0.05, 0.20, 0.50, 0.80, 0.95]
        self.ZMET = [-2., -1., -0.5, 0.0, 0.2]
        self.AV   = [0., 0.1, 0.3, 1., 2., 5.]
        
        self.phot = photometry
        self.z = z

        self.template_dir = template_dir
        self.line_dir = line_dir

        self.template_z = self.closest_redshift(z)

        self.templates = []
        self.cont_templates = []
        self.line_templates = []

    def closest_redshift(self, z):
        """
        Choose the template redshift whose cosmic age is
        closest to the galaxy age.
        """

        age = cosmo.age(z).value
        grid_age = cosmo.age(self.REDSHIFT_GRID).value

        return self.REDSHIFT_GRID[np.argmin(np.abs(grid_age-age))]

    def load_continuum(self):

        for sfh in ["w", "q"]:

            suffix = "_wR.npy" if sfh == "w" else ".npy"

            for ssfr in self.SSFR:
                for t50 in self.T50:
                    for zmet in self.ZMET:
                        for av in self.AV:

                            fn = (
                                f"{self.template_dir}/"
                                f"s{ssfr}_"
                                f"t{t50}_"
                                f"Z{zmet}_"
                                f"A{av}_"
                                f"z{self.template_z}"
                                f"{suffix}")

                            wave, flux = np.load(fn)

                            phot = self.phot.interpolate_template(wave*(1+self.z),flux)

                            self.cont_templates.append({"wave": wave,"flux": flux,"phot": phot,
                                "params": {"ssfr": ssfr,"t50": t50,"Z": zmet,"Av": av,"type": sfh,}})
                            self.templates.append({"wave": wave,"flux": flux,"phot": phot,
                                "params": {"ssfr": ssfr,"t50": t50,"Z": zmet,"Av": av,"type": sfh,}})
    def load_lines(self, lines):

        for line in lines:

            if line == "SII":
                continue

            if line == "Ha":

                ratios = np.linspace(0,0.5,11)

                for r in ratios:
                    wave, flux = np.load(f"{self.line_dir}/Ha_SII_{r}_line.npy")

                    self.line_templates.append({"name": f"Ha_{r:.2f}","wave": wave,"flux": flux,
                        "phot": self.phot.interpolate_template(wave*(1+self.z),flux)})
                    self.templates.append({"name": f"Ha_{r:.2f}","wave": wave,"flux": flux,
                        "phot": self.phot.interpolate_template(wave*(1+self.z),flux)})
            else:
                wave, flux = np.load(f"{self.line_dir}/{line}_line.npy")

                self.line_templates.append({"name": line,"wave": wave,"flux": flux,
                    "phot": self.phot.interpolate_template(wave*(1+self.z),flux)})
                self.templates.append({"name": f"Ha_{r:.2f}","wave": wave,"flux": flux,
                        "phot": self.phot.interpolate_template(wave*(1+self.z),flux)})
    @property
    def continuum_matrix(self):
        return np.array([t["phot"] for t in self.cont_templates])

    @property
    def emission_matrix(self):
        return np.array([t["phot"] for t in self.line_templates])

    @property
    def design_matrix(self):
        return np.vstack([self.continuum_matrix,self.emission_matrix])

    @property
    def continuum_params(self):
        return np.array([t["params"] for t in self.templates])
        
    def select_templates(self, phot_flux, phot_err, n_keep=10,
                     min_coeff=1e-22, expand=False):
        """
        Select continuum templates for spectral fitting using photometric NNLS.
    
        Parameters
        ----------
        phot_flux : array
            Observed photometric fluxes.
    
        phot_err : array
            Photometric uncertainties.
    
        n_keep : int
            Number of strongest continuum templates to keep.
    
        min_coeff : float
            Minimum NNLS coefficient to consider significant.
    
        expand : bool
            If True, expand selected templates with neighboring grid points.
    
        Returns
        -------
        selected : list
            Selected continuum template dictionaries.
        coeffs : array
            NNLS coefficients for selected templates.
        """
    
        # Full design matrix
        X = self.design_matrix
    
        # Weight by uncertainties
        A = (X / phot_err).T
        b = phot_flux / phot_err
    
        # Fit
        C, _ = nnls(A, b)
    
        # Remove numerical noise
        valid = np.where(C > min_coeff)[0]
    
        if len(valid) == 0:
            return [], np.array([])
    
        # Rank templates by contribution
        strongest = valid[np.argsort(C[valid])[::-1]]
    
        # Keep top templates
        keep_idx = strongest[:n_keep]
    
        selected = [
            self.templates[i]
            for i in keep_idx
        ]
        # print(selected)
        coeffs = C[keep_idx]

        rmv_lines = []
        for t in selected:
            rmv_lines.append(not 'name' in t)

        selected = np.array(selected)[np.array(rmv_lines)]
        coeffs = np.array(coeffs)[np.array(rmv_lines)]
        
        if expand:
            selected = self.expand_templates(selected)
    
        return selected, coeffs
    
    def expand_templates(self, selected):
        """
        Add neighboring templates in parameter space.
    
        Parameters
        ----------
        selected : list
            Templates selected from photometry.
    
        Returns
        -------
        expanded : list
            Expanded template list.
        """
    
        expanded = []
    
        for t in selected:
    
            p = t["params"]
    
            for temp in self.cont_templates:
    
                q = temp["params"]
    
                # Same SFH type
                if q["type"] != p["type"]:
                    continue
    
                # Count parameter differences
                dist = (
                    abs(q["ssfr"] - p["ssfr"]) +
                    abs(q["t50"] - p["t50"]) +
                    abs(q["Z"] - p["Z"]) +
                    abs(q["Av"] - p["Av"])
                )
    
                # Keep close neighbors
                if dist <= 1:
                    expanded.append(temp)
    
        # Remove duplicates
        expanded = list({
            id(t): t for t in expanded
        }.values())
    
        return expanded

class Grism_template(object):
    def __init__(self, wave, flux):
        self.wave = wave
        self.flux = flux

    def redshift_spec(self, z):
        return (self.wave * (1 + z ), self.flux / (1 + z ))

def add_lines_to_template(templates, lines, line_dir):
    for sid in templates:
        for line in lines:
            if line == "SII":
                continue
        
            if line == "Ha":
        
                ratios = np.linspace(0,0.5,11)
        
                for r in ratios:
                    wave, flux = np.load(f"{line_dir}/Ha_SII_{r}_line.npy")
                    templates[sid][f"Ha_SII_{r}"] = Grism_template(wave,flux)
            else:
                wave, flux = np.load(f"{line_dir}/{line}_line.npy")
                templates[sid][line] = Grism_template(wave,flux)

def line_templates(lines, line_dir):
    templates = {}
    for line in lines:
        if line == "SII":
            continue
    
        if line == "Ha":
    
            ratios = np.linspace(0,0.5,11)
    
            for r in ratios:
                wave, flux = np.load(f"{line_dir}/Ha_SII_{r}_line.npy")
                templates[f"Ha_SII_{r}"] = Grism_template(wave,flux)

                wave, flux = np.load(f"{line_dir}/SII_{r}_line.npy")
                templates[f"SII_{r}"] = Grism_template(wave,flux)
            
            wave, flux = np.load(f"{line_dir}/{line}_line.npy")
            templates[line] = Grism_template(wave,flux)
        
        else:
            wave, flux = np.load(f"{line_dir}/{line}_line.npy")
            templates[line] = Grism_template(wave,flux)

        # if os.path.isfile(f"{line_dir}/{line}_line_split.npy"):
        if line in split_line_dict:
            for split in split_line_dict[line]:
                wave, flux = np.load(f"{line_dir}/{split}_line_split.npy")
                templates[split] = Grism_template(wave,flux)
        
    return templates
    
def expand_templates(tdict, coeff, OK, line_templates):
    """
    Expand tied emission-line templates into individual templates.

    Parameters
    ----------
    tdict : dict
        Original fitted template dictionary.

    coeff : array
        Fitted coefficients.

    OK : array
        Template validity mask.

    line_templates : dict
        Expanded templates:
        {
          "Ha": {...},
          "OIII": {...},
          "SIII": {...}
        }

    Returns
    -------
    expanded_templates : dict
    expanded_coeff : ndarray
    expanded_OK : ndarray
    names : list
    """

    expanded_templates = {}
    expanded_coeff = []
    expanded_OK = []
    names = []

    coeff_idx = 0

    for sid in tdict:

        expanded_templates[sid] = {}

        for name, template in tdict[sid].items():

            # ------------------------
            # Halpha + SII tied template
            # ------------------------
            if isinstance(name, str) and name.startswith("Ha"):

                f = name.split("_")[2]

                expanded_templates[sid][f"Ha_{f}"] = (line_templates["Ha"])

                expanded_templates[sid][f"SII_{f}"] = (line_templates[f"SII_{f}"])

                expanded_coeff.extend([coeff[coeff_idx], coeff[coeff_idx]])

                expanded_OK.extend([OK[coeff_idx], OK[coeff_idx]])

                names.extend([f"Ha_{f}",f"SII_{f}"])


            # ------------------------
            # OIII doublet
            # ------------------------
            elif name == "OIII":

                expanded_templates[sid]["OIII-4959"] = (line_templates["OIII-4959"])

                expanded_templates[sid]["OIII-5008"] = (line_templates["OIII-5008"])

                expanded_coeff.extend([coeff[coeff_idx], coeff[coeff_idx]])

                expanded_OK.extend([OK[coeff_idx], OK[coeff_idx]])

                names.extend(["OIII-4959","OIII-5008"])


            # ------------------------
            # SIII doublet
            # ------------------------
            elif name == "SIII":

                expanded_templates[sid]["SIII-9068"] = (line_templates["SIII-9068"])

                expanded_templates[sid]["SIII-9531"] = (line_templates["SIII-9531"])

                expanded_coeff.extend([coeff[coeff_idx], coeff[coeff_idx]])

                expanded_OK.extend([OK[coeff_idx], OK[coeff_idx]])

                names.extend(["SIII-9068","SIII-9531"])


            # ------------------------
            # Normal template
            # ------------------------
            else:

                expanded_templates[sid][name] = template

                expanded_coeff.append(coeff[coeff_idx])

                expanded_OK.append(OK[coeff_idx])

                names.append(name)


            coeff_idx += 1


    return (expanded_templates,np.asarray(expanded_coeff),np.asarray(expanded_OK),names)


def tophat_templates(wmin,wmax,bandwidth,wave=None,dw=1.0,overlap=0.0,offset=0.0,normalize=True,):
    """
    Generate normalized top-hat templates.

    Parameters
    ----------
    wmin, wmax : float
        Wavelength limits.

    bandwidth : float
        Width of each top-hat.

    wave : ndarray, optional
        Wavelength grid. If None, one is generated.

    dw : float
        Wavelength spacing when generating the grid.

    overlap : float
        Fractional overlap between neighboring templates.
        0.0 = adjacent
        0.5 = 50% overlap

    offset : float
        Shift applied to the template centers.

    normalize : bool
        Normalize each template to unit integral.

    Returns
    -------
    wave : ndarray
        Wavelength grid.

    templates : dict
        Dictionary of template flux arrays.
    """

    if wave is None:
        wave = np.arange(wmin, wmax + dw, dw)

    step = bandwidth * (1.0 - overlap)

    centers = np.arange(
        wmin + bandwidth/2 + offset,
        wmax - bandwidth/2 + step + offset,
        step
    )

    templates = {}

    for i, c in enumerate(centers):

        left = c - bandwidth/2
        right = c + bandwidth/2

        flux = np.zeros_like(wave, dtype=float)

        mask = (wave >= left) & (wave < right)
        flux[mask] = 1.0

        if normalize and np.any(mask):
            flux /= np.max(flux)

        templates[i] = flux

    return wave, templates

def generate_1d_templates(gx, scl, filts, trim=1):

    wlim = {
        'F090W': [7700,10000],
        'F115W': [9700,13500],
        'F150W': [12500,17000],
        'F200W': [16000,23000]
    }

    if trim == 1:
        wtrim = {
            'F090W': [8000,10000],
            'F115W': [10000,13000],
            'F150W': [13000,16900],
            'F200W': [17500,22500]
        }
    else:
        wtrim = {
            'F090W': [8200,9800],
            'F115W': [10200,12800],
            'F150W': [13300,16500],
            'F200W': [17500,22200]
        }

    dith = np.linspace(-46,46,3)

    cache = {}

    for split in filts:

        cache[split] = {}

        ww = gx.obj.beams[split][0].spec['lam']*1e4
        IDX = np.where(
            (ww > wtrim[split][0]) &
            (ww < wtrim[split][1])
        )[0]

        Wp = ww[IDX]
        for d in dith:

            templates = {}
            interp_flux = []
            template_sid = []

            cnt = 0

            for ii, sid in enumerate(gx.seg_ids):

                if scl[ii] < 5:

                    wave, temp = tophat_templates(
                        wlim[split][0],
                        wlim[split][1],
                        bandwidth=138*scl[ii],
                        offset=d
                    )

                else:

                    wave, temp = tophat_templates(
                        wlim[split][0],
                        wlim[split][1],
                        bandwidth=(wlim[split][1]-wlim[split][0])/6,
                        offset=d
                    )

                templates[sid] = {}

                for k, flux in temp.items():

                    spec = Grism_template(wave,flux*1.e-19)

                    templates[sid][k] = spec

                    interp_flux.append(np.interp(Wp,wave,flux))
                    template_sid.append(sid)

                    cnt += 1

            cache[split][d] = {
                "templates": templates,
                "Wp": Wp,
                "IDX": IDX,
                "interp": np.asarray(interp_flux),
                "template_sid": np.asarray(template_sid),
                "count": cnt,
            }

    return cache