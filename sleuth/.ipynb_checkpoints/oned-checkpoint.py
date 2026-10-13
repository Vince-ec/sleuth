import numpy as np
from .templates import line_dict, Grism_template, tophat_templates, generate_1d_templates

def OneDExtraction(gx, filters):
    dith = np.linspace(-46,46,3)

    ONED_int = {}
    ONED_reg = {}
    
    for pupil in filters:
        ONED_int[pupil] = {}
        ONED_reg[pupil] = {}
        
        F = []
        E = []
        fgrid = []
        egrid = []
    
        for d in dith:
    
            cache = gx.template_cache[pupil][d]
    
            chi,A,C,okt,co = gx.Fit_Pupil(pupil,cache["templates"], 0,return_covar=True)
            template_matrix = cache["interp"]
                
            okt = np.asarray(okt)
            
            Cfull = np.zeros(okt.size)
            Cfull[okt] = C
            
            C = Cfull[len(gx.obj.beams[pupil]):]            
            co = co[len(gx.obj.beams[pupil]):]
            Fp = C @ template_matrix
            Ep = (co**2) @ (template_matrix**2)
            
            miniflux = []
            minierr = []
    
           
            for sid in  gx.seg_ids:
            
                idx = cache["template_sid"] == sid
            
                miniflux.append(C[idx] @ template_matrix[idx])
            
                minierr.append((co[idx]**2) @ (template_matrix[idx]**2))
            
    
            F.append(Fp)
            E.append(np.sqrt(Ep))
            fgrid.append(miniflux)
            egrid.append(minierr)
    
        ONED_int[pupil]['wave'] = cache["Wp"][::-1]
        ONED_int[pupil]['flux'] = np.mean(F, axis = 0)[::-1]
        ONED_int[pupil]['err'] = np.sqrt(np.mean(np.array(E)**2, axis=0))[::-1]

        for i, sid in enumerate(gx.seg_ids):
            flux = np.array([fgrid[d][i] for d in range(len(dith))])
            err = np.array([egrid[d][i] for d in range(len(dith))])

            ONED_reg[pupil][sid] = {}
            
            ONED_reg[pupil][sid]["wave"] = cache["Wp"][::-1]
            ONED_reg[pupil][sid]["flux"] = np.mean(flux, axis=0)[::-1]
            ONED_reg[pupil][sid] ["err"] = np.sqrt(np.mean(np.array(err)**2, axis=0))[::-1]
            
    return ONED_int, ONED_reg