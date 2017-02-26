'''
Wrapper written by Fei Dai to get Kepler LCs.
'''

import kplr
client = kplr.API()
import matplotlib.pyplot as plt
import numpy as np

def get_kepler_lc(desig,sap,llc):
    # Find a KOI.
    #koi = client.koi(desig)
    koi = client.star(int(desig))
    #koi = client.planet(desig)
    #kois = client.kois(where="koi_period>200") find kois satisfying some conditions
    #print dir(koi)

    # This KOI has an associated star.
    #star = koi.star
    #print(star.kic_teff)

    # Download the lightcurves for this KOI.
    #lcs = koi.get_light_curves(short_cadence=False)
    sap_test = ['pdc','sap']
    lcs = koi.get_light_curves()
    if llc == 0: test = 'slc'
    if llc == 1: test = 'llc'
    #text_file = open("data/"+str(desig)+"_"+sap_test[sap]+"_"+test+".txt", "w")
    text_file = open("data/"+str(desig)+"_raw.txt", "w")
    time, flux, ferr, quality = [], [], [], []
    j=1
    for lc in lcs:
        with lc.open() as f:
            print(lc.filename)

            if test not in lc.filename:
                #j+=1
                continue
        # The lightcurve data are in the first FITS HDU.
            hdu_data = f[1].data
            #print dir(hdu_data)

            time_tmp = np.array(hdu_data["time"])
            if sap == 1:
                flux_tmp = np.array(hdu_data["sap_flux"])
                ferr_tmp = np.array(hdu_data["sap_flux_err"])
                quality_tmp = np.array(hdu_data["sap_quality"])
            if sap == 0:
                flux_tmp = np.array(hdu_data["PDCsap_flux"])
                ferr_tmp = np.array(hdu_data["PDCsap_flux_err"])
                quality_tmp = np.array(hdu_data["sap_quality"])
            good = np.where(np.isnan(flux_tmp)==0)
            time_tmp = time_tmp[good]
            flux_tmp = flux_tmp[good]
            ferr_tmp = ferr_tmp[good]
            quality_tmp = quality_tmp[good]

            #TODO: is this the quarter-by-quarter detrending?
            n =1
            a=np.polyfit(time_tmp,flux_tmp,n)
            flux_model = np.zeros_like(time_tmp)
            for k in range(n+1): flux_model = flux_model+time_tmp**(n-k)*a[k]

            flux_tmp = flux_tmp/flux_model
            ferr_tmp = ferr_tmp/np.median(flux_model)
            #flux_tmp = flux_tmp/np.median(flux_tmp)

            time +=list(time_tmp)
            flux +=list(flux_tmp)
            ferr +=list(ferr_tmp)
            quality +=list(quality_tmp)

            for i in range(len(time_tmp)):
                text_file.write(str(time_tmp[i])+'   ')
                text_file.write(str(flux_tmp[i])+'   ')
                text_file.write(str(ferr_tmp[i])+'   ')
                text_file.write(str(j)+'\n')
            j+=1
    '''
    plt.scatter(time,flux)
    plt.show()
    '''
#plt.close()


    text_file.close()

    if len(time) > 20:
        return True
    else:
        return False
#'Kepler-3b'
#get_kepler_lc('7303287',sap =0, llc = 0)#note sap ==1 use sap flux, sap ==0 use PDC flux; if llc ==0 use short cadence; if slc ==1 use long cadence
