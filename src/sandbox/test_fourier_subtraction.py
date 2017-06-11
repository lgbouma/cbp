'''
Given a normalized flux from the KEBC, try using astrobase.varbase's lcfit to
fit a fourier series to the magnitude time series and look at the residual.
'''

import numpy as np, matplotlib.pyplot as plt
import astrobase.varbase as vb
import os
from astropy.io import ascii
from numpy import array as nparr

def get_kepler_ebs_info():
    '''
    Read in the nicely formatted astropy table of the Kepler Eclipsing
    Binary Catalog information. (Includes morphology parameter, kepler
    magnitudes, whether short cadence data exists, and periods).
    See Prsa et al. (2011) and the subsequent papers in the KEBC series.
    '''

    #get Kepler EB data (e.g., the period)
    keb_path = '../data/kebc_v3_170611.csv'
    cols = 'KIC,period,period_err,bjd0,bjd0_err,morph,GLon,GLat,kmag,Teff,SC'
    cols = tuple(cols.split(','))
    tab = ascii.read(keb_path,comment='#')
    currentcols = tab.colnames
    for ix, col in enumerate(cols):
        tab.rename_column(currentcols[ix], col)

    tab.remove_column('col12') # now table has correct features

    return tab


def main():

    kebc = get_kepler_ebs_info()

    rawd = '../data/keplerebcat_LCs/data/raw/'
    ebpaths = [rawd+p for p in os.listdir(rawd)]
    ebids = [ebp.strip('.raw') for ebp in os.listdir(rawd)]

    for ix, ebpath in enumerate(ebpaths[:100]): #TODO run on everything
        ebid = ebids[ix]
        period = float(kebc[kebc['KIC']==int(ebid)]['period'])
        kmag = float(kebc[kebc['KIC']==int(ebid)]['kmag'])
        morph = float(kebc[kebc['KIC']==int(ebid)]['morph'])

        tab = ascii.read(ebpath)

        print('Read KIC {:s}; has {:d} points; period {:.3g} days.'.format(\
                ebid, len(tab), period))

        #fit a fourier series to the magnitude time series (default 8th order)
        fdict = vb.fourier_fit_magseries(
                nparr(tab['bjd']),nparr(tab['dtr_flux']),nparr(tab['dtr_err']),
                period, sigclip=4.0, plotfit='test'+str(ix)+'.png',
                ignoreinitfail=True)

if __name__ == '__main__':
    main()
