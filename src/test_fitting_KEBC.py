'''
Given a detrended flux from the KEBC, try using different methods to fit the
binary signal, and get the cleanest possible residuals.

The KEBC data is:
    bjd, phase, raw_flux, raw_err, corr_flux, corr_err, dtr_flux, dtr_err
with header information including the period and morphology of the EB.

After fitting the phased LC, the idea is to apply the astrobase.periodbase
periodogram routines to the residuals.

Possible methods for fitting out a contact binary's signal:
1) astrobase.varbase.lcfit's fourier_fit_magseries: fit a Fourier series to
the magnitude time series.
2) astrobase.varbase.lcfit's spline_fit_magseries: fit a univariate spline to
the magnitude time series.
3) High order Legendre polynomial ("Savitzky-Golay")
4) Armstrong et al. (2014)'s whitening procedure (bin the phased LC over a grid
that becomes iteratively finer when the LC features are sharper)
5) PHOEBE: physics-based model.
'''

import numpy as np, matplotlib.pyplot as plt
from astrobase.varbase import lcfit as lcf
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
    keb_path = '../data/kepler_eb_catalog_v3.csv'
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

        fourierdir = '../results/fourier_subtraction_diagnostics/'
        fourierpath = fourierdir + 'test'+str(ix)+'.png'
        splinedir = '../results/spline_subtraction_diagnostics/'
        splinepath = splinedir + 'test'+str(ix)+'.png'
        sgdir = '../results/savgol_subtraction_diagnostics/'
        sgpath = sgdir + 'test'+str(ix)+'.png'

        if os.path.exists(fourierpath) and os.path.exists(splinepath) \
        and os.path.exists(sgpath):

            print('Found stuff in {:d}. Continue.'.format(ix))
            continue
        else:
            tab = ascii.read(ebpath)

        print('Read KIC {:s}; has {:d} points; period {:.3g} days.'.format(\
                ebid, len(tab), period))

        #fit a fourier series to the magnitude time series (default 8th order)
        if not os.path.exists(fourierpath):
            try:
                fdict = lcf.fourier_fit_magseries(
                    nparr(tab['bjd']),
                    nparr(tab['dtr_flux']),
                    nparr(tab['dtr_err']),
                    period,
                    sigclip=6.0,
                    plotfit=fourierpath,
                    ignoreinitfail=True,
                    isnormalizedflux=True)
            except:
                print('error in {:d}. Continue.'.format(ix))

        #fit a univariate spline to the magnitude time series
        if not os.path.exists(splinepath):
            try:
                sdict = lcf.spline_fit_magseries(
                    nparr(tab['bjd']),
                    nparr(tab['dtr_flux']),
                    nparr(tab['dtr_err']),
                    period,
                    sigclip=6.0,
                    plotfit=splinepath,
                    ignoreinitfail=True,
                    isnormalizedflux=True)

                print('{:d}'.format(ix))
            except:
                print('error in {:d} (spline). Continue.'.format(ix))

        #apply a Savitzky-Golay filter to the magnitude time series
        if not os.path.exists(sgpath):
            try:
                sdict = lcf.savgol_fit_magseries(
                    nparr(tab['bjd']),
                    nparr(tab['dtr_flux']),
                    nparr(tab['dtr_err']),
                    period,
                    windowlength=None,
                    polydeg=2,
                    sigclip=6.0,
                    plotfit=sgpath,
                    isnormalizedflux=True)
                print('{:d}'.format(ix))

            except:
                print('error in {:d} (savgol). Continue.'.format(ix))

    print('\nDone testing fits.')

if __name__ == '__main__':
    main()
