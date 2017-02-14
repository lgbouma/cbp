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
3) astrobase.varbase.lcfit's savgol_fit_magseries: apply a Savitzky-Golay
filter to the magnitude time series.
4) High order Legendre polynomial.

Any of 1-4, iteratively.

5) Armstrong et al. (2014)'s whitening procedure (bin the phased LC over a grid
that becomes iteratively finer when the LC features are sharper)
6) PHOEBE: physics-based model.
'''

import numpy as np, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path
import matplotlib.patches as patches
from astrobase.varbase import lcfit as lcf
import os
from astropy.io import ascii
from numpy import array as nparr
import numpy.ma as ma


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


def _fits(times, mags, errs, ix, period, fourierpath, splinepath, sgpath,
        diagnosticplots=True):
    '''
    Run the fitting subroutines.

    Args:
        diagnosticplots (bool): whether to use the given paths (fourierpath,
        splinepath, sgpath, etc) to write diagnostic plots. If False, the
        returned dicts can still be used (to make better plots, say with
        residuals).
    '''

    if not diagnosticplots:
        fourierpath, splinepath, sgpath = False, False, False

    #fit a fourier series to the magnitude time series (default 8th order)
    if not os.path.exists(fourierpath) or not diagnosticplots:
        try:
            fdict = lcf.fourier_fit_magseries(
                times,mags,errs,
                period,
                sigclip=6.0,
                plotfit=fourierpath,
                ignoreinitfail=True,
                isnormalizedflux=True)
            print(fdict.keys())
        except:
            print('error in {:d}. Continue.'.format(ix))

    #fit a univariate spline to the magnitude time series
    if not os.path.exists(splinepath) or not diagnosticplots:
        try:
            spdict = lcf.spline_fit_magseries(
                times,mags,errs,
                period,
                sigclip=6.0,
                plotfit=splinepath,
                ignoreinitfail=True,
                isnormalizedflux=True)

            print('{:d}'.format(ix))
        except:
            print('error in {:d} (spline). Continue.'.format(ix))

    #apply a Savitzky-Golay filter to the magnitude time series
    find = np.isfinite(times) & np.isfinite(mags) & np.isfinite(errs)
    fmags = mags[find]
    polydeg = 2
    windowlength = max(
            polydeg+3,
            int(len(fmags)/200) # window over 1/200th of length of thing.
            )
    if windowlength % 2 == 0:
        windowlength += 1

    sgdicts = []
    for winlen in [windowlength, 2*windowlength+1, 4*windowlength+1]:
        if not os.path.exists(sgpath) or not diagnosticplots:
            try:
                sgdict = lcf.savgol_fit_magseries(
                    times,mags,errs,
                    period,
                    windowlength=winlen,
                    polydeg=polydeg,
                    sigclip=6.0,
                    plotfit=sgpath,
                    isnormalizedflux=True)
                print('{:d}'.format(ix))

                sgdicts.append(sgdict)

            except:
                print('error in {:d} (savgol). Continue.'.format(ix))

    return [fdict, spdict, sgdicts[0], sgdicts[1], sgdicts[2]]


def _residual_plots(dicts, titleinfo):

    #TODO: could add a "fitdiagnostics" dictionary inside each dictionary in
    #dicts. This would keep track of fourierorder/nknots/windowlength/polydeg
    #etc as appropriate, so that when iteratively fitting + reducing, we'd
    #be able to keep track of what operations we're doing.

    #fdict, sdict, sgdict = dicts[0], dicts[1], dicts[2]
    dtitles= [
        'fourier series. order: {:d}\n$\chi^2$: {:.3g}, red$\chi^2$: {:.3g}'.format(
        int(dicts[0]['fitinfo']['fourierorder']), dicts[0]['fitchisq'], dicts[0]['fitredchisq']),
        'univariate spline. knots: {:d}\n$\chi^2$: {:.3g}, red$\chi^2$: {:.3g}'.format(
        int(dicts[1]['fitinfo']['nknots']), dicts[1]['fitchisq'], dicts[1]['fitredchisq']),
        'savitzky-golay. windowlen: {:d}\npolydeg: {:d}, $\chi^2$: {:.3g}, red$\chi^2$: {:.3g}'.format(
        dicts[2]['fitinfo']['windowlength'], dicts[2]['fitinfo']['polydeg'],dicts[2]['fitchisq'],dicts[2]['fitredchisq']),
        'savitzky-golay. windowlen: {:d}\npolydeg: {:d}, $\chi^2$: {:.3g}, red$\chi^2$: {:.3g}'.format(
        dicts[3]['fitinfo']['windowlength'], dicts[3]['fitinfo']['polydeg'],dicts[3]['fitchisq'],dicts[3]['fitredchisq']),
        'savitzky-golay. windowlen: {:d}\npolydeg: {:d}, $\chi^2$: {:.3g}, red$\chi^2$: {:.3g}'.format(
        dicts[4]['fitinfo']['windowlength'], dicts[4]['fitinfo']['polydeg'],dicts[4]['fitchisq'],dicts[4]['fitredchisq'])
        ]

    plt.close('all')
    f, axs = plt.subplots(figsize=(len(dicts)*4,8), nrows=2, ncols=len(dicts),
            sharex=True)

    for ix, thisdict in enumerate(dicts):

        tms = thisdict['magseries']
        tfi = thisdict['fitinfo']
        axs[0,ix].errorbar(tms['phase'],tms['mags'], fmt='ko',
                yerr=tms['errs'], markersize=1., capsize=0, alpha=0.1)
        axs[0,ix].plot(tms['phase'],tfi['fitmags'], 'r-', linewidth=2.0)
        axs[0,ix].set(ylabel='normalized flux')
        axs[0,ix].set_title(dtitles[ix], fontsize=11)

        thisresidual = tms['mags']-tfi['fitmags']
        axs[1,ix].errorbar(tms['phase'], thisresidual,
                fmt='ko',yerr=tms['errs'],markersize=2., capsize=0,
                alpha=0.05)
        axs[1,ix].set(xlabel='phase', ylabel='residual')
        res_ylim = axs[1,ix].get_ylim()
        minylim = min(map(abs,res_ylim))
        axs[1,ix].set(ylim=(-minylim,minylim))

        dicts[ix]['fitinfo']['residual'] = thisresidual

    ebid, npts, period, kmag, morph = titleinfo[0], titleinfo[1], \
            titleinfo[2], titleinfo[3], titleinfo[4]
    st = f.suptitle('ebid: {:s}, npts: {:d}, period {:.3g} days, kmag {:.3g}, morph {:.3g}'.format(
        ebid, int(npts), period, kmag, morph), fontsize=14, y=1.06)

    f.tight_layout()
    f.savefig('../results/residual_diagnostics/'+ebid+'.png',
            bbox_inches='tight',bbox_extra_artists=[st], dpi=220)

    return dicts


def get_residual_flux_vs_time(dicts, rescaletomedian=True):
    '''
    Given the residual as a function of phase, get it as a residual as a
    function of time. (N.b. this returns dicts in place).

    Args:
        dicts: list of dictionaries with keys "residuals", "mags", "phase",
        "errs", "fitmags", "fitepoch". It doesn't have times.
    '''

    for ix, thisdict in enumerate(dicts):

        tms = thisdict['magseries']
        tfi = thisdict['fitinfo']
        ptimes = tms['times']
        pmags = tms['mags']
        presiduals = tfi['residual']
        perrs = tms['errs']
        phase = tms['phase']

        rtimeorder = np.argsort(ptimes)
        rtimes = ptimes[rtimeorder]
        rphase = phase[rtimeorder]
        rmags = presiduals[rtimeorder]
        rerrs = perrs[rtimeorder]

        if rescaletomedian:
            median_mag = np.median(rmags)
            rmags = rmags + median_mag

        dicts[ix]['fitinfo']['rtimes'] = rtimes
        dicts[ix]['fitinfo']['rphase'] = rphase
        dicts[ix]['fitinfo']['rmags'] = rmags
        dicts[ix]['fitinfo']['rerrs'] = rerrs

    return dicts


def get_windowed_goodness_of_fit(dicts, titleinfo, N=20, datatype='keplerLC'):
    '''
    Update dicts with the N worst-fitting EB-period windows (as measured by
    chi-squared, and ignoring windows without enough points).
    '''

    assert datatype == 'keplerLC', 'Other data require decisions inre: minnpts_perwindow.'
    # E.g., Kepler long cadence. < 1.5 hours 
    minnpts_perwindow = 3

    for ix, thisdict in enumerate(dicts):

        tms = thisdict['magseries']
        tfi = thisdict['fitinfo']
        rtimes = tfi['rtimes']
        rphase = tfi['rphase']
        rmags = tfi['rmags']
        rerrs = tfi['rerrs']

        ebid, npts, period, kmag, morph = titleinfo[0], titleinfo[1], titleinfo[2], titleinfo[3], titleinfo[4]

        # make windows. E.g., if times started at 0, these would be 
        # {[0,1],[1,2],[2,3],...,[0.5,1.5],[1.5,2.5],...}

        t_by_p = rtimes / period

        windowstart, windowend = int(min(t_by_p)), int(np.ceil(max(t_by_p)))

        # windowsl is a list of arrays, where the array contents are the 
        # boolean values passed to times, mags, and errs to get Χ^2 per window
        start, end = windowstart, windowstart+1
        windowsl = []
        while end < windowend:
            windowsl.append([(t_by_p>start) & (t_by_p<end)])
            windowsl.append([(t_by_p>start+0.5) & (t_by_p<end+0.5)])
            start += 1
            end += 1

        chisql, nptsinwindowl = [], []
        for window in windowsl:
            wtimes = rtimes[window]
            wphase = rphase[window]
            wmags = rmags[window]
            werrs = rerrs[window]

            # make sure we have at least one point in this window. windows
            # with too few points, e.g., b/c of a gap in observing, are dealt
            # with after the initial list is constructed.
            if np.size(wmags) == 0:
                chisql.append(-99)
                nptsinwindowl.append(0)
                continue

            # wmags are already a residual
            wchisq = np.sum(wmags * wmags / (werrs * werrs)) / len(wmags)

            chisql.append(wchisq)
            nptsinwindowl.append(len(wmags))

        nptsinwindowarr = nparr(nptsinwindowl)

        # interesting windows, point-counts, and chisquareds must have more
        # points than the minimum number, per window.
        windows_int = nparr(windowsl)[nptsinwindowarr > minnpts_perwindow]
        nptsinwindow_int = nptsinwindowarr[nptsinwindowarr > minnpts_perwindow]
        chisq_int = nparr(chisql)[nptsinwindowarr > minnpts_perwindow]

        meannpts = np.mean(nptsinwindow_int)
        mediannpts = np.median(nptsinwindow_int)
        stdnpts = np.std(nptsinwindow_int)
        assert abs(meannpts - mediannpts) / meannpts < stdnpts, 'Mean+median should be close, else significance cut below flawed'

        # if the number of points in the window is greater than 2σ below
        # the mean number of points, keep the window.
        keepinds = (nptsinwindow_int > meannpts - 2*stdnpts)
        nptsinwindow_keep = nptsinwindow_int[keepinds]
        windows_keep = windows_int[keepinds]
        chisq_keep = chisq_int[keepinds]

        # flag the N (e.g., 20) worst windows. Construct a dictionary to store
        # the information and return it with dicts.
        sargs = np.argsort(chisq_keep)
        worstfits = {
                'chisq':chisq_keep[sargs][-N:],
                'windows':windows_keep[sargs][-N:],
                'nptsinwindows':nptsinwindow_keep[sargs][-N:]}

        dicts[ix]['worstfits'] = worstfits

    print('Done finding worst fitting windows.')

    return dicts


def _visualize_worst_fits(dicts, titleinfo, N=10):
    '''
    Make a plot with 3 mega-panels.
    Panel 1: residual vs time (showing residuals both from spline fit and one
    of the savitzky-golay fits). Flag the windows where the fits are worst
    (with some box/highlighting scheme).
    Panel 2: spline residuals vs time zoomed in on each window. 2 rows, 5
    columns.
    Panel 3: savgol residuals vs time zoomed in on each window. 2 rows, 5
    columns.
    '''

    # assume dicts is a list of 5 dictionaries 1) fourier series, 2) spline,
    # 3-5) savitzky-golay.
    spldict = dicts[1]
    sgchisqs = []
    for sgd in dicts[2:5]:
        sgchisqs.append(sgd['fitchisq'])
    # select the savitzky-golay fit with the smallest chi-squared.
    sgdict = dicts[2+np.argmin(sgchisqs)]

    # matplotlib figure structure
    plt.close('all')
    nrows, ncols = 6, 5
    f = plt.figure(figsize=(ncols*5.5,nrows*3.5))
    gs = GridSpec(nrows, ncols) # 5 rows, 5 columns

    # row 0 is spline residual time series
    axt_spl = f.add_subplot(gs[0,:])
    # row 1 is savgol residual time series
    axt_sg = f.add_subplot(gs[1,:], sharex=axt_spl)
    splaxs = []
    for i in range(2,4): # spline residual time series rows 2 and 3.
        for j in range(0,ncols):
            splaxs.append(f.add_subplot(gs[i,j]))
    sgaxs = []
    for i in range(4,6): # savgol residual time series rows 4 and 5.
        for j in range(0,ncols):
            sgaxs.append(f.add_subplot(gs[i,j]))

    # spline and savgol residual time series
    axt_spl.errorbar(spldict['fitinfo']['rtimes'], spldict['fitinfo']['rmags'],
            fmt='ko', yerr=spldict['fitinfo']['rerrs'], markersize=2.,
            capsize=0, alpha=0.8, zorder=1)
    axt_sg.errorbar(sgdict['fitinfo']['rtimes'], sgdict['fitinfo']['rmags'],
            fmt='ko', yerr=sgdict['fitinfo']['rerrs'], markersize=2.,
            capsize=0, alpha=0.8, zorder=1)

    codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY,]

    # plot the "windows" as matplotlib patches in the background
    # also plot the residuals as a function of time in the appropriate subplots
    # for each of these windows.

    # first for spline timeseries
    chisq = spldict['worstfits']['chisq'][::-1]
    windows = spldict['worstfits']['windows'][::-1]
    ylim = axt_spl.get_ylim()
    for ix in range(ncols*2):
        wintimes = spldict['fitinfo']['rtimes'][windows[ix][0]]
        wmax, wmin = max(wintimes), min(wintimes)

        verts = [
            (wmin, ylim[0]), # left, bottom
            (wmin, ylim[1]), # left, top
            (wmax, ylim[1]), # right, top
            (wmax, ylim[0]), # right, bottom
            (0., 0.), # ignored
            ]

        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='orange', lw=0, zorder=0)
        axt_spl.add_patch(patch)
        # note which window in order of chisq deviation this is.
        axt_spl.text(wmin+(wmax-wmin)/2., ylim[0]+0.05*(ylim[1]-ylim[0]),
                str(ix)+'.', horizontalalignment='center', fontsize='x-small')

        # plot residual vs time in window on lower panel subplot
        winmags = spldict['fitinfo']['rmags'][windows[ix][0]]
        winerrs = spldict['fitinfo']['rerrs'][windows[ix][0]]

        splaxs[ix].errorbar(wintimes, winmags, fmt='ko',
            yerr=winerrs, markersize=2., capsize=0, zorder=1)


    # second for savgol timeseries
    chisq = sgdict['worstfits']['chisq'][::-1]
    windows = sgdict['worstfits']['windows'][::-1]
    ylim = axt_sg.get_ylim()
    for ix in range(ncols*2):
        wintimes = sgdict['fitinfo']['rtimes'][windows[ix][0]]
        wmax, wmin = max(wintimes), min(wintimes)

        verts = [
            (wmin, ylim[0]), # left, bottom
            (wmin, ylim[1]), # left, top
            (wmax, ylim[1]), # right, top
            (wmax, ylim[0]), # right, bottom
            (0., 0.), # ignored
            ]

        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='green', lw=0, zorder=0,
                alpha=0.5)

        axt_sg.add_patch(patch)
        # note which window in order of chisq deviation this is.
        axt_sg.text(wmin+(wmax-wmin)/2., ylim[0]+0.05*(ylim[1]-ylim[0]),
                str(ix)+'.', horizontalalignment='center', fontsize='x-small')

        # plot residual vs time in window on lower panel subplot
        winmags = sgdict['fitinfo']['rmags'][windows[ix][0]]
        winerrs = sgdict['fitinfo']['rerrs'][windows[ix][0]]

        sgaxs[ix].errorbar(wintimes, winmags, fmt='ko',
            yerr=winerrs, markersize=2., capsize=0, zorder=1)


    ebid, npts, period, kmag, morph = titleinfo[0], titleinfo[1], \
            titleinfo[2], titleinfo[3], titleinfo[4]
    st = f.suptitle(\
            'ebid: %s, npts: %d, period (equal to window size) %.3g days,'
            'kmag %.3g, morph %.3g' %
            (ebid, int(npts), period, kmag, morph), fontsize=14, y=1.005)

    f.tight_layout()

    #FIXME: fine-tune figure
    f.savefig('../results/fit_residual_visualization/'+ebid+'.png',
            bbox_inches='tight',bbox_extra_artists=[st], dpi=220)

    print('Done plotting fit_residual_visualization.')


def fit_test():

    kebc = get_kepler_ebs_info()

    rawd = '../data/keplerebcat_LCs/data/raw/'
    ebpaths = [rawd+p for p in os.listdir(rawd)]
    ebids = [ebp.strip('.raw') for ebp in os.listdir(rawd)]

    #for each eclipsing binary in our subset of the KEBC catalog, process!
    for ix, ebpath in enumerate(ebpaths[:2]):

        ebid = ebpath.split('/')[-1].strip('.raw')
        period = float(kebc[kebc['KIC']==int(ebid)]['period'])
        kmag = float(kebc[kebc['KIC']==int(ebid)]['kmag'])
        morph = float(kebc[kebc['KIC']==int(ebid)]['morph'])

        fourierdir = '../results/fourier_subtraction_diagnostics/'
        fourierpath = fourierdir + 'test_'+str(ebid)+'.png'
        splinedir = '../results/spline_subtraction_diagnostics/'
        splinepath = splinedir + 'test_'+str(ebid)+'.png'
        sgdir = '../results/savgol_subtraction_diagnostics/'
        sgpath = sgdir + 'test_'+str(ebid)+'.png'

        if os.path.exists(fourierpath) and os.path.exists(splinepath) \
        and os.path.exists(sgpath):
            print('Found stuff in {:d}. Continue.'.format(ix))
            continue

        elif os.path.exists(
                '../results/residual_diagnostics/'+ebid+'.png') \
        and \
            os.path.exists(
                '../results/fit_residual_visualization/'+ebid+'.png'):

            print('Found {:s}. Continue.'.format(ebid))
            continue

        else:
            tab = ascii.read(ebpath)

        print('Read KIC {:s}; has {:d} points; period {:.3g} days.'.format(\
                ebid, len(tab), period))
        titleinfo = [ebid, len(tab), period, kmag, morph]

        times = nparr(tab['bjd'])
        mags = nparr(tab['dtr_flux'])
        errs = nparr(tab['dtr_err'])

        if not os.path.exists(
                '../results/residual_diagnostics/'+ebid+'.png') \
        or not os.path.exists(
                '../results/fit_residual_visualization/'+ebid+'.png'):

            # fourier series, univariate spline, savitzky-golay
            dicts = _fits(times, mags, errs, ix, period, fourierpath,
                splinepath, sgpath, diagnosticplots=False)

            # generate residual_diagnostics, and also update dicts to have
            # residuals as a key.
            dicts = _residual_plots(dicts, titleinfo)

            # given the residual as a function of phase, get it as a residual as a
            # function of time. (N.b. this returns dicts in place).
            dicts = get_residual_flux_vs_time(dicts)

            # window residuals(time) over the period of the EB. 
            dicts = get_windowed_goodness_of_fit(dicts, titleinfo, N=20)

            _visualize_worst_fits(dicts, titleinfo, N=10)

        else:
            print('Found {:s}. Continue.'.format(ebid))
            continue


    print('\nDone testing fits.')

    return dicts


if __name__ == '__main__':
    dicts = fit_test()
