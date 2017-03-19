import pdb
import numpy as np, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import logging
from datetime import datetime

from numpy import nan as npnan, median as npmedian, \
    isfinite as npisfinite, min as npmin, max as npmax, abs as npabs, \
    sum as npsum, array as nparr, std as npstd
from astrobase.varbase import lcfit as lcf
from astrobase.varbase.lcfit import spline_fit_magseries
from astrobase import lcmath

#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.kepler' % parent_name)

def LOGDEBUG(message):
    if LOGGER:
        LOGGER.debug(message)
    elif DEBUG:
        print('%sZ [DBUG]: %s' % (datetime.utcnow().isoformat(), message))

def LOGINFO(message):
    if LOGGER:
        LOGGER.info(message)
    else:
        print('%sZ [INFO]: %s' % (datetime.utcnow().isoformat(), message))

def LOGERROR(message):
    if LOGGER:
        LOGGER.error(message)
    else:
        print('%sZ [ERR!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGWARNING(message):
    if LOGGER:
        LOGGER.warning(message)
    else:
        print('%sZ [WRN!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGEXCEPTION(message):
    if LOGGER:
        LOGGER.exception(message)
    else:
        print(
            '%sZ [EXC!]: %s\nexception was: %s' % (
                datetime.utcnow().isoformat(),
                message, format_exc()
                )
            )


######################
# PLOTTING UTILITIES #
######################
def orosz_style_flux_vs_time(lcdat, flux_to_use='sap'):
    '''
    Make a plot in the style of Fig S4 of Orosz et al. (2012) showing Kepler
    data, colored by quarter.

    Args:
    lcdat (dict): dictionary with keys of quarter number for a single object.
        The value of the keys are the lightcurve dictionaries returned by
        `astrokep.read_kepler_fitslc`. The lightcurves have been detrended,
        using `detrend_lightcurve`, and normalized.

    flux_to_use (str): 'sap' or 'pdc' for whether to start the plot using
        simple aperture photometry from Kepler, or the presearch data
        conditioned photometry. By default, SAP.

    Returns: nothing, but saves the plot with a smart name to
        ../results/colored_flux_vs_time/
    '''

    assert flux_to_use == 'sap' or flux_to_use == 'pdc'

    keplerid = lcdat[list(lcdat.keys())[0]]['objectinfo']['keplerid']

    colors = ['r', 'g', 'b', 'k']
    plt.close('all')
    f, axs = plt.subplots(nrows=2, ncols=1, figsize=(20,10), sharex=True)

    for ix, quarter_number in enumerate(lcdat.keys()):

        for axix, ax in enumerate(axs):

            lc = lcdat[quarter_number]['dtr'][flux_to_use]

            if axix == 0:
                times = lc['times']
                fluxs = lc['fluxs']
                errs = lc['errs']

            elif axix == 1:
                times = lc['times']
                fluxs = lc['fluxs_dtr_norm']
                errs = lc['errs_dtr_norm']

            thiscolor = colors[ix%len(colors)]

            ax.plot(times, fluxs, c=thiscolor, linestyle='-',
                    marker='o', markerfacecolor=thiscolor,
                    markeredgecolor=thiscolor, ms=0.1, lw=0.1)
            if axix == 0:
                fitfluxs = lc['fitfluxs_legendre']
                ax.plot(times, fitfluxs, c='k', linestyle='-',
                    lw=0.5)

        # keep track of min/max times for setting xlims
        if ix == 0:
            min_time = npmin(times)
            max_time = npmax(times)
        elif ix > 0:
            if npmin(times) < min_time:
                min_time = npmin(times)
            if npmax(times) > max_time:
                max_time = npmax(times)


    # label axes, set xlimits for entire time series.
    timelen = max_time - min_time

    axs[0].set(xlabel='', ylabel=flux_to_use+' Flux [counts/s]',
            xlim=[min_time-timelen*0.03,max_time+timelen*0.03],
            title='KIC ID {:s}, {:s}, quality flag > 0.'.\
            format(str(keplerid), flux_to_use))
    axs[1].set(xlabel='Time (BJD something)',
            ylabel='Normalized flux',
            title='Detrended (n=10 Legendre series), normalized by median.')

    f.tight_layout()

    savedir = '../results/colored_flux_vs_time/'
    plotname = str(keplerid)+'_'+flux_to_use+'_os.png'
    f.savefig(savedir+plotname, dpi=300)


def whitenedplot_5row(lcd, ap='sap'):
    '''
    Make a plot in the style of Fig S4 of Orosz et al. (2012) showing Kepler
    data, colored by quarter, but extended to include periodograms,
    phase-folded normalized and detrended flux, and then whitened flux.

    Args:
    lcd (dict): dictionary with all the data.

    ap (str): 'sap' or 'pdc' for whether to start the plot using
        simple aperture photometry from Kepler, or the presearch data
        conditioned photometry. By default, SAP.

    Returns: nothing, but saves the plot with a smart name to
        ../results/whitened_diagnostic/
    '''

    assert ap == 'sap' or ap == 'pdc'

    keplerid = lcd[list(lcd.keys())[0]]['objectinfo']['keplerid']

    colors = ['r', 'g', 'b', 'gray']

    # Set up matplotlib figure and axes.
    plt.close('all')
    nrows, ncols = 5, 5
    f = plt.figure(figsize=(16, 10))
    gs = GridSpec(nrows, ncols) # 5 rows, 5 columns

    # row 0: SAP/PDC timeseries
    ax_raw = f.add_subplot(gs[0,:])
    # row 1: detrended & normalized
    ax_dtr = f.add_subplot(gs[1,:], sharex=ax_raw)
    # row 2: periodograms
    axs_pg = []
    for i in range(2,3):
        for j in range(0,ncols):
            axs_pg.append(f.add_subplot(gs[i,j]))
    # row 3: phase-folded normalized, detrended flux
    axs_pf = []
    for i in range(3,4):
        for j in range(0,ncols):
            axs_pf.append(f.add_subplot(gs[i,j]))
    # row 4: whitened timeseries
    ax_w = f.add_subplot(gs[4,:], sharex=ax_raw)


    LOGINFO('Beginning whitened plot. KEPID %s (%s)' %
            (str(keplerid), ap))
    # ALL-TIME SERIES (rows 0,1,4)
    for ix, qnum in enumerate(lcd.keys()):

        for axix, ax in enumerate([ax_raw, ax_dtr, ax_w]):

            if axix == 0:
                lc = lcd[qnum]['dtr'][ap]
                times = lc['times']
                fluxs = lc['fluxs']
                errs = lc['errs']
            elif axix == 1:
                lc = lcd[qnum]['dtr'][ap]
                times = lc['times']
                fluxs = lc['fluxs_dtr_norm']
                errs = lc['errs_dtr_norm']
            elif axix == 2:
                lc = lcd[qnum]['white'][ap]['whiteseries']
                times = lc['times']
                fluxs = lc['fluxes']
                errs = lc['errs']

            thiscolor = colors[int(qnum)%len(colors)]

            if axix != 1:
                ax.plot(times, fluxs, c=thiscolor, linestyle='-',
                        marker='o', markerfacecolor=thiscolor,
                        markeredgecolor=thiscolor, ms=0.1, lw=0.1)
            elif axix == 1: # short-period fits require fine-tuning :(
                ax.scatter(times, fluxs, c=thiscolor, s=0.1,
                        edgecolors=thiscolor, zorder=10)

            if axix == 0:
                fitfluxs = lc['fitfluxs_legendre']
                ax.plot(times, fitfluxs, c='k', linestyle='-', lw=0.5)
            elif axix == 1:
                pfitfluxs = lcd[qnum]['white'][ap]['fitinfo']['fitmags']
                pfittimes = lcd[qnum]['white'][ap]['magseries']['times']

                wtimeorder = np.argsort(pfittimes)
                tfitfluxes = pfitfluxs[wtimeorder]
                tfittimes = pfittimes[wtimeorder]

                ax.plot(tfittimes, tfitfluxes, c='k', linestyle='-', lw=0.5,
                        zorder=0)


            txt = '%d' % (int(qnum))
            txt_x = npmin(times) + (npmax(times)-npmin(times))/2
            txt_y = npmin(fluxs) - (npmax(fluxs)-npmin(fluxs))/4

            ax.text(txt_x, txt_y, txt, horizontalalignment='center',
                    verticalalignment='center')


        # keep track of min/max times for setting xlims
        if ix == 0:
            min_time = npmin(times)
            max_time = npmax(times)
        elif ix > 0:
            if npmin(times) < min_time:
                min_time = npmin(times)
            if npmax(times) > max_time:
                max_time = npmax(times)

    # label axes, set xlimits for entire time series.
    timelen = max_time - min_time

    kebc_period = float(lcd[list(lcd.keys())[0]]['kebwg_info']['period'])
    ax_raw.get_xaxis().set_ticks([])
    ax_dtr.get_xaxis().set_ticks([])
    xmin, xmax = min_time-timelen*0.03, max_time+timelen*0.03
    ax_raw.set(xlabel='', ylabel=ap+' flux\n[counts/s]',
        xlim=[xmin,xmax],
        title='KICID:{:s}, {:s}, q_flag>0, KEBC_period: {:.7f} day.'.format(
        str(keplerid), ap, kebc_period) + ' (n=10 legendre series fit)')
    fitphasedeg = lcd[list(lcd.keys())[0]]['white']['sap']['fitinfo']['legendredeg']
    dtr_txt='Fit: n=%d legendre series to phase-folded by quarter.' \
        % (fitphasedeg)
    ax_dtr.text(0.5,0.98, dtr_txt, horizontalalignment='center',
            verticalalignment='top', transform=ax_dtr.transAxes)
    ax_dtr.set(ylabel='normalized,\ndetrended flux')
    ax_w.hlines([0.005,-0.005], xmin, xmax,
            colors='k',
            linestyles='--',
            zorder=-20)
    ax_w.set(xlabel='time [day]',
            ylabel='whitened flux',
            ylim=[-0.01,0.01])

    # PERIODOGRAMS
    qnums = np.sort(nparr(list(
            set(np.unique(np.sort(np.random.randint(1,len(lcd),size=ncols))))&\
            set(list(lcd.keys()))
            )))
    if len(lcd) <= 5:
        qnums = nparr(list(lcd.keys()))
    else:
        while len(qnums) != 5:
            newq = np.random.randint(1, max(list(lcd.keys())))
            qnums = np.sort(nparr(list(
                    set(np.unique(np.sort(np.insert(qnums, 0, newq))))&\
                    set(list(lcd.keys()))
                    )))
    if len(lcd) < 5:
        axs_pg = axs_pg[:len(lcd)]
        axs_pf = axs_pf[:len(lcd)]

    for ix, ax in enumerate(axs_pg):
        qnum = qnums[ix]

        ax.plot(lcd[qnum]['per'][ap]['periods'],
                lcd[qnum]['per'][ap]['lspvals'],
                'k-')

        pwr_ylim = ax.get_ylim()
        selperiod = lcd[qnum]['fineper'][ap]['selperiod']
        ax.vlines(selperiod, 0, 1.2, colors='r', linestyles=':', alpha=0.8, zorder=20)

        selforcedkebc = lcd[qnum]['per'][ap]['selforcedkebc']
        txt = 'q: %d, %s' % (int(qnum), selforcedkebc)
        ax.text(0.96,0.9,txt,horizontalalignment='right',
                verticalalignment='center',
                transform=ax.transAxes)

        ax.set(xlabel='', xscale='log')
        ax.get_xaxis().set_ticks([])
        if ix == 0:
            ax.set(ylabel='PDM power')
        else:
            ax.set(ylabel='')

    # PHASE-FOLDED LCS
    for ix, ax in enumerate(axs_pf):

        qnum = qnums[ix]
        pflux = lcd[qnum]['white'][ap]['magseries']['mags']
        phase = lcd[qnum]['white'][ap]['magseries']['phase']
        pfitflux = lcd[qnum]['white'][ap]['fitinfo']['fitmags']

        thiscolor = colors[int(qnum)%len(colors)]

        ax.plot(phase, pflux, c=thiscolor, linestyle='-',
                marker='o', markerfacecolor=thiscolor,
                markeredgecolor=thiscolor, ms=0.1, lw=0.1, zorder=0)
        ax.plot(phase, pfitflux, c='k', linestyle='-',
                lw=0.5, zorder=2)

        initperiod = lcd[qnum]['per'][ap]['selperiod']
        selperiod = lcd[qnum]['fineper'][ap]['selperiod']

        txt = 'q: %d' % (int(qnum))
        ax.text(0.98, 0.98, txt, horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)

        ax.get_xaxis().set_ticks([])
        pf_txt = 'P_init: %.7f day\nP_sel: %.7f day' % (initperiod, selperiod)
        ax.text(0.02, 0.02, pf_txt, horizontalalignment='left',
                verticalalignment='bottom', transform=ax.transAxes)
        if ix == 0:
            ax.set(ylabel='phase-folded norm,\ndtr flux')
        else:
            ax.set(ylabel='')


    f.tight_layout(h_pad=-1)

    LOGINFO('Made whitened plot. Now saving...')
    savedir = '../results/whitened_5row_diagnostic/'
    plotname = str(keplerid)+'_'+ap+'_w.png'
    f.savefig(savedir+plotname, dpi=300)


def whitenedplot_6row(lcd, ap='sap', stage='', inj=False):
    '''
    Make a plot in the style of Fig S4 of Orosz et al. (2012) showing Kepler
    data, colored by quarter, but extended to include periodograms,
    phase-folded normalized and detrended flux, and then whitened flux,
    and then redetrended flux.

    Args:
    lcd (dict): dictionary with all the data.

    ap (str): 'sap' or 'pdc' for whether to start the plot using
        simple aperture photometry from Kepler, or the presearch data
        conditioned photometry. By default, SAP.

    stage (str): stage of processing at which this was made. E.g., "redtr_inj"
        if after redtrending and injecting transits.

    Returns: nothing, but saves the plot with a smart name to
        ../results/whitened_diagnostic/
    '''

    assert ap == 'sap' or ap == 'pdc'

    keplerid = lcd[list(lcd.keys())[0]]['objectinfo']['keplerid']

    colors = ['r', 'g', 'b', 'gray']

    # Set up matplotlib figure and axes.
    plt.close('all')
    nrows, ncols = 6, 5
    f = plt.figure(figsize=(16, 10))
    gs = GridSpec(nrows, ncols) # 5 rows, 5 columns

    # row 0: SAP/PDC timeseries
    ax_raw = f.add_subplot(gs[0,:])
    # row 1: detrended & normalized
    ax_dtr = f.add_subplot(gs[1,:], sharex=ax_raw)
    # row 2: periodograms
    axs_pg = []
    for i in range(2,3):
        for j in range(0,ncols):
            axs_pg.append(f.add_subplot(gs[i,j]))
    # row 3: phase-folded normalized, detrended flux
    axs_pf = []
    for i in range(3,4):
        for j in range(0,ncols):
            axs_pf.append(f.add_subplot(gs[i,j]))
    # row 4: whitened timeseries
    ax_w = f.add_subplot(gs[4,:], sharex=ax_raw)
    # row 5: redetrended & renormalized
    ax_redtr = f.add_subplot(gs[5,:], sharex=ax_raw)


    LOGINFO('Beginning whitened plot. KEPID %s (%s)' %
            (str(keplerid), ap))
    # ALL-TIME SERIES (rows 0,1,4)
    for ix, qnum in enumerate(lcd.keys()):

        for axix, ax in enumerate([ax_raw, ax_dtr, ax_w, ax_redtr]):

            if axix == 0:
                lc = lcd[qnum]['dtr'][ap]
                times = lc['times']
                fluxs = lc['fluxs']
                errs = lc['errs']
            elif axix == 1:
                lc = lcd[qnum]['dtr'][ap]
                times = lc['times']
                fluxs = lc['fluxs_dtr_norm']
                errs = lc['errs_dtr_norm']
            elif axix == 2:
                lc = lcd[qnum]['white'][ap]['whiteseries']
                times = lc['times']
                fluxs = lc['fluxes']
                errs = lc['errs']
            elif axix == 3:
                lc = lcd[qnum]['redtr'][ap]
                times = lc['times']
                fluxs = lc['fluxs'] - lc['fitfluxs_legendre']
                errs = lc['errs']

            thiscolor = colors[int(qnum)%len(colors)]

            if axix != 1:
                ax.plot(times, fluxs, c=thiscolor, linestyle='-',
                        marker='o', markerfacecolor=thiscolor,
                        markeredgecolor=thiscolor, ms=0.1, lw=0.1)
            elif axix == 1: # fits require fine-tuning for sizes/widths :(
                ax.scatter(times, fluxs, c=thiscolor, s=0.1,
                        edgecolors=thiscolor, zorder=10)

            if axix == 0:
                fitfluxs = lc['fitfluxs_legendre']
                ax.plot(times, fitfluxs, c='k', linestyle='-', lw=0.5)
            elif axix == 1:
                pfitfluxs = lcd[qnum]['white'][ap]['fitinfo']['fitmags']
                pfittimes = lcd[qnum]['white'][ap]['magseries']['times']

                wtimeorder = np.argsort(pfittimes)
                tfitfluxes = pfitfluxs[wtimeorder]
                tfittimes = pfittimes[wtimeorder]

                ax.plot(tfittimes, tfitfluxes, c='k', linestyle='-', lw=0.5,
                        zorder=0)
            elif axix == 2:
                fitfluxs = lcd[qnum]['redtr'][ap]['fitfluxs_legendre']
                fittimes = lcd[qnum]['redtr'][ap]['times']
                ax.plot(fittimes, fitfluxs, c='k', linestyle='-', lw=0.5,
                        zorder=10)


            txt = '%d' % (int(qnum))
            txt_x = npmin(times) + (npmax(times)-npmin(times))/2
            txt_y = npmin(fluxs) - (npmax(fluxs)-npmin(fluxs))/4

            ax.text(txt_x, txt_y, txt, horizontalalignment='center',
                    verticalalignment='center')


        # keep track of min/max times for setting xlims
        if ix == 0:
            min_time = npmin(times)
            max_time = npmax(times)
        elif ix > 0:
            if npmin(times) < min_time:
                min_time = npmin(times)
            if npmax(times) > max_time:
                max_time = npmax(times)

    # label axes, set xlimits for entire time series.
    timelen = max_time - min_time

    kebc_period = float(lcd[list(lcd.keys())[0]]['kebwg_info']['period'])
    ax_raw.get_xaxis().set_ticks([])
    ax_dtr.get_xaxis().set_ticks([])
    xmin, xmax = min_time-timelen*0.03, max_time+timelen*0.03
    ax_raw.set(xlabel='', ylabel=ap+' flux\n[counts/s]',
        xlim=[xmin,xmax],
        title='KICID:{:s}, {:s}, q_flag>0, KEBC_period: {:.7f} day.'.format(
        str(keplerid), ap, kebc_period) + ' (n=20 legendre series fit)')
    fitphasedeg = lcd[list(lcd.keys())[0]]['white']['sap']['fitinfo']['legendredeg']
    dtr_txt='Fit: n=%d legendre series to phase-folded by quarter.' \
        % (fitphasedeg)
    ax_dtr.text(0.5,0.98, dtr_txt, horizontalalignment='center',
            verticalalignment='top', transform=ax_dtr.transAxes)
    ax_dtr.set(ylabel='normalized,\ndetrended flux')
    ax_w.hlines([0.005,-0.005], xmin, xmax,
            colors='k',
            linestyles='--',
            zorder=-20)

    w_txt='fit: n=20 legendre series.'
    ax_w.text(0.5,0.98, w_txt, horizontalalignment='center',
            verticalalignment='top', transform=ax_w.transAxes)

    ax_redtr.hlines([0.005,-0.005], xmin, xmax,
            colors='k',
            linestyles='--',
            zorder=-20)
    ylim = [-0.01,0.01]
    if inj:
        ylim = [-0.015,0.015]
    ax_w.set(ylabel='whitened flux',
            ylim=ylim)
    ax_redtr.set(xlabel='time [day]',
            ylabel='redtr flux',
            ylim=ylim)


    # PERIODOGRAMS
    if len(lcd) > 1:
        qnums = np.sort(nparr(list(
                set(np.unique(np.sort(np.random.randint(1,len(lcd),size=ncols))))&\
                set(list(lcd.keys()))
                )))
    if len(lcd) <= 5 or len(lcd) == 1:
        qnums = nparr(list(lcd.keys()))
    else:
        while len(qnums) != 5:
            newq = np.random.randint(1, max(list(lcd.keys())))
            qnums = np.sort(nparr(list(
                    set(np.unique(np.sort(np.insert(qnums, 0, newq))))&\
                    set(list(lcd.keys()))
                    )))
    if len(lcd) < 5:
        axs_pg = axs_pg[:len(lcd)]
        axs_pf = axs_pf[:len(lcd)]

    for ix, ax in enumerate(axs_pg):
        qnum = qnums[ix]

        ax.plot(lcd[qnum]['per'][ap]['periods'],
                lcd[qnum]['per'][ap]['lspvals'],
                'k-')

        pwr_ylim = ax.get_ylim()
        selperiod = lcd[qnum]['fineper'][ap]['selperiod']
        ax.vlines(selperiod, 0, 1.2, colors='r', linestyles=':', alpha=0.8, zorder=20)

        selforcedkebc = lcd[qnum]['per'][ap]['selforcedkebc']
        txt = 'q: %d, %s' % (int(qnum), selforcedkebc)
        ax.text(0.96,0.9,txt,horizontalalignment='right',
                verticalalignment='center',
                transform=ax.transAxes)

        ax.set(xlabel='', xscale='log')
        ax.get_xaxis().set_ticks([])
        if ix == 0:
            ax.set(ylabel='PDM power')
        else:
            ax.set(ylabel='')

    # PHASE-FOLDED LCS
    for ix, ax in enumerate(axs_pf):

        qnum = qnums[ix]
        pflux = lcd[qnum]['white'][ap]['magseries']['mags']
        phase = lcd[qnum]['white'][ap]['magseries']['phase']
        pfitflux = lcd[qnum]['white'][ap]['fitinfo']['fitmags']

        thiscolor = colors[int(qnum)%len(colors)]

        ax.plot(phase, pflux, c=thiscolor, linestyle='-',
                marker='o', markerfacecolor=thiscolor,
                markeredgecolor=thiscolor, ms=0.1, lw=0.1, zorder=0)
        ax.plot(phase, pfitflux, c='k', linestyle='-',
                lw=0.5, zorder=2)

        initperiod = lcd[qnum]['per'][ap]['selperiod']
        selperiod = lcd[qnum]['fineper'][ap]['selperiod']

        txt = 'q: %d' % (int(qnum))
        ax.text(0.98, 0.98, txt, horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)

        ax.get_xaxis().set_ticks([])
        pf_txt = 'P_init: %.7f day\nP_sel: %.7f day' % (initperiod, selperiod)
        ax.text(0.02, 0.02, pf_txt, horizontalalignment='left',
                verticalalignment='bottom', transform=ax.transAxes)
        if ix == 0:
            ax.set(ylabel='phase-folded norm,\ndtr flux')
        else:
            ax.set(ylabel='')


    f.tight_layout(h_pad=-1)

    # Figure out names and write.
    savedir = '../results/whitened_diagnostic/'
    if 'inj' in stage:
        savedir += 'inj/'
    elif 'inj' not in stage:
        savedir += 'no_inj/'
    plotname = str(keplerid)+'_'+ap+stage+'.png'
    if 'eb_sbtr' in stage:
        # override for EB subtraction tests
        savedir = '../results/whitened_diagnostic/eb_subtraction/'

    f.savefig(savedir+plotname, dpi=300)

    LOGINFO('Made & saved whitened plot to {:s}'.format(savedir+plotname))


def add_inset_axes(ax,fig,rect,axisbg='w'):
    '''
    Copied directly from
    http://stackoverflow.com/questions/17458580/embedding-small-plots-inside-subplots-in-matplotlib
    '''
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg,frameon=False)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax



def dipsearchplot(lcd, allq, ap=None, stage='', inj=False, varepoch='bls',
        phasebin=0.002, inset=True):
    '''
    The phased lightcurves in this plot shamelessly rip from those written by
    Waqas Bhatti in astrobase.checkplot.

    This plot looks like:
    |           flux vs time for all quarters              |
    | phased 1     |    phased 2      |       phased 3     |
    | phased 4     |    phased 5      |       periodogram  |

    Args:
    lcd (dict): dictionary with all the quarter-keyed data.

    allq (dict): dictionary with all the stitched full-time series data.

    ap (str): 'sap' or 'pdc' for whether to start the plot using
        simple aperture photometry from Kepler, or the presearch data
        conditioned photometry. By default, SAP.

    stage (str): stage of processing at which this was made. E.g., "redtr_inj"
        if after redtrending and injecting transits.

    varepoch (str or None): if 'bls', goes by the ingress/egress computed in
        binned-phase from BLS. If 'splfit', tries to find a good epoch by
        spline fitting to the binned phase-folded LC (formerly implemented, now
        broken).

    Returns: nothing, but saves the plot with a smart name to
        ../results/dipsearchplot/
    '''

    assert ap == 'sap' or ap == 'pdc'

    keplerid = lcd[list(lcd.keys())[0]]['objectinfo']['keplerid']

    colors = ['r', 'g', 'b', 'gray']

    # Set up matplotlib figure and axes.
    plt.close('all')
    nrows, ncols = 3, 3
    f = plt.figure(figsize=(16, 10))
    gs = GridSpec(nrows, ncols) # 5 rows, 5 columns

    # row 0: SAP/PDC timeseries
    ax_raw = f.add_subplot(gs[0,:])
    # phased at 5 favorite periods
    axs_φ = []
    for i in range(1,3):
        for j in range(0,ncols):
            if i == 2 and j == 2:
                continue
            else:
                axs_φ.append(f.add_subplot(gs[i,j]))
    # periodogram
    ax_pg = f.add_subplot(gs[2,2])

    f.tight_layout(h_pad=-0.7)

    LOGINFO('Beginning dipsearchplot. KEPID %s (%s)' %
            (str(keplerid), ap))

    ###############
    # TIME SERIES #
    ###############
    qnums = np.unique(allq['dipfind']['tfe'][ap]['qnums'])
    lc = allq['dipfind']['tfe'][ap]
    quarters = lc['qnums']
    ylim_raw = [-0.015,0.015]

    for ix, qnum in enumerate(qnums):

        times = lc['times'][quarters==qnum]
        fluxs = lc['fluxs'][quarters==qnum]
        errs = lc['errs'][quarters==qnum]

        thiscolor = colors[int(qnum)%len(colors)]

        ax_raw.plot(times, fluxs, c=thiscolor, linestyle='-',
                marker='o', markerfacecolor=thiscolor,
                markeredgecolor=thiscolor, ms=0.1, lw=0.1)

        txt = '%d' % (int(qnum))
        txt_x = npmin(times) + (npmax(times)-npmin(times))/2
        txt_y = npmedian(fluxs) - 2*npstd(fluxs)
        if txt_y < ylim_raw[0]:
            txt_y = min(ylim_raw) + 0.001

        ax_raw.text(txt_x, txt_y, txt, horizontalalignment='center',
                verticalalignment='center')

        # keep track of min/max times for setting xlims
        if ix == 0:
            min_time = npmin(times)
            max_time = npmax(times)
        elif ix > 0:
            if npmin(times) < min_time:
                min_time = npmin(times)
            if npmax(times) > max_time:
                max_time = npmax(times)

    # label axes, set xlimits for entire time series.
    timelen = max_time - min_time
    p = allq['inj_model']
    injdepth = (p['params'].rp)**2

    kebc_period = float(lcd[list(lcd.keys())[0]]['kebwg_info']['period'])
    ax_raw.get_xaxis().set_ticks([])
    xmin, xmax = min_time-timelen*0.03, max_time+timelen*0.03
    ax_raw.set(xlabel='', ylabel=ap.upper()+' relative flux (redtr)',
        xlim=[xmin,xmax],
        ylim = ylim_raw)
    ax_raw.set_title(
        'KIC:{:s}, {:s}, q_flag>0, KEBC_P: {:.4f}. '.format(
        str(keplerid), ap.upper(), kebc_period)+\
        'day, inj, dtr, whitened, redtr (n=20 legendre series fit). '+\
        'Depth_inj: {:.4f}'.format(injdepth),
        fontsize='xx-small')
    ax_raw.hlines([0.005,-0.005], xmin, xmax,
            colors='k',
            linestyles='--',
            zorder=-20)

    ############################
    # PHASE-FOLDED LIGHTCURVES #
    ############################
    pgd = allq['dipfind']['bls'][ap]['coarsebls']
    nbestperiods = pgd['nbestperiods']
    for ix, ax in enumerate(axs_φ):

        foldperiod = nbestperiods[ix]

        if isinstance(varepoch,str) and varepoch == 'splfit':
            try:
                assert 0
                lc = allq['dipfind']['tfe'][ap]
                ftimes = lc['times']
                ffluxs = lc['fluxs']
                ferrs = lc['errs']
                spfit = spline_fit_magseries(ftimes,
                                             ffluxs,
                                             ferrs,
                                             foldperiod,
                                             magsarefluxes=True,
                                             sigclip=None)
                varepoch = spfit['fitinfo']['fitepoch']
                if len(varepoch) != 1:
                    varepoch = varepoch[0]

                LOGINFO('KEPID %s (%s) making phased LC. P: %.6f, t_0: %.5f' %
                        (str(keplerid), ap, foldperiod, varepoch))

            except Exception as e:
                LOGEXCEPTION(
                'option not yet implemented (call lcmath.phase_magseries)'
                )

        # Recover quantities to plot, defined on φ=[0,1]
        fbls = allq['dipfind']['bls'][ap]['finebls'][foldperiod]
        plotφ = fbls['φ']
        plotfluxs = fbls['flux_φ']
        binplotφ = fbls['binned_φ']
        binplotfluxs = fbls['binned_flux_φ']
        φ_0 = fbls['φ_0']
        if ix == 0:
            bestφ_0 = φ_0
        φ_ing, φ_egr = fbls['φ_ing'],fbls['φ_egr']
        # Want to wrap over φ=[-2,2]
        plotφ = np.concatenate(
                (plotφ-2.,plotφ-1.,plotφ,plotφ+1.,plotφ+2.)
                )
        plotfluxs = np.concatenate(
                (plotfluxs,plotfluxs,plotfluxs,plotfluxs,plotfluxs)
                )
        binplotφ = np.concatenate(
                (binplotφ-2.,binplotφ-1.,binplotφ,binplotφ+1.,binplotφ+2.)
                )
        binplotfluxs = np.concatenate(
                (binplotfluxs,binplotfluxs,binplotfluxs,binplotfluxs,binplotfluxs)
                )

        # Make the phased LC plot
        ax.scatter(plotφ-φ_0,plotfluxs,marker='o',s=2,color='gray')
        # Overlay the binned phased LC plot
        if phasebin:
            ax.scatter(binplotφ-φ_0,binplotfluxs,marker='o',s=10,color='blue')

        if inset:
            subpos = [0.03,0.79,0.25,0.2]
            iax = add_inset_axes(ax,f,subpos)

            iax.scatter(plotφ-φ_0,plotfluxs,marker='o',s=2*0.3,color='gray')
            if phasebin:
                iax.scatter(binplotφ-φ_0,binplotfluxs,marker='o',s=10*0.3,color='blue')
            iaxylim = iax.get_ylim()

            iax.set(ylabel='', xlim=[-0.7,0.7])
            iax.get_xaxis().set_visible(False)
            iax.get_yaxis().set_visible(False)

        t0 = min_time + φ_0*foldperiod
        txt = 'P_fold: {:.5f} d\nt_0: {:.5f} \n$\phi_0$= {:.5f}'.format(
            foldperiod, t0, φ_0)
        ax.text(0.98, 0.02, txt, horizontalalignment='right',
                verticalalignment='bottom',
                transform=ax.transAxes, fontsize='x-small')

        ax.set(ylabel='', xlim=[-0.1,0.1])
        axylim = ax.get_ylim()
        ax.vlines([0.], min(axylim), max(axylim), colors='red',
                linestyles='-', alpha=0.9, zorder=-1)
        if inset:
            iax.vlines([-0.5,0.5], min(iaxylim), max(iaxylim), colors='black',
                    linestyles='-', alpha=0.7, zorder=30)
            iax.vlines([0.], min(iaxylim), max(iaxylim), colors='red',
                    linestyles='-', alpha=0.9, zorder=30)


    ################
    # PERIODOGRAMS #
    ################
    pgdc = allq['dipfind']['bls'][ap]['coarsebls']
    pgdf = allq['dipfind']['bls'][ap]['finebls']

    ax_pg.plot(pgdc['periods'], pgdc['lspvals'], 'k-')

    pwr_ylim = ax_pg.get_ylim()
    nbestperiods = pgdc['nbestperiods']
    cbestperiod = pgd['bestperiod']

    # We want the FINE best period, not the coarse one (although the frequency
    # grid might actually be small enough that this doesn't improve much!)
    fbestperiod = pgdf[cbestperiod]['serialdict']['bestperiod']

    best_t0 = min_time + bestφ_0*fbestperiod
    ax_pg.vlines(nbestperiods, min(pwr_ylim), max(pwr_ylim), colors='r',
            linestyles=':', alpha=0.8, zorder=20)

    p = allq['inj_model']
    injperiod = p['params'].per
    inj_t0 = p['params'].t0
    ax_pg.vlines(injperiod, min(pwr_ylim), max(pwr_ylim), colors='g',
            linestyles='-', alpha=0.8, zorder=10)

    selforcedkebc = lcd[qnum]['per'][ap]['selforcedkebc']
    txt = 'P_inj: %.4f d\nP_rec: %.4f d\nt_0,inj: %.4f\nt_0,rec: %.4f' % \
          (injperiod, fbestperiod, inj_t0, best_t0)
    ax_pg.text(0.96,0.96,txt,horizontalalignment='right',
            verticalalignment='top',
            transform=ax_pg.transAxes)

    ax_pg.set(xlabel='period [d]', xscale='log')
    ax_pg.get_yaxis().set_ticks([])
    ax_pg.set(ylabel='BLS power')



    # Figure out names and write.
    savedir = '../results/dipsearchplot/'
    if 'inj' in stage:
        savedir += 'inj/'
    elif 'inj' not in stage:
        savedir += 'no_inj/'
    plotname = str(keplerid)+'_'+ap+stage+'.png'

    f.savefig(savedir+plotname, dpi=300, bbox_inches='tight')

    LOGINFO('Made & saved whitened plot to {:s}'.format(savedir+plotname))


def plot_iterwhiten_3row(lcd, allq, ap='sap', stage='', inj=False):

    #lcd[qnum]['white'][inum][ap]['w*']`, for * in (fluxs,errs,times,phases)

    assert ap == 'sap' or ap == 'pdc'
    keplerid = lcd[list(lcd.keys())[0]]['objectinfo']['keplerid']
    kebc_period = nparr(float(lcd[list(lcd.keys())[0]]['kebwg_info']['period']))

    qnums = list(lcd.keys())

    for qnum in qnums:
        inums = list(lcd[qnum]['white'].keys())
        for inum in inums:
            ap = 'sap'

            # Set up matplotlib figure and axes.
            plt.close('all')
            nrows, ncols = 3, 2
            f = plt.figure(figsize=(16, 10))
            gs = GridSpec(nrows, ncols) # 5 rows, 5 columns
            # row 0: dtr timeseries
            ax_dtr = f.add_subplot(gs[0,:])
            # row 2: detrended & normalized
            ax_w = f.add_subplot(gs[2,:], sharex=ax_dtr)
            # row1, col0: PDM periodogram
            ax_pg = f.add_subplot(gs[1,0])
            # row1, col1: phase-folded dtr 
            ax_pf = f.add_subplot(gs[1,1])
            LOGINFO('Beginning iterwhiten plot. KEPID %s (%s)' % (
                str(keplerid), ap))

            # TIMESERIES
            for axix, ax in enumerate([ax_dtr, ax_w]):
                if axix == 0 and inum == 0:
                    times = lcd[qnum]['dtr'][ap]['times']
                    fluxs = lcd[qnum]['dtr'][ap]['fluxs_dtr_norm']
                    errs =  lcd[qnum]['dtr'][ap]['errs_dtr_norm']
                elif axix == 0 and inum > 0:
                    times = lcd[qnum]['white'][inum-1][ap]['legdict']['whiteseries']['times']
                    fluxs = lcd[qnum]['white'][inum-1][ap]['legdict']['whiteseries']['wfluxsresid']
                    errs =  lcd[qnum]['white'][inum-1][ap]['legdict']['whiteseries']['errs']
                elif axix == 1:
                    times = lcd[qnum]['white'][inum][ap]['legdict']['whiteseries']['times']
                    fluxs = lcd[qnum]['white'][inum][ap]['legdict']['whiteseries']['wfluxs']
                    fitfluxs = lcd[qnum]['white'][inum][ap]['legdict']['whiteseries']['wfluxslegfit']
                    errs =  lcd[qnum]['white'][inum][ap]['legdict']['whiteseries']['errs']

                meanflux = np.mean(fluxs)
                rms_biased = float(np.sqrt(np.sum((fluxs-meanflux)**2) / len(fluxs)))

                ax.plot(times, fluxs, linestyle='-', marker='o',
                       markerfacecolor='black', markeredgecolor='black',
                       ms=1, lw=0.2, zorder=1)
                if axix == 1:
                    ax.plot(times, fitfluxs, linestyle='-', marker='o',
                           markerfacecolor='black', markeredgecolor='black',
                           ms=0, lw=1, color='red', zorder=2)

                txt = 'RMS: %.4g' % (rms_biased)
                ax.text(0.02, 0.02, txt, horizontalalignment='left',
                    verticalalignment='bottom',
                    transform=ax.transAxes)

            ax_dtr.set(ylabel='dtr flux')
            ax_pg.set(ylabel='PDM power')

            # PERIODOGRAM
            ax_pg.plot(lcd[qnum]['white'][inum][ap]['per']['periods'],
                      lcd[qnum]['white'][inum][ap]['per']['lspvals'],
                      'k-')

            selperiod = lcd[qnum]['white'][inum][ap]['per']['selperiod']
            inj_period = allq['inj_model']['params'].per
            pwr_ylim = ax_pg.get_ylim()
            ax_pg.vlines(selperiod, min(pwr_ylim), max(pwr_ylim), colors='r', linestyles='--', alpha=0.8,
                         zorder=20, label='P sel')
            ax_pg.vlines(kebc_period, min(pwr_ylim), max(pwr_ylim), colors='g', linestyles='--', alpha=0.8,
                     zorder=20, label='P EB')
            ax_pg.vlines(inj_period, min(pwr_ylim), max(pwr_ylim), colors='b', linestyles=':', alpha=0.8,
                     zorder=20, label='P CBP')
            ax_pg.legend(fontsize='xx-small', loc='lower right')
            ax_pg.set_ylim(pwr_ylim)
            ax_pg.set(xscale='log')

            # PHASE-FOLD
            pflux = lcd[qnum]['white'][inum][ap]['legdict']['magseries']['mags']
            phase = lcd[qnum]['white'][inum][ap]['legdict']['magseries']['phase']
            pfitflux = lcd[qnum]['white'][inum][ap]['legdict']['fitinfo']['fitmags']
            legdeg = lcd[qnum]['white'][inum][ap]['legdict']['fitinfo']['legendredeg']

            thiscolor = 'blue'
            ax_pf.plot(phase, pflux, c=thiscolor, linestyle='-',
                    marker='o', markerfacecolor=thiscolor,
                    markeredgecolor=thiscolor, ms=0.1, lw=0.1, zorder=0)
            ax_pf.plot(phase, pfitflux, c='k', linestyle='-',
                    lw=0.5, zorder=2)

            selperiod = lcd[qnum]['white'][inum][ap]['fineper']['selperiod']

            txt = 'q: %d, inum: %d\nlegdeg: %d, npts: %d' % (
                    int(qnum), int(inum), int(legdeg), len(pflux))
            ax_pf.text(0.98, 0.98, txt, horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax_pf.transAxes)

            pf_txt = 'P EB: %.7f day\nP sel: %.7f day' % (kebc_period, selperiod)
            ax_pf.text(0.02, 0.02, pf_txt, horizontalalignment='left',
                    verticalalignment='bottom', transform=ax_pf.transAxes)
            ax_pf.set(ylabel='phased dtr flux')

            f.tight_layout()
            fname = '{:s}_qnum{:s}_inum{:s}_sap.png'.format(
                    str(keplerid), str(int(qnum)), str(int(inum)))
            savedir = '../results/eb_subtraction_diagnostics/iterwhiten/'

            f.savefig(savedir+fname, dpi=300, bbox_inches='tight')
