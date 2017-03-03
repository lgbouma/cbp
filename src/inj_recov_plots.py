import pdb
import numpy as np, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import logging
from datetime import datetime

from numpy import nan as npnan, median as npmedian, \
    isfinite as npisfinite, min as npmin, max as npmax, abs as npabs, \
    sum as npsum, array as nparr
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

    # Figure out names and write.
    savedir = '../results/whitened_diagnostic/'
    if 'inj' in stage:
        savedir += 'inj/'
    elif 'inj' not in stage:
        savedir += 'no_inj/'
    plotname = str(keplerid)+'_'+ap+stage+'.png'

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



def dipsearchplot(lcd, allq, ap=None, stage='', inj=False, varepoch='min',
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

    varepoch (str or None): if 'min', tries to actually find a good epoch so
        that your phase-folded LCs have dips at phase 0.5

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

    # TIME SERIES
    qnums = np.unique(allq['dipfind']['tfe'][ap]['qnums'])
    lc = allq['dipfind']['tfe'][ap]
    quarters = lc['qnums']

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
        txt_y = npmin(fluxs) + (npmax(fluxs)-npmin(fluxs))/6

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

    kebc_period = float(lcd[list(lcd.keys())[0]]['kebwg_info']['period'])
    ax_raw.get_xaxis().set_ticks([])
    xmin, xmax = min_time-timelen*0.03, max_time+timelen*0.03
    ax_raw.set(xlabel='', ylabel=ap.upper()+' relative flux (redtr)',
        xlim=[xmin,xmax],
        ylim = [-0.015,0.015])
    ax_raw.set_title(
        'KIC:{:s}, {:s}, q_flag>0, KEBC_P: {:.4f} '.format(
        str(keplerid), ap.upper(), kebc_period)+\
        'day, inj, dtr, whitened, redtr (n=20 legendre series fit)',
        fontsize='small')
    ax_raw.hlines([0.005,-0.005], xmin, xmax,
            colors='k',
            linestyles='--',
            zorder=-20)


    # PERIODOGRAMS
    pgd = allq['dipfind']['bls'][ap]

    ax_pg.plot(pgd['periods'], pgd['lspvals'], 'k-')

    pwr_ylim = ax_pg.get_ylim()
    nbestperiods = pgd['nbestperiods']
    bestperiod = pgd['bestperiod']
    ax_pg.vlines(nbestperiods, min(pwr_ylim), max(pwr_ylim), colors='r',
            linestyles=':', alpha=0.8, zorder=20)

    p = allq['inj_model']
    injperiod = p['params'].per
    ax_pg.vlines(injperiod, min(pwr_ylim), max(pwr_ylim), colors='g',
            linestyles='-', alpha=0.8, zorder=10)

    selforcedkebc = lcd[qnum]['per'][ap]['selforcedkebc']
    txt = 'P_inj: %.4f d\nP_rec: %.4f d' % (injperiod, bestperiod)
    ax_pg.text(0.96,0.96,txt,horizontalalignment='right',
            verticalalignment='top',
            transform=ax_pg.transAxes)

    ax_pg.set(xlabel='period [d]', xscale='log')
    ax_pg.get_yaxis().set_ticks([])
    ax_pg.set(ylabel='BLS power')

    # PHASE-FOLDED LIGHTCURVES
    lc = allq['dipfind']['tfe'][ap]
    times = lc['times']
    fluxs = lc['fluxs']
    errs = lc['errs']

    for ix, ax in enumerate(axs_φ):

        foldperiod = nbestperiods[ix]

        # figure out the epoch, if it's None, use the min of the time
        if varepoch is None:
            varepoch = np.min(times)

        # if the varepoch is 'min', then fit a spline to the light curve
        # phased using the min of the time, find the fit mag minimum and use
        # the time for that as the varepoch
        elif isinstance(varepoch,str) and varepoch == 'min':
            try:
                spfit = spline_fit_magseries(times,
                                             fluxs,
                                             errs,
                                             foldperiod,
                                             magsarefluxes=True,
                                             sigclip=None)
                varepoch = spfit['fitinfo']['fitepoch']
                if len(varepoch) != 1:
                    varepoch = varepoch[0]
            except Exception as e:
                LOGEXCEPTION('spline fit failed, using min(times) as epoch')
                varepoch = np.min(stimes)

        LOGINFO('KEPID %s (%s) plotting phased LC. period: %.6f, epoch: %.5f' %
                (str(keplerid), ap, foldperiod, varepoch))

        # phase the magseries
        phasedlc = lcmath.phase_magseries(times,
                                          fluxs,
                                          foldperiod,
                                          varepoch,
                                          wrap=True,
                                          sort=True)
        plotphase = phasedlc['phase']
        plotfluxs = phasedlc['mags']

        # if we're supposed to bin the phases, do so
        if phasebin:
            binphasedlc = lcmath.phase_bin_magseries(plotphase,
                                              plotfluxs,
                                              binsize=phasebin)
            binplotphase = binphasedlc['binnedphases']
            binplotfluxs = binphasedlc['binnedmags']

        else:
            binplotphase = None
            binplotfluxs = None

        # finally, make the phased LC plot
        ax.scatter(plotphase,plotfluxs,marker='o',s=2,color='gray')
        # overlay the binned phased LC plot if we're making one
        if phasebin:
            ax.scatter(binplotphase,binplotfluxs,marker='o',s=10,color='blue')

        if inset:
            subpos = [0.03,0.79,0.25,0.2]
            iax = add_inset_axes(ax,f,subpos)

            iax.scatter(plotphase,plotfluxs,marker='o',s=2*0.3,color='gray')
            if phasebin:
                iax.scatter(binplotphase,binplotfluxs,marker='o',s=10*0.3,color='blue')
            iaxylim = iax.get_ylim()

            iax.set(ylabel='', xlim=[-0.7,0.7])
            iax.get_xaxis().set_visible(False)
            iax.get_yaxis().set_visible(False)

        txt = 'P_fold: %.5f d' % (foldperiod)
        ax.text(0.98, 0.02, txt, horizontalalignment='right',
                verticalalignment='bottom',
                transform=ax.transAxes, fontsize='small')

        ax.set(ylabel='', xlim=[-0.1,0.1])
        axylim = ax.get_ylim()
        ax.vlines([0.], min(axylim), max(axylim), colors='red',
                linestyles='-', alpha=0.9, zorder=-1)
        if inset:
            iax.vlines([-0.5,0.5], min(iaxylim), max(iaxylim), colors='black',
                    linestyles='-', alpha=0.7, zorder=30)
            iax.vlines([0.], min(iaxylim), max(iaxylim), colors='red',
                    linestyles='-', alpha=0.9, zorder=30)



    # Figure out names and write.
    savedir = '../results/dipsearchplot/'
    if 'inj' in stage:
        savedir += 'inj/'
    elif 'inj' not in stage:
        savedir += 'no_inj/'
    plotname = str(keplerid)+'_'+ap+stage+'.png'

    f.savefig(savedir+plotname, dpi=300, bbox_inches='tight')

    LOGINFO('Made & saved whitened plot to {:s}'.format(savedir+plotname))

