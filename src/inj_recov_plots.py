import pdb
import matplotlib
matplotlib.use('Agg')
import numpy as np, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import logging
from datetime import datetime

from numpy import nan as npnan, median as npmedian, \
    isfinite as npisfinite, min as npmin, max as npmax, abs as npabs, \
    sum as npsum, array as nparr, std as npstd

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
    dtr_txt='Fit: n=%d legendre series to phase-folded by quarter.'
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

    #plt.style.use('utils/lgb.mplstyle')

    #TODO: fix ylabels to mean correct things
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
                min_inum = np.min(list(lcd[qnum]['white'].keys()))
                lc = lcd[qnum]['white'][min_inum][ap]['legdict']['whiteseries']
                times = lc['times']
                fluxs = lc['wfluxs']
                errs = lc['errs']
            elif axix == 3:
                max_inum = np.max(list(lcd[qnum]['white'].keys()))
                lc = lcd[qnum]['white'][max_inum][ap]['legdict']['whiteseries']
                times = lc['times']
                fluxs = lc['wfluxsresid']
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
                pass
                #min_inum = np.min(list(lcd[qnum]['white'].keys()))
                #lc = lcd[qnum]['white'][min_inum][ap]['legdict']['whiteseries']
                #pfitfluxs = lc['wfluxslegfit']
                #pfittimes = lc['times']

                #ax.plot(tfittimes, tfitfluxes, c='k', linestyle='-', lw=0.5,
                #        zorder=0)
            elif axix == 2:
                min_inum = np.min(list(lcd[qnum]['white'].keys()))
                lc = lcd[qnum]['white'][min_inum][ap]['legdict']['whiteseries']
                fitfluxs = lc['wfluxslegfit']
                fittimes = lc['times']
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
        str(keplerid), ap, kebc_period) + ' (n=?? legendre series fit)')
    dtr_txt='Fit: legendre series to phase-folded by quarter.'
    ax_dtr.text(0.5,0.98, dtr_txt, horizontalalignment='center',
            verticalalignment='top', transform=ax_dtr.transAxes)
    ax_dtr.set(ylabel='normalized,\ndetrended flux')
    ax_w.hlines([0.005,-0.005], xmin, xmax,
            colors='k',
            linestyles='--',
            zorder=-20)

    w_txt='fit: n=?? legendre series.'
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
    qnums = np.sort(qnums)
    if len(lcd) < 5:
        axs_pg = axs_pg[:len(lcd)]
        axs_pf = axs_pf[:len(lcd)]

    for ix, ax in enumerate(axs_pg):
        qnum = qnums[ix]

        pgd = lcd[qnum]['white'][0][ap]['per']

        ax.plot(pgd['periods'],
                pgd['lspvals'],
                'k-')

        pwr_ylim = ax.get_ylim()
        selperiod = lcd[qnum]['white'][0][ap]['fineper']['selperiod']
        ax.vlines(selperiod, 0, 1.2, colors='r', linestyles=':', alpha=0.8, zorder=20)

        selforcedkebc = lcd[qnum]['white'][0][ap]['fineper']['selforcedkebc']
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
        pflux = lcd[qnum]['white'][0][ap]['legdict']['magseries']['mags']
        phase = lcd[qnum]['white'][0][ap]['legdict']['magseries']['phase']
        pfitflux = lcd[qnum]['white'][0][ap]['legdict']['fitinfo']['fitmags']

        thiscolor = colors[int(qnum)%len(colors)]

        ax.plot(phase, pflux, c=thiscolor, linestyle='-',
                marker='o', markerfacecolor=thiscolor,
                markeredgecolor=thiscolor, ms=0.1, lw=0.1, zorder=0)
        ax.plot(phase, pfitflux, c='k', linestyle='-',
                lw=0.5, zorder=2)

        initperiod = lcd[qnum]['white'][0][ap]['per']['selperiod']
        selperiod = lcd[qnum]['white'][0][ap]['fineper']['selperiod']

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
    if inj:
        savedir += 'inj/'
    else:
        savedir += 'real/'
    plotname = str(keplerid)+'_'+ap+stage+'.png'
    if 'eb_sbtr' in stage:
        # override for EB subtraction tests
        savedir = '../results/whitened_diagnostic/eb_subtraction/'

    f.savefig(savedir+plotname, dpi=300)

    LOGINFO('Made & saved whitened plot to {:s}'.format(savedir+plotname))


def _add_inset_axes(ax,fig,rect,axisbg='w'):
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
        binned-phase from BLS.

    Returns: nothing, but saves the plot with a smart name to
        ../results/dipsearchplot/
    '''

    #plt.style.use('utils/lgb.mplstyle')
    assert ap == 'sap' or ap == 'pdc'

    keplerid = lcd[list(lcd.keys())[0]]['objectinfo']['keplerid']
    LOGINFO('beginning dipsearch plot, KIC {:s}'.format(
        str(keplerid)))

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
    MAD = npmedian( npabs ( lc['fluxs'] - npmedian(lc['fluxs']) ) )
    ylim_raw = [np.median(lc['fluxs']) - 4*(1.48*MAD),
                np.median(lc['fluxs']) + 2*(1.48*MAD)]

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
    p = allq['inj_model'] if inj else np.nan
    injdepth = (p['params'].rp)**2 if inj else np.nan

    kebc_period = float(lcd[list(lcd.keys())[0]]['kebwg_info']['period'])
    ax_raw.get_xaxis().set_ticks([])
    xmin, xmax = min_time-timelen*0.03, max_time+timelen*0.03
    ax_raw.set(xlabel='', ylabel=ap.upper()+' relative flux',
        xlim=[xmin,xmax],
        ylim = ylim_raw)
    ax_raw.set_title(
        'KIC:{:s}, {:s}, q_flag>0, KEBC_P: {:.4f}. '.format(
        str(keplerid), ap.upper(), kebc_period)+\
        'day, inj={:s}, dtr, iter whitened. '.format(str(inj))+\
        'Depth_inj: {:.4g}'.format(injdepth),
        fontsize='xx-small')
    ax_raw.hlines([0.005,-0.005], xmin, xmax,
            colors='k',
            linestyles='--',
            zorder=-20)

    ############################
    # PHASE-FOLDED LIGHTCURVES #
    ############################
    pgdf = allq['dipfind']['bls'][ap]['finebls']
    periods_powers = [(k, max(pgdf[k]['serialdict']['lspvals'])) \
            for k in list(pgdf.keys())]
    nbestperiods = [per for (per,power) in sorted(periods_powers,
            key=lambda pair:pair[1], reverse=True)]

    for ix, ax in enumerate(axs_φ):

        if ix == len(nbestperiods):
            break
        foldperiod = nbestperiods[ix]

        # Recover quantities to plot, defined on φ=[0,1]
        fbls = allq['dipfind']['bls'][ap]['finebls'][foldperiod]
        plotφ = fbls['φ']
        plotfluxs = fbls['flux_φ']
        binplotφ = fbls['binned_φ']
        binplotfluxs = fbls['binned_flux_φ']
        φ_0 = fbls['φ_0']
        if ix == 0:
            bestφ_0 = φ_0
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

        # Compute the BLS model
        φ_ing, φ_egr = fbls['φ_ing'], fbls['φ_egr']
        δ = fbls['serialdict']['blsresult']['transdepth']

        def _get_bls_model_flux(φ, φ_0, φ_ing, φ_egr, δ, median_flux):
            flux = np.zeros_like(φ)
            for ix, phase in enumerate(φ-φ_0):
                if phase < φ_ing - φ_0 or phase > φ_egr - φ_0:
                    flux[ix] = median_flux
                else:
                    flux[ix] = median_flux - δ
            return flux

        # Overplot the BLS model
        bls_flux = _get_bls_model_flux(np.arange(-2,2,0.005), φ_0, φ_ing,
                φ_egr, δ, np.median(plotfluxs))
        ax.step(np.arange(-2,2,0.005)-φ_0, bls_flux, where='mid',
                c='red', ls='-')

        # Overlay the inset plot
        if inset:
            subpos = [0.03,0.03,0.25,0.2]
            iax = _add_inset_axes(ax,f,subpos)

            iax.scatter(plotφ-φ_0,plotfluxs,marker='o',s=2*0.1,color='gray')
            if phasebin:
                iax.scatter(binplotφ-φ_0,binplotfluxs,marker='o',s=10*0.2,
                        color='blue', zorder=20)
            iymin = np.mean(plotfluxs) - 3.5*np.std(plotfluxs)
            iymax = np.mean(plotfluxs) + 2*np.std(plotfluxs)
            iax.set_ylim([iymin, iymax])
            iaxylim = iax.get_ylim()

            iax.set(ylabel='', xlim=[-0.7,0.7])
            iax.get_xaxis().set_visible(False)
            iax.get_yaxis().set_visible(False)

        t0 = min_time + φ_0*foldperiod
        φ_dur = φ_egr - φ_ing if φ_egr > φ_ing else (1+φ_egr) - φ_ing
        T_dur = foldperiod * φ_dur

        txt = 'P_fold: {:.2f} d\n T_dur: {:.1f} hr\n t_0: {:.1f},$\phi_0$= {:.2f}'.format(
                foldperiod, T_dur*24., t0, φ_0)
        ax.text(0.98, 0.02, txt, horizontalalignment='right',
                verticalalignment='bottom',
                transform=ax.transAxes, fontsize='xx-small')

        ax.set(ylabel='', xlim=[-0.1,0.1])
        ymin = np.mean(plotfluxs) - 3*np.std(plotfluxs)
        ymax = np.mean(plotfluxs) + 1.5*np.std(plotfluxs)
        ax.set_ylim([ymin, ymax])
        axylim = ax.get_ylim()
        ax.vlines([0.],min(axylim),min(axylim)+0.05*(max(axylim)-min(axylim)),
                colors='red', linestyles='-', alpha=0.9, zorder=-1)
        ax.set_ylim([ymin, ymax])
        if inset:
            iax.vlines([-0.5,0.5], min(iaxylim), max(iaxylim), colors='black',
                    linestyles='-', alpha=0.7, zorder=30)
            iax.vlines([0.], min(iaxylim),
                    min(iaxylim)+0.05*(max(iaxylim)-min(iaxylim)),
                    colors='red', linestyles='-', alpha=0.9, zorder=-1)
            iax.set_ylim([iymin, iymax])


    ################
    # PERIODOGRAMS #
    ################
    pgdc = allq['dipfind']['bls'][ap]['coarsebls']
    pgdf = allq['dipfind']['bls'][ap]['finebls']

    ax_pg.plot(pgdc['periods'], pgdc['lspvals'], 'k-', zorder=10)

    pwr_ylim = ax_pg.get_ylim()

    pgdf = allq['dipfind']['bls'][ap]['finebls']
    periods_powers = [(k, max(pgdf[k]['serialdict']['lspvals'])) \
            for k in list(pgdf.keys())]
    nbestperiods = [per for (per,power) in sorted(periods_powers,
            key=lambda pair:pair[1], reverse=True)]
    cbestperiod = nbestperiods[0]

    # We want the FINE best period, not the coarse one (although the frequency
    # grid might actually be small enough that this doesn't improve much!)
    fbestperiod = pgdf[cbestperiod]['serialdict']['bestperiod']

    best_t0 = min_time + bestφ_0*fbestperiod
    # Show 5 _selected_ periods in red.
    ax_pg.vlines(nbestperiods, min(pwr_ylim), max(pwr_ylim), colors='r',
            linestyles='solid', alpha=1, lw=1, zorder=-5)
    # Underplot 5 best coarse periods. (Shows negative cases).
    ax_pg.vlines(pgdc['nbestperiods'][:6], min(pwr_ylim), max(pwr_ylim),
            colors='gray', linestyles='dotted', lw=1, alpha=0.7, zorder=-10)

    p = allq['inj_model'] if inj else np.nan
    injperiod = p['params'].per if inj else np.nan
    inj_t0 = p['params'].t0 if inj else np.nan
    if inj:
        ax_pg.vlines(injperiod, min(pwr_ylim), max(pwr_ylim), colors='g',
                linestyles='-', alpha=0.8, zorder=10)

    if inj:
        txt = 'P_inj: %.4f d\nP_rec: %.4f d\nt_0,inj: %.4f\nt_0,rec: %.4f' % \
              (injperiod, fbestperiod, inj_t0, best_t0)
    else:
        txt = 'P_rec: %.4f d\nt_0,rec: %.4f' % \
              (fbestperiod, best_t0)

    ax_pg.text(0.96,0.96,txt,horizontalalignment='right',
            verticalalignment='top',
            transform=ax_pg.transAxes)

    ax_pg.set(xlabel='period [d]', xscale='log')
    ax_pg.get_yaxis().set_ticks([])
    ax_pg.set(ylabel='BLS power')
    ax_pg.set(ylim=[min(pwr_ylim),max(pwr_ylim)])
    ax_pg.set(xlim=[min(pgdc['periods']),max(pgdc['periods'])])

    # Figure out names and write.
    savedir = '../results/dipsearchplot/'
    if 'inj' in stage:
        savedir += 'inj/'
    elif 'inj' not in stage:
        savedir += 'real/'
    plotname = str(keplerid)+'_'+ap+stage+'.png'

    f.savefig(savedir+plotname, dpi=300, bbox_inches='tight')

    LOGINFO('Made & saved dipsearch plot to {:s}'.format(savedir+plotname))


def plot_iterwhiten_3row(lcd, allq, ap='sap', stage='', inj=False, δ=None):

    #plt.style.use('utils/lgb.mplstyle')

    #lcd[qnum]['white'][inum][ap]['w*']`, for * in (fluxs,errs,times,phases)

    assert ap == 'sap' or ap == 'pdc'
    keplerid = lcd[list(lcd.keys())[0]]['objectinfo']['keplerid']
    kebc_period = nparr(float(lcd[list(lcd.keys())[0]]['kebwg_info']['period']))

    qnums = list(lcd.keys())
    selind = np.random.randint(low=0, high=len(qnums))
    selqnum = qnums[selind]

    for qnum in qnums:
        thisqflag, thisiflag = 0, 0
        if thisqflag > 0 :
            continue
        inums = list(lcd[qnum]['white'].keys())
        for inum in inums:
            if thisiflag > 0:
                continue
            if 'eb_sbtr' in stage:
                if not (inum == min(inums) or inum == max(inums) or \
                inum == int((max(inums)-min(inums))/2.)):
                    continue
            else:
                if (not inum == int((max(inums)-min(inums))/2.)) and \
                (not qnum == selqnum):
                    continue

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
            if inj:
                inj_period = allq['inj_model']['params'].per
            pwr_ylim = ax_pg.get_ylim()
            ax_pg.vlines(selperiod, min(pwr_ylim), max(pwr_ylim), colors='r',
                    linestyles='--', alpha=0.8, zorder=20, label='P sel')
            ax_pg.vlines(kebc_period, min(pwr_ylim), max(pwr_ylim), colors='g',
                    linestyles='--', alpha=0.8, zorder=20, label='P EB')
            if inj:
                ax_pg.vlines(inj_period, min(pwr_ylim), max(pwr_ylim),
                        colors='b', linestyles=':', alpha=0.8, zorder=20,
                        label='P CBP')
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
            duty_cycle = 0.8 # as an estimate, 80% of the data are out
            norbs = int(duty_cycle*(max(times)-min(times))/selperiod)

            txt = 'q: %d, inum: %d\nlegdeg: %d, npts: %d, norbs: %d' % (
                    int(qnum), int(inum), int(legdeg), len(pflux), norbs)
            ax_pf.text(0.98, 0.98, txt, horizontalalignment='right',
                    verticalalignment='top',
                    transform=ax_pf.transAxes)

            pf_txt = 'P EB: %.7f day\nP sel: %.7f day' % (kebc_period, selperiod)
            ax_pf.text(0.02, 0.02, pf_txt, horizontalalignment='left',
                    verticalalignment='bottom', transform=ax_pf.transAxes)
            ax_pf.set(ylabel='phased dtr flux')

            f.tight_layout()

            savedir = '../results/eb_subtraction_diagnostics/'
            if inj:
                savedir += 'inj/'
            else:
                savedir += 'real/'

            if inj:
                fname = '{:s}_qnum{:s}_inum{:s}_{:s}sap.png'.format(
                        str(keplerid), str(int(qnum)), str(int(inum)),
                        str(δ))
            else:
                fname = '{:s}_qnum{:s}_inum{:s}_sap.png'.format(
                        str(keplerid), str(int(qnum)), str(int(inum)))

            f.savefig(savedir+fname, dpi=200, bbox_inches='tight')

            LOGINFO('Made & saved 3row whitened plot to {:s}'.format(
                savedir+fname))
            thisiflag += 1

        thisqflag += 1
