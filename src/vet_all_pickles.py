'''
The "dipsearchplots" and "whitened_diagnostics" made immediately from the
pipeline are insufficient to rule out many cases.
'''

def dipsearchplot(lcd, allq, ap='sap', inj=False, varepoch='bls',
        phasebin=0.002, inset=True):
    '''
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
    plotname = str(keplerid)+'_'+ap+stage+'.png'

    f.savefig(savedir+plotname, dpi=300, bbox_inches='tight')

    LOGINFO('Made & saved dipsearch plot to {:s}'.format(savedir+plotname))


