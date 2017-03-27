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


