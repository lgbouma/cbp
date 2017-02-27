import numpy as np, os
import inj_recov as ir


def N_lc_check(N, ors=False, whitened=True, stage='redtr'):
    '''
    Run LC processing on N KEBC objects.

    stage:
        'pw' if post-whitening. 'redtr' if post-redetrending.
    '''
    np.random.seed(42)
    seeds = np.random.randint(0, 99999999, size=N)
    for s in seeds:
        np.random.seed(s)

        lcd = ir.retrieve_random_lc()
        kicid = str(lcd[list(lcd.keys())[0]]['objectinfo']['keplerid'])

        pwdumpmatch = [f for f in os.listdir('../data/injrecov_pkl/') if
                f.endswith('.p') and f.startswith(kicid)]

        if len(pwdumpmatch) < 1:

            lcd = ir.detrend_allquarters(lcd, σ_clip=5., legendredeg=20)
            lcd = ir.normalize_allquarters(lcd, dt='dtr')
            lcd = ir.run_periodograms_allquarters(lcd)
            lcd = ir.select_eb_period(lcd, fine=False)
            lcd = ir.run_fineperiodogram_allquarters(lcd)
            lcd = ir.select_eb_period(lcd, fine=True)
            lcd = ir.whiten_allquarters(lcd, σ_clip=5.)
            #kicid = ir.save_lightcurve_data(lcd, stage='pw')
            lcd = ir.redetrend_allquarters(lcd, σ_clip=5., legendredeg=20)
            lcd = ir.normalize_allquarters(lcd, dt='redtr')
            kicid = ir.save_lightcurve_data(lcd, stage=stage)

        #lcd = ir.load_lightcurve_data(kicid, stage='pw')
        lcd = ir.load_lightcurve_data(kicid, stage=stage)

        dones = os.listdir('../results/whitened_diagnostic/')
        matches = [f for f in dones if f.startswith(kicid)]
        if len(matches)>0:
            print('Found whitened_diagnostic, continuing.')
            continue

        if ors:
            ir.orosz_style_flux_vs_time(lcd, flux_to_use='sap')
            ir.orosz_style_flux_vs_time(lcd, flux_to_use='pdc')

        if whitened:
            if stage == 'pw':
                ir.whitenedplot_5row(lcd, ap='sap')
                ir.whitenedplot_5row(lcd, ap='pdc')
            elif stage == 'redtr':
                ir.whitenedplot_6row(lcd, ap='sap')
                ir.whitenedplot_6row(lcd, ap='pdc')

def plots(os=False, whitened=True):

    kicid = 12016304
    # load in post-whitening
    lcd = ir.load_lightcurve_data(kicid, stage='pw')

    if os:
        ir.orosz_style_flux_vs_time(lcd, flux_to_use='sap')
        ir.orosz_style_flux_vs_time(lcd, flux_to_use='pdc')

    if whitened:
        ir.whitenedplot(lcd, ap='sap')
        ir.whitenedplot(lcd, ap='pdc')


if __name__ == '__main__':

    N_lc_check(20, stage='redtr')




######
#TODO#
######
'''
immediate:
* find all "time chunks". fit out the "smooth-in"

* implement trim_near_gaps. (nb. requires "stitching" to get full LC)

* add "gapfind" to astrokep (e.g., from lcmath) & then drop points near gaps
* harder whitened flux ylims

* broader "whitening" (subtract more/all freqs);
* might want to apply a low-pass (frequency) filter to filter out
high-freuqency stellar variability?


*Detrend+normalize:
    Match the KEBC detrending? As-is, I think I'm leaving in trends that are
    too big.

*Period finder:
    Assess for what fraction we need to revert to KEBC period.

*Matched-filter search for boxes on prewhitened flux.

astrokep.filter_kepler_lcdict: basically should do what's been implemented in
 `detrend_lightcurve`, and `normalize_lightcurve`.

astrokep.find_lightcurve_gaps

 ALSO:
lcmath.find_lc_timegroups -- basically an implementation of astrokep's
 find_lightcurve_gaps, already done.

astrokep.stitch_lightcurve_gaps

astrokep.keplerflux_to_keplermag

ALSO:
 PyKE is also worth assessing.
'''
