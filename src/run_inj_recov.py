import numpy as np, os
import inj_recov as ir


def N_lc_check(N, ors=True, whitened=True):
    seeds = np.random.randint(0, 99999999, size=N)
    for s in seeds:
        np.random.seed(s)

        lcd = ir.retrieve_random_lc()
        kicid = str(lcd[list(lcd.keys())[0]]['objectinfo']['keplerid'])
        dones = os.listdir('../results/whitened_diagnostic/')
        matches = [f for f in dones if f.startswith(kicid)]
        if len(matches)>0:
            continue

        lcd = ir.detrend_allquarters(lcd)
        lcd = ir.normalize_allquarters(lcd)
        lcd = ir.run_periodograms_allquarters(lcd)
        lcd = ir.select_eb_period(lcd)
        lcd = ir.whiten_allquarters(lcd)
        kicid = ir.save_lightcurve_data(lcd, stage='pw')

        lcd = ir.load_lightcurve_data(kicid, stage='pw')

        if ors:
            ir.orosz_style_flux_vs_time(lcd, flux_to_use='sap')
            ir.orosz_style_flux_vs_time(lcd, flux_to_use='pdc')

        if whitened:
            ir.whitenedplot(lcd, ap='sap')
            ir.whitenedplot(lcd, ap='pdc')


def three_lc_check():
    for s in [1111, 1234, 23948]:
        np.random.seed(s)

        lcd = ir.retrieve_random_lc()
        lcd = ir.detrend_allquarters(lcd)
        lcd = ir.normalize_allquarters(lcd)
        lcd = ir.run_periodograms_allquarters(lcd)
        lcd = ir.select_eb_period(lcd)
        lcd = ir.whiten_allquarters(lcd)
        ir.save_lightcurve_data(lcd, stage='pw')


def one_lc_check():
    np.random.seed(1234)

    lcd = ir.retrieve_random_lc()
    lcd = ir.detrend_allquarters(lcd)
    lcd = ir.normalize_allquarters(lcd)
    lcd = ir.run_periodograms_allquarters(lcd)
    lcd = ir.select_eb_period(lcd)
    lcd = ir.whiten_allquarters(lcd)
    ir.save_lightcurve_data(lcd, stage='pw')


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

    #three_lc_check()
    #one_lc_check()
    #plots()

    N_lc_check(10)




######
#TODO#
######
'''
Immediate (Monday):
* Add sigclip;
* add "gapfind" to astrokep (e.g., from lcmath) (& drop points near gaps);
* broader "whitening" (subtract more/all freqs);
* harder whitened flux ylims
* Debug random quarter number selection to be robust w/ N_lc_check



*Detrend+normalize:
    Match the KEBC detrending. As-is, I think I'm leaving in trends that are
    too big.
    Sigclip, somewhere

*Period finder:
    Assess for what fraction we need to revert to KEBC period.

*Plot whitened flux vs time

*Matched-filter search for boxes on prewhitened flux.


astrokep.filter_kepler_lcdict: basically should do what's been implemented in
 `detrend_lightcurve`, and `normalize_lightcurve`.

astrokep.find_lightcurve_gaps

 ALSO:
lcmath.find_lc_timegroups -- basically an implementation of astrokep's
 find_lightcurve_gaps, already done.



astrokep.stitch_lightcurve_gaps

astrokep.keplerflux_to_keplermag

lcmath.sigclip_magseries -- can be rewritten with the appropriate
 asymmetric sigma clipping.

ALSO:
 PyKE is also worth assessing.
'''
