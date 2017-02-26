import numpy as np
import inj_recov as ir

def main():
    pass

def three_lc_check():
    for s in [1111, 1234, 23948]:
        np.random.seed(s)

        lcd = ir.test_retrieve_lcs()
        lcd = ir.detrend_allquarters(lcd)
        lcd = ir.normalize_allquarters(lcd)

        ir.orosz_style_flux_vs_time(lcd, flux_to_use='sap')
        ir.orosz_style_flux_vs_time(lcd, flux_to_use='pdc')

        lcd = ir.run_periodograms_allquarters(lcd)
        lcd = ir._select_eb_period(lcd)
        #TODO
        lcd = ir.whiten_allquarters(lcd)

        ir.save_lightcurve_data(lcd)


if __name__ == '__main__':
    #three_lc_check()

    np.random.seed(1234)
    lcd = ir.test_retrieve_lcs()
    lcd = ir.detrend_allquarters(lcd)
    lcd = ir.normalize_allquarters(lcd)

    #ir.orosz_style_flux_vs_time(lcd, flux_to_use='sap')
    #ir.orosz_style_flux_vs_time(lcd, flux_to_use='pdc')

    lcd = ir.run_periodograms_allquarters(lcd)
    lcd = ir._select_eb_period(lcd)
    lcd = ir.whiten_allquarters(lcd)

    ir.save_lightcurve_data(lcd)




######
#TODO#
######
'''
*Detrend+normalize:
    Match the KEBC detrending. As-is, I think I'm leaving in trends that are
    too big.
    Sigclip, somewhere

*Run period finder:
    SPDM phase-fold on the max period it finds

*Subtract out normalized_dtr_flux(phase)

*Plot whitened flux vs time

*Matched-filter search for boxes on prewhitened flux.

*Save the lightcurve data to not have to rerun slow periodograms


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
