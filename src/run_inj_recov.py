import numpy as np
import inj_recov as ir

def main():
    pass

if __name__ == '__main__':
    #23948
    #1111
    #1234
    np.random.seed(1234)
    lcd = ir.test_retrieve_lcs()
    lcd = ir.detrend_allquarters(lcd)
    lcd = ir.normalize_allquarters(lcd)

    ir.orosz_style_flux_vs_time(lcd, flux_to_use='sap')
    ir.orosz_style_flux_vs_time(lcd, flux_to_use='pdc')


    #TODO: implement sigclipping in reduction


######
#NOTES#
'''
TODO: notice/implement

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

 In varbase:
- fourier_fit_magseries: fit an arbitrary order Fourier series to a
                         magnitude/flux time series.
- spline_fit_magseries: fit a univariate cubic spline to a magnitude/flux time
                        series with a specified spline knot fraction.
- savgol_fit_magseries: apply a Savitzky-Golay smoothing filter to a
                        magnitude/flux time series, returning the resulting
                        smoothed function as a "fit".
- legendre_fit_magseries: fit a Legendre function of the specified order to the
                          magnitude/flux time series.

 ALSO:
 PyKE is also worth assessing.

 ALSO:
 What did the KEBC team do to detrend?
'''
