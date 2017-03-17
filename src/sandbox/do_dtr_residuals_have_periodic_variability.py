import os
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import inj_recov as ir
import time

def plot_random_quarter(times, fluxs, kicid, eb_period, fluxs_gpr=None, fluxs_std=None):

    plt.close('all')
    f, axs = plt.subplots(figsize=(16,10), nrows=3, ncols=1)
    axs[0].plot(times, fluxs, color='black', linestyle='-', marker='o', markersize=0.1, lw=0.5)
    axs[1].plot(times, fluxs, color='black', linestyle='-', marker='o', markersize=1, lw=0.5)
    axs[2].plot(times, fluxs, color='black', linestyle='-', marker='o', markersize=1, lw=0.5)

    lw = 1
    if fluxs_gpr and fluxs_std:
        X_plot = np.linspace(min(times), max(times), 10000)[:, None]
        plt.plot(X_plot, fluxs_gpr, color='darkorange', lw=lw,
                 label='GPR (%s)' % gpr.kernel_)
        plt.fill_between(X_plot[:, 0], fluxs_gpr - fluxs_std, fluxs_gpr + fluxs_std,
                 color='darkorange',
                 alpha=0.2)
    if fluxs_gpr and not fluxs_std:
        X_plot = np.linspace(min(times), max(times), 10000)[:, None]
        plt.plot(X_plot, fluxs_gpr, color='darkorange', lw=lw,
                 label='GPR (%s)' % gpr.kernel_)

    a1_xlim = axs[1].get_xlim()
    a1_xstart = np.random.uniform(low=min(a1_xlim), high=max(a1_xlim)-10)
    axs[1].set_xlim([a1_xstart, a1_xstart+10.])

    a0_xlim = axs[0].get_xlim()
    a2_xstart = np.random.uniform(low=min(a0_xlim), high=max(a0_xlim)-10)
    while abs(a2_xstart - a1_xstart)<10. : a2_xstart = np.random.uniform(low=min(a0_xlim), high=max(a0_xlim)-10)
    axs[2].set_xlim([a2_xstart, a2_xstart+10.])

    a0ylim = axs[0].get_ylim()
    axs[0].vlines([a1_xstart, a1_xstart+10.], min(a0ylim), max(a0ylim), linestyle='--')
    axs[0].vlines([a2_xstart, a2_xstart+10.], min(a0ylim), max(a0ylim), linestyle='--')
    axs[0].legend(loc='best')
    axs[0].set_ylim(a0ylim)
    axs[0].set(ylabel='redetr flux',
        title='KICID {:d}, EB_period: {:.5f} (originally subtracted)'.format(
        int(kicid), float(eb_period) ), fontsize='xx-small')

    f.tight_layout()
    fname = str(kicids[kicid_ind])+'_'+stage+'.png'
    f.savefig('../results/eb_subtraction_diagnostics/'+fname, dpi=300)


def get_data():
    '''
    Get flux vs time for a random quarter of data.
    '''

    origstage = 'dipsearch_inj'
    δ = 1/16./100. # try to ignore injection
    stage = origstage + '_' + str(δ)

    df = pd.read_csv('../results/injrecovresult/irresult_sap_top1.csv')
    kicids = np.unique(df['kicid'])
    kicid_ind = np.random.randint(low=0, high=len(kicids))
    kicid = kicids[kicid_ind]

    lcd = ir.load_lightcurve_data(kicid, stage=stage)
    allq = ir.load_allq_data(kicid, stage=stage)

    qnums = list(lcd.keys())
    qnum = np.random.randint(low=min(qnums), high=max(qnums))
    while qnum not in qnums:
        qnum = np.random.randint(low=min(qnums), high=max(qnums))

    lc = lcd[qnum]['redtr']['sap']
    fluxs = lc['fluxs'] - lc['fitfluxs_legendre']
    times = lc['times']

    kebc_period = float(lcd[list(lcd.keys())[0]]['kebwg_info']['period'])

    return times, fluxs, kicid, kebc_period


def get_fits(times, fluxs, eb_period, GP_std=True):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared

    X = np.reshape(times, (len(times),1))
    y = fluxs

    # Stealing following snippets from plot_compare_gpr_krr.py, scikit-learn
    # docs.
    # Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
    # Fit the Gaussian process with an ExpSineSquared kernel, choosing the
    # hyperparameters based on gradient-ascent on the marginal likelihood function.
    # N.b. the prior mean must be specified. in GaussianProcessRegressor,, this is
    # done by assuming it's zero if kwarg normalize_y=False, or with the training
    # data's mean if `normalize_y=True.

    gp_kernel = ExpSineSquared(eb_period/2.,
            eb_period,
            periodicity_bounds=(eb_period/5., eb_period*5),
            length_scale_bounds=(0.5*eb_period/5., 0.5*eb_period*5))
    #gp_kernel = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) \
    #    + WhiteKernel(1e-1)
    gpr = GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True)
    stime = time.time()
    print("Beginning GPR fit")
    gpr.fit(X, y)
    print("Time for GPR fitting: %.3f" % (time.time() - stime))

    X_plot = np.linspace(min(X), max(X), 10000)[:, None]

    # Predict using GPR
    if GP_std:
        stime = time.time()
        y_gpr, y_std = gpr.predict(X_plot, return_std=True)
        print("Time for GPR prediction with standard-deviation: %.3f"
              % (time.time() - stime))
    else:
        stime = time.time()
        y_gpr = gpr.predict(X_plot, return_std=False)
        print("Time for GPR prediction: %.3f" % (time.time() - stime))

    if GP_std:
        return y_gpr, y_std
    else:
        return y_gpr


def main(get_fits_flag):
    times, fluxs, kicid, eb_period = get_data()

    #implementing the GP regression here would be cool too.
    #b/c the timescale is now all unknown and stuff too
    #NOTE: GP regression is very slow, as implemented in scikit learn. So,
    #forget that idea, for now!

    #FIXME
    #phase dispersion minimization, what do we see?
    #nb: one logic for this could be: if RMS is too big to reasonably
    #find an Rp=4Rearth (6,8,10 whatever) planet, then try to continue
    #whitening.
    if get_fits_flag:
        fluxs_gpr, fluxs_std = get_fits(times, fluxs, eb_period)
        plot_random_quarter(times, fluxs, kicid, eb_period,
                fluxs_gpr=fluxs_gpr,
                fluxs_std=fluxs_std)

    else:
        plot_random_quarter(times, fluxs, kicid, eb_period)


if __name__ == '__main__':
    get_fits_flag = True

    np.random.seed(42)
    main(get_fits_flag)
