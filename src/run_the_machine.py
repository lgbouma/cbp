'''
>>> python run_the_machine.py --help
usage: run_the_machine.py [-h] [-p] [-ir] [-q]

This is a short period EB injection-recovery machine (injection is optional).

optional arguments:
  -h, --help           show this help message and exit
  -ir, --injrecovtest  inject and recover periodic transits for a small number
                       of trial stars
  -p, --pkltocsv       process all the pkl files made by injrecovtest to csv
                       results
  -q, --quicklcd       if you need a quick `lcd` to play with, this option
                       returns it (useful in IPython, to easily explore the
                       data structures)
'''

import numpy as np, os
import inj_recov as ir
import inj_recov_plots as irp
import injrecovresult_analysis as irra
import argparse

def get_lcd(stage='redtr', inj=None, allq=None):
    '''
    Get the lightcurve dictionary from saved pickles for a given stage and
    injection mode. (Random injected depth).
    '''
    datapath = '../data/injrecov_pkl/'
    if inj:
        datapath += 'inj/'
        stage += '_inj'
    else:
        datapath += 'no_inj/'
    pklnames = [f for f in os.listdir(datapath) if stage in f]

    kicids = np.unique([pn.split('_')[0] for pn in pklnames])
    sind = np.random.randint(0, len(kicids))
    kicid = kicids[sind]

    lcd, loadfailed = ir.load_lightcurve_data(kicid, stage=stage, δ='whatever')
    if 'dipsearch' in stage:
        allq, loadfailed = ir.load_allq_data(kicid, stage=stage, δ='whatever')

    if isinstance(allq, dict):
        return lcd, allq
    else:
        return lcd


def injrecov_test1(N,
        whitened=True, ds=True,
        stage=None,
        inj=None,
        iwplot=False,
        nwhiten_max=8,
        nwhiten_min=1):
    '''
    Inject transits and recover them on N entries from the Kepler Eclipsing
    Binary Catalog. There are two important objects:
    `lcd` organizes everything by quarter. `allq` stitches over all quarters.
    Results get written to '../results/injrecovresult'

    Currently implemented:

        inject a realistic transit signal at δ=(2,1,1/2,1/4,1/8,1/16,1/32)%
            depth, anywhere from P_CBP=(4-80)x P_EB.
        detrend (legendre and [30,30]σ sigclip)->
        normalize (median by quarter)->
        iterative whitening via PDM period-selection and legendre fitting (both
            in phase, then in time) ->
        find dips (BLS, over all the quarters)

    Args:
        N (int): the number of entries to inject/recover over. Also serves as
        the RNG seed (for random selection of the targets).

        stage (str): one of stages:
            'pw' if post-whitening.
            'redtr' if post-redetrending.
            'dipsearch' if doing injection recovery.

        nwhiten_max (int): maximum number of iterative whitenings to do.

        nwhiten_min (int): minimum number of iterative whitenings to do.

        inj (bool): True if you're injecting (fixes names of things).

        whitened (bool), ds (bool): whether to create the whitened 6row and
            dipsearch plots (both diagnostics), respectively.
    '''

    np.random.seed(N)
    seeds = np.random.randint(0, 99999999, size=N)

    stage = stage+'_inj' if inj else stage
    predir = 'inj/' if 'inj' in stage else 'no_inj/'
    origstage = stage

    for s in seeds:
        np.random.seed(s)

        lcd, lcflag = ir.retrieve_random_lc()
        kicid = str(lcd[list(lcd.keys())[0]]['objectinfo']['keplerid'])
        if lcflag:
            continue

        # Inject transits, whiten lightcurves, find dips.
        δarr = np.array([1.,1/2.,1/4.,1/8.,1/16.,1/32.])/100.
        for δ in δarr:
            # Control flow for injection & iterative whitening.
            stage = origstage + '_' + str(δ)
            pklmatch = [f for f in os.listdir('../data/injrecov_pkl/'+predir) if
                    f.endswith('.p') and f.startswith(kicid) and stage in f]
            if len(pklmatch) > 0:
                print('Found {:s}, {:f}, continue'.format(kicid, δ))
                continue
            else:
                lcd, allq = ir.inject_transit_known_depth(lcd, δ)
                lcd = ir.detrend_allquarters(lcd, σ_clip=30., inj=inj)
                lcd = ir.normalize_allquarters(lcd, dt='dtr')
                lcd = ir.iterative_whiten_allquarters(lcd, σ_clip=[30.,5.],
                        nwhiten_max=nwhiten_max, nwhiten_min=nwhiten_min,
                        rms_floor=5e-4)
                if 'eb_sbtr' in stage:
                    kicid = ir.save_lightcurve_data(lcd,allq=allq,stage=stage)
                allq = ir.find_dips(lcd, allq, method='bls')
                if 'dipsearch' in stage:
                    kicid = ir.save_lightcurve_data(lcd,allq=allq,stage=stage)

        # Write results and make plots.
        for δ in δarr:
            stage = origstage + '_' + str(δ)
            lcd, loadfailed = ir.load_lightcurve_data(kicid, stage=stage)
            if loadfailed:
                continue
            if 'dipsearch' in stage:
                allq, loadfailed = ir.load_allq_data(kicid, stage=stage)
                if loadfailed:
                    continue

            # Write results tables. (Control flow logic: this is automatically
            # done for any run. So you'd need to delete the table before ANY
            # run, and it'll reconstruct the table based on everything in the
            # saved pickles.)
            if 'dipsearch' in stage:
                if inj:
                    irra.write_injrecov_result(lcd, allq, stage=stage)

            # Make plots.
            if ds:
                doneplots = os.listdir('../results/dipsearchplot/'+predir)
                plotmatches = [f for f in doneplots if f.startswith(kicid) and
                        stage in f]
                if len(plotmatches)>0:
                    print('\nFound dipsearchplot, continuing.\n')
                    continue
                if 'dipsearch' in stage:
                    irp.dipsearchplot(lcd, allq, ap='sap', stage=stage, inj=inj)

            if whitened:
                doneplots = os.listdir('../results/whitened_diagnostic/'+predir)
                plotmatches = [f for f in doneplots if f.startswith(kicid) and
                        stage in f]
                if len(plotmatches)>0:
                    print('\nFound whitened_diagnostic, continuing.\n')
                    continue
                if 'pw' in stage:
                    irp.whitenedplot_5row(lcd, ap='sap', stage=stage)
                elif 'redtr' in stage:
                    irp.whitenedplot_6row(lcd, ap='sap', stage=stage, inj=inj)
                elif 'dipsearch' in stage:
                    try:
                        irp.whitenedplot_6row(lcd, ap='sap', stage=stage, inj=inj)
                    except:
                        with open('LOGS/error.txt', 'a') as f:
                            f.write('whitenedplot_6row passed exception')
                        continue

            if iwplot:
                    irp.plot_iterwhiten_3row(lcd, allq, stage=stage, inj=inj,
                            δ=δ)

        # Summarize results tables in a text file!
        if inj:
            irra.summarize_injrecov_result()


def pkls_to_results_csvs(stage='dipsearch', inj=True):
    '''
    Process all the pickles in ../data/injrecov_pkl/inj/* csv result files.
    (This is necessary because some of the injections had different RNG seeds).
    '''
    pklnames = os.listdir('../data/injrecov_pkl/inj')
    lcdnames = [pn for pn in pklnames if 'allq' not in pn]
    allqnames = [pn for pn in pklnames if 'allq' in pn]

    stage = stage+'_inj' if inj else stage
    origstage = stage

    # These files are all regenerated by this routine. Since this is done by
    # appending in irra.write_injrecov_result, we need to delete them.
    fs_to_rm = ['../results/injrecovresult/irresult_sap_top1.csv',
                '../results/injrecovresult/irresult_sap_allN.csv',
                '../results/injrecovresult/summary.txt']
    for f in fs_to_rm:
        if os.path.exists(f):
            os.remove(f)

    for lcdname in lcdnames:
        kicid = lcdname.split('_')[0]
        δ = lcdname.split('_')[-1].split('.p')[0]
        stage = origstage + '_' + str(δ)
        lcd, loadfailed = ir.load_lightcurve_data(kicid, stage=stage)
        if loadfailed:
            continue
        allq, loadfailed = ir.load_allq_data(kicid, stage=stage)
        if loadfailed:
            continue
        irra.write_injrecov_result(lcd, allq, stage=stage)

    irra.summarize_injrecov_result()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='This is a short period EB injection-recovery machine '+\
                    '(injection is optional).')
    parser.add_argument('-ir', '--injrecovtest', action='store_true',
        help='inject and recover periodic transits for a small number of '+\
             'trial stars')
    parser.add_argument('-p', '--pkltocsv', action='store_true',
        help='process all the pkl files made by injrecovtest to csv results')
    parser.add_argument('-q', '--quicklcd', action='store_true',
        help='if you need a quick `lcd` to play with, this option returns it'+\
             ' (useful in IPython, to easily explore the data structures)')

    args = parser.parse_args()

    if args.quicklcd:
        lcd = get_lcd(stage='dtr', inj=False)
        lcd, allq = get_lcd(stage='dipsearch', inj=True)

    if args.injrecovtest:
        injrecov_test1(103, stage='dipsearch', inj=True, ds=True, whitened=True)

    if args.pkltocsv:
        pkls_to_results_csvs()
