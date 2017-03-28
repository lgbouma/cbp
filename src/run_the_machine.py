'''
>>> python run_the_machine.py --help
usage: run_the_machine.py [-h] [-ir] [-N NSTARS] [-p] [-frd] [-q]

This is a short period EB injection-recovery machine (injection is optional).

optional arguments:
  -h, --help            show this help message and exit
  -ir, --injrecovtest   inject and recover periodic transits for a small
                        number of trial stars. must specify N.
  -N NSTARS, --Nstars NSTARS
                        int number of stars to inject/recov on (& RNG seed).
                        required if running injrecovtest
  -p, --pkltocsv        process all the pkl files made by injrecovtest to csv
                        results
  -frd, --findrealdips  search real short period contact EBs for transiting
                        planets
  -q, --quicklcd        if you need a quick `lcd` to play with, this option
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


def recov(inj=False,
        stage=None,
        nwhiten_max=10,
        nwhiten_min=1,
        rms_floor=5e-4,
        iwplot=False,
        whitened=True,
        ds=True
        ):
    '''
    See docstring for injrecov. This does identical recovery, but has different
    enough control flow to merit a separate function.
    '''
    assert not inj
    stage = stage+'_inj' if inj else stage+'_real'
    predir = 'inj/' if 'inj' in stage else 'real/'

    keepgoing, blacklist = True, []
    while keepgoing:
        lcd, lcflag, blacklist = ir.retrieve_next_lc(stage=stage,
                blacklist=blacklist)
        if lcflag:
            if lcflag == 'finished':
                print('finished searching!')
            elif lcflag == True:
                print('broke out of realsearch early.')
            break

        kicid = str(lcd[list(lcd.keys())[0]]['objectinfo']['keplerid'])

        pklmatch = [f for f in os.listdir('../data/injrecov_pkl/'+predir) if
                f.endswith('.p') and f.startswith(kicid) and stage in f]

        if len(pklmatch) > 0:
            print('Found {:s} pkl (skip processing)'.format(kicid))
        else:
            lcd = ir.detrend_allquarters(lcd, σ_clip=30., inj=inj)
            lcd = ir.normalize_allquarters(lcd, dt='dtr')
            lcd = ir.iterative_whiten_allquarters(lcd, σ_clip=[30.,5.],
                    nwhiten_max=nwhiten_max, nwhiten_min=nwhiten_min,
                    rms_floor=rms_floor)
            allq = {}
            allq = ir.find_dips(lcd, allq, method='bls')
            if 'realsearch' in stage:
                kicid = ir.save_lightcurve_data(lcd,allq=allq,stage=stage)

        # Write results and make plots.
        lcd, loadfailed = ir.load_lightcurve_data(kicid, stage=stage)
        if loadfailed:
            print('broke out of realsearch at load_lightcurve_data.')
            break
        if 'realsearch' in stage:
            allq, loadfailed = ir.load_allq_data(kicid, stage=stage)
            if loadfailed:
                print('broke out of realsearch at load_allq_data.')
                continue

        # Append results to tables. If you want to rewrite (e.g., because
        # you've multiple-append the same ones) run inj_recovresultanalysis.py
        if 'realsearch' in stage:
            fblserr = irra.write_search_result(lcd, allq, inj=inj, stage=stage)
            if fblserr:
                print('error in finebls -> got 0 len pgdf. forced continue')

        # Make plots.
        if ds and not fblserr:
            doneplots = os.listdir('../results/dipsearchplot/'+predir)
            plotmatches = [f for f in doneplots if f.startswith(kicid) and
                    stage in f]
            if len(plotmatches)>0:
                print('\nFound dipsearchplot, continuing.\n')
            elif 'realsearch' in stage:
                irp.dipsearchplot(lcd, allq, ap='sap', stage=stage, inj=inj)

        if whitened:
            doneplots = os.listdir('../results/whitened_diagnostic/'+predir)
            plotmatches = [f for f in doneplots if f.startswith(kicid) and
                    stage in f]
            if len(plotmatches)>0:
                print('\nFound whitened_diagnostic, continuing.\n')
            elif 'realsearch' in stage:
                try:
                    irp.whitenedplot_6row(lcd, ap='sap', stage=stage, inj=inj)
                except:
                    print('whitenedplot_6row gave exception (kic:{:s})'.format(
                        str(keplerid)))

        if iwplot:
            irp.plot_iterwhiten_3row(lcd, allq, stage=stage, inj=inj,
                        δ=δ)

        # Summarize results tables in a text file!
        #FIXME write an analog of this that summarizes result from realsearch.
        #irra.summarize_injrecov_result()


def injrecov(inj=True,
        N=None,
        stage=None,
        nwhiten_max=10,
        nwhiten_min=1,
        rms_floor=5e-4,
        iwplot=False,
        whitened=True,
        ds=True
        ):
    '''
    Inject transits, and find dips in short period binaries in the Kepler
    Eclipsing Binary Catalog. There are two important objects: `lcd` organizes
    everything by quarter. `allq` stitches over all quarters.

    If doing injection & recovery, results get written to
    '../results/injrecovresult'.

    If doing a real search for dips, results get written to
    '../results/real_search'.

    Currently implemented:

        inject a realistic transit signal at δ=(1,1/2,1/4,1/8,1/16,1/32)%
            depth, anywhere from P_CBP=(4-80)x P_EB.
        detrend (legendre and [30,30]σ sigclip)->
        normalize (median by quarter)->
        iterative whitening via PDM period-selection and legendre fitting (both
            in phase, then in time) ->
        find dips (BLS, over all the quarters)

    args/kwargs:

        inj (bool): True if you're injecting (fixes names of things, and
        what routines to call).

        N (int): the number of entries to inject/recover over. Also serves as
        the RNG seed (for random selection of the targets). Only needed if
        inj==True.

        stage (str): one of stages:
            'pw' if post-whitening.
            'dipsearch' if doing injection recovery.
            'realsearch' if you're searching for real dips.

        nwhiten_max (int): maximum number of iterative whitenings to do.

        nwhiten_min (int): minimum number of iterative whitenings to do.

        iwplot, whitened, ds (all bools): whether to create the
            eb_subtraction_diagnostic 3row whitened plot, the whitened 6row
            plot, and dipsearch plots (all diagnostics), respectively.
    '''

    assert inj
    np.random.seed(N)
    seeds = np.random.randint(0, 99999999, size=N)

    stage = stage+'_inj' if inj else stage+'_real'
    predir = 'inj/' if 'inj' in stage else 'real/'
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
                        rms_floor=rms_floor)
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
                irra.write_search_result(lcd, allq, inj=inj, stage=stage)

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
    # appending in irra.write_search_result, we need to delete them.
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
        irra.write_search_result(lcd, allq, inj=inj, stage=stage)

    irra.summarize_injrecov_result()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='This is a short period EB injection-recovery machine '+\
                    '(injection is optional).')
    parser.add_argument('-ir', '--injrecovtest', action='store_true',
        help='inject and recover periodic transits for a small number of '+\
             'trial stars. must specify N.')
    parser.add_argument('-N', '--Nstars', type=int, default=None,
        help='int number of stars to inject/recov on (& RNG seed). '+\
             'required if running injrecovtest')
    parser.add_argument('-p', '--pkltocsv', action='store_true',
        help='process all the pkl files made by injrecovtest to csv results')
    parser.add_argument('-frd', '--findrealdips', action='store_true',
        help='search real short period contact EBs for transiting planets')
    parser.add_argument('-q', '--quicklcd', action='store_true',
        help='if you need a quick `lcd` to play with, this option returns it'+\
             ' (useful in IPython, to easily explore the data structures)')

    args = parser.parse_args()

    if (args.injrecovtest and args.findrealdips):
        parser.error('Choose either (injection&recovery) XOR findrealdips')
    if (args.quicklcd and (args.findrealdips or args.injrecovtest)):
        parser.error('quicklcd must be run without any other options')
    if (args.injrecovtest and not isinstance(N,int)):
        parser.error('The --injrecovtest argument requires -N')

    if args.quicklcd:
        lcd = get_lcd(stage='dtr', inj=False)
        lcd, allq = get_lcd(stage='dipsearch', inj=True)

    if args.injrecovtest:
        injrecov(inj=True, N=args.Nstars, stage='dipsearch', ds=True,
                whitened=True, nwhiten_max=8, nwhiten_min=1, rms_floor=5e-4)

    if args.findrealdips:
        recov(inj=False, stage='realsearch', ds=True, whitened=True,
                nwhiten_max=10, nwhiten_min=1, rms_floor=5e-4)

    if args.pkltocsv:
        pkls_to_results_csvs()
