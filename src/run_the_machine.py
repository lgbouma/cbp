'''
>>> python run_the_machine.py --help
usage: run_the_machine.py [-h] [-ir] [-N NSTARS] [-p] [-inj INJ] [-frd] [-c]
                          [-kicid KICID] [-nw NWORKERS] [-q]

This is a short period EB injection-recovery machine (injection is optional).

optional arguments:
  -h, --help            show this help message and exit
  -ir, --injrecovtest   Inject and recover periodic transits for a small
                        number of trial stars. Must specify N.
  -N NSTARS, --Nstars NSTARS
                        int number of stars to inject/recov on (& RNG seed).
                        required if running injrecovtest
  -p, --pkltocsv        Process ur pkl files to csv results. Needs inj arg.
  -inj INJ, --inj INJ   1 if u want to process inj&recov results, 0 if real
                        results.
  -frd, --findrealdips  Search real short period contact EBs for transiting
                        planets
  -c, --cluster         Use this flag if you are running on a cluster.
                        Requires kicid.
  -kicid KICID, --kicid KICID
                        KIC ID of the system you want to load.
  -nw NWORKERS, --nworkers NWORKERS
                        Number of workers for MPI.
  -q, --quicklcd        if you need a quick `lcd` to play with, this option
                        returns it (useful in IPython, to easily explore the
                        data structures)
'''

import numpy as np, os
import inj_recov as ir
import inj_recov_plots as irp
import injrecovresult_analysis as irra
import argparse, socket

#############
## GLOBALS ##
#############

global HOSTNAME, DATADIR
HOSTNAME = socket.gethostname()
DATADIR = '../data/' if 'della' not in HOSTNAME else '/tigress/lbouma/data/'
PKLDIR = '/media/luke/LGB_tess_data/cbp_data_injrecov_pkl_real/' # external HD

###############
## FUNCTIONS ##
###############

def get_lcd(stage=None):
    '''
    Get the lightcurve dictionary from saved pickles for a given stage.
    '''
    datapath = PKLDIR
    pklnames = [f for f in os.listdir(datapath) if stage in f]

    kicid = 5302006

    lcd, loadfailed = ir.load_lightcurve_data(
                                kicid, stage=stage, datapath=datapath)
    if 'dipsearch' in stage or 'realsearch' in stage:
        allq, loadfailed = ir.load_allq_data(
                                kicid, stage=stage, datapath=datapath)

    if isinstance(allq, dict):
        return lcd, allq
    else:
        return lcd


def recov(inj=False, stage=None, nwhiten_max=10, nwhiten_min=1, rms_floor=5e-4,
        iwplot=False, whitened=True, ds=True, min_pf_SNR=3., kicid=None,
        nworkers=None):
    '''
    See docstring for injrecov. This does identical recovery, but has different
    enough control flow to merit a separate function.

    kwargs:
        min_pf_SNR (float): minimum phase-folded SNR with which to actually
        make plots (otherwise, do not make them to save compute time).

        Other kwargs are described in injrecov docstring.
    '''
    assert not inj
    stage = stage+'_inj' if inj else stage+'_real'
    predir = 'inj/' if 'inj' in stage else 'real/'

    keepgoing, blacklist = True, []
    while keepgoing:
        lcd, lcflag, blacklist = ir.retrieve_next_lc(stage=stage,
                blacklist=blacklist, kicid=kicid)
        if lcflag:
            if lcflag == 'finished':
                print('finished searching!')
                break
            elif lcflag == True:
                if kicid:
                  print('Caught an error, escaping.')
                  break
                else:
                  print('Caught an error, going to next LC.')
                  continue

        kicid = str(lcd[list(lcd.keys())[0]]['objectinfo']['keplerid'])

        pklmatch = [f for f in os.listdir(DATADIR+'injrecov_pkl/'+predir) if
                f.endswith('.p') and f.startswith(kicid) and stage in f]

        if len(pklmatch) > 0:
            print('Found {:s} pkl (skip processing)'.format(kicid))
        else:
            lcd = ir.detrend_allquarters(lcd, σ_clip=30., inj=inj)
            lcd = ir.normalize_allquarters(lcd, dt='dtr')
            lcd = ir.iterative_whiten_allquarters(lcd, σ_clip=[30.,5.],
                    nwhiten_max=nwhiten_max, nwhiten_min=nwhiten_min,
                    rms_floor=rms_floor, nworkers=nworkers)
            allq = {}
            allq = ir.find_dips(lcd, allq, method='bls', nworkers=nworkers)
            if 'realsearch' in stage:
                kicid = ir.save_lightcurve_data(lcd,allq=allq,stage=stage,
                        tossiterintermed=True)

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
            fblserr, results = irra.write_search_result(lcd, allq, inj=inj,
                    stage=stage)
            if fblserr:
                print('error in finebls -> got 0 len pgdf. forced continue')

        # Make plots.
        if ds and not fblserr and np.all(results['SNR_rec_pf']>min_pf_SNR):
            doneplots = os.listdir('../results/dipsearchplot/'+predir)
            plotmatches = [f for f in doneplots if f.startswith(kicid) and
                    stage in f]
            if len(plotmatches)>0:
                print('\nFound dipsearchplot, continuing.\n')
            elif 'realsearch' in stage:
                irp.dipsearchplot(lcd, allq, ap='sap', stage=stage, inj=inj)
        elif ds and fblserr:
            #need to bash-touch an empty file for control flow to not pause
            plotname = kicid+'_'+'sap'+stage+'.png'
            path = '../results/dipsearchplot/'+predir+plotname
            print('\nSince fblserr, touch {:s}.\n'.format(path))
            with open(path, 'a'):
                os.utime(path, None)

        if whitened and np.all(results['SNR_rec_pf']>min_pf_SNR):
            doneplots = os.listdir('../results/whitened_diagnostic/'+predir)
            plotmatches = [f for f in doneplots if f.startswith(kicid) and
                    stage in f]
            if len(plotmatches)>0:
                print('\nFound whitened_diagnostic, continuing.\n')
            elif 'realsearch' in stage:
                try:
                    irp.whitenedplot_6row(lcd, ap='sap', stage=stage, inj=inj)
                except:
                    print('ERR: whitenedplot_6row gave exception (kic:{:s})'.\
                        format(str(kicid)))
                    #need to bash-touch an empty file for control flow
                    plotname = kicid+'_'+'sap'+stage+'.png'
                    path = '../results/whitened_diagnostic/'+predir+plotname
                    print('\nSince 6row ERR, touch {:s}.\n'.format(path))
                    with open(path, 'a'):
                        os.utime(path, None)

        if iwplot and np.all(results['SNR_rec_pf']>min_pf_SNR):
            irp.plot_iterwhiten_3row(lcd, allq, stage=stage, inj=inj,
                        δ=δ)

        if kicid:
            print('Finished {:s}.'.format(str(kicid)))
            keepgoing = False



def injrecov(inj=True, N=None, stage=None, nwhiten_max=10, nwhiten_min=1,
        rms_floor=5e-4, iwplot=False, whitened=True, ds=True, kicid=None,
        injrecovtest=None, nworkers=None):
    '''
    Inject transits, and find dips in short period binaries in the Kepler
    Eclipsing Binary Catalog. There are two important objects: `lcd` organizes
    everything by quarter. `allq` stitches over all quarters.

    Currently implemented:

        if injrecovtest:
            inject a transit signal at δ=(1,1/2,1/4,1/8,1/16,1/32)% depth,
            anywhere from ln(P_CBP) ~ U(ln(3.5xP_EB),ln(150days))
        elif injrecov:
            inject a transit signal ln(δ) ~ U(ln(1%),ln([1/64]%)) depth,
            anywhere from ln(P_CBP) ~ U(ln(3.5xP_EB),ln(150days))
        detrend (legendre and [30,30]σ sigclip)->
        normalize (median by quarter)->
        iterative whitening via PDM period-selection and legendre fitting (both
            in phase, then in time) ->
        find dips (BLS, over all the quarters). From 3.5xP_EB to 150 days.

    args/kwargs:

        inj (bool): True if you're injecting (fixes names of things, and
        what routines to call).

        N (int): if injrecovtest, N is both the RNG seed and the number of LCs
        to inject and recov on. If injrecov (the Real Deal), it is just the RNG
        seed, passed by the calling routine, unique to this injrection/recovery
        experiment (I use a unique integer, also related to the slurm job
        array).

        stage (str): one of stages:
            'dipsearch' if doing injection recovery.
            'realsearch' if you're searching for real dips.

        nwhiten_max (int): maximum number of iterative whitenings to do.

        nwhiten_min (int): minimum number of iterative whitenings to do.

        iwplot, whitened, ds (all bools): whether to create the
            eb_subtraction_diagnostic 3row whitened plot, the whitened 6row
            plot, and dipsearch plots (all diagnostics), respectively.

        kicid (int): the KIC ID of star to inject/recover on. If injrecovtest,
            is not needed; will choose a random star. If injrecov, will use the
            kicid.

        injrecovtest (bool): whether you're running a test (bigger dips,
        easiesr recovery) or the Real Deal.
    '''
    assert inj
    # If injrecovtest, then N is both the RNG seed and the number of LCs. If
    # injrecov, it is just the RNG seed.
    np.random.seed(N)
    seeds = np.random.randint(0, 99999999, size=N)

    stage = stage+'_inj' if inj else stage+'_real'
    predir = 'inj/' if 'inj' in stage else 'real/'
    origstage = stage

    # To save disk space (since in production this is run for a large number of
    # lightcurves), impose a maximum number of pickle files to save.
    nsavedpkls, maxnpkls = len(os.listdir(DATADIR+'injrecov_pkl/'+predir)), 50
    for s in seeds:
        # Try retrieving light curve dictionary.
        if injrecovtest:
            np.random.seed(s)
            lcd, lcflag = ir.retrieve_random_lc()
        else:
            lcd, lcflag = ir.retrieve_injrecov_lc(kicid=kicid)
        if lcflag and injrecovtest:
            continue
        elif lcflag and not injrecovtest:
            print('ERR: retrieve_injrecov_lc failed.')
            break

        # Inject transits, whiten lightcurves, find dips.
        kicid = str(lcd[list(lcd.keys())[0]]['objectinfo']['keplerid'])
        if injrecovtest:
            δarr = np.array([1.,1/2.,1/4.,1/8.,1/16.,1/32.])/100.
        elif not injrecovtest:
            ln_RpbyRs = np.random.uniform(
                    low=np.log(5e-3),
                    high=np.log(0.2),
                    size=1)
            δarr = np.array( (np.e**ln_RpbyRs)**2 )

        for δ in δarr:
            # Control flow for injection & iterative whitening.
            stage = origstage + '_' + str(δ)
            pklmatch = [f for f in os.listdir(DATADIR+'injrecov_pkl/'+predir)
                    if f.endswith('.p') and f.startswith(kicid) and stage in f]
            if len(pklmatch) > 0 and injrecovtest:
                print('Found {:s}, {:f}, continue'.format(kicid, δ))
                continue
            else:
                lcd, allq = ir.inject_transit_known_depth(lcd, δ)
                lcd = ir.detrend_allquarters(lcd, σ_clip=30., inj=inj)
                lcd = ir.normalize_allquarters(lcd, dt='dtr')
                lcd = ir.iterative_whiten_allquarters(lcd, σ_clip=[30.,5.],
                        nwhiten_max=nwhiten_max, nwhiten_min=nwhiten_min,
                        rms_floor=rms_floor, nworkers=nworkers)
                allq = ir.find_dips(lcd, allq, method='bls', nworkers=nworkers)
                if 'dipsearch' in stage and injrecovtest:
                    kicid = ir.save_lightcurve_data(lcd,allq=allq,stage=stage,
                            tossiterintermed=True)
                if 'dipsearch' in stage and not injrecovtest and nsavedpkls<maxnpkls:
                    kicid = ir.save_lightcurve_data(lcd,allq=allq,stage=stage,
                            tossiterintermed=True)

        # Write results and make plots.
        for δ in δarr:
            stage = origstage + '_' + str(δ)
            if injrecovtest:
                lcd, loadfailed = ir.load_lightcurve_data(kicid, stage=stage)
            if loadfailed:
                continue
            if 'dipsearch' in stage and injrecovtest:
                allq, loadfailed = ir.load_allq_data(kicid, stage=stage)
                if loadfailed:
                    continue

            # Write results tables. (Control flow logic: this is automatically
            # done for any run. So you'd need to delete the table before ANY
            # run, and it'll reconstruct the table based on everything in the
            # saved pickles.)
            if 'dipsearch' in stage:
                #FIXME: improve write_search_result to write the correct parameters
                fblserr, results = irra.write_search_result(lcd, allq, inj=inj,
                        stage=stage)
                if fblserr:
                    print('error in finebls ->got 0 len pgdf. forced continue')

            # Save disk space by saving less junk data. In this case, only save
            # core results as a csv file, then remove the bulky pickle and do
            # not write diagnostic plots. Note that "stage" has δ in it, which
            # prevents name degeneracy.
            if nsavedpkls > maxnpkls:
                savedir = DATADIR+'injrecov_summ/'
                csvname = str(kicid)+'_'+stage+'.csv'
                results.to_csv(savedir+csvname, index=False, header=False)
                continue

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

        # Summarize results tables in a text file.
        if injrecovtest:
            irra.summarize_injrecov_result()


def pkls_to_results_csvs(inj=None):
    '''
    Process all the pickles in ../data/injrecov_pkl/inj/* csv result files.
    (This is necessary because some of the injections had different RNG seeds).
    (It's also much cleaner than mixing it with any other routines).
    '''
    subdir = 'inj' if inj else 'real'
    # I maybe messed up the names somewhere.
    stage = 'dipsearch' if inj else 'realsearch_real'

    pklnames = os.listdir(DATADIR+'injrecov_pkl/'+subdir)
    lcdnames = [pn for pn in pklnames if 'allq' not in pn]
    allqnames = [pn for pn in pklnames if 'allq' in pn]

    stage = stage+'_inj' if inj else stage
    origstage = stage

    # These files are all regenerated by this routine. Since this is done by
    # appending in irra.write_search_result, we need to delete them.
    if inj:
        fs_to_rm = ['../results/injrecovresult/irresult_sap_top1.csv',
                    '../results/injrecovresult/irresult_sap_allN.csv',
                    '../results/injrecovresult/summary.txt']
    else:
        fs_to_rm = ['../results/real_search/irresult_sap_top1.csv',
                    '../results/real_search/irresult_sap_allN.csv',
                    '../results/real_search/summary.txt',
                    '../results/real_search/candidates_sort.csv']

    for f in fs_to_rm:
        if os.path.exists(f):
            os.remove(f)

    for lcdname in lcdnames:
        kicid = lcdname.split('_')[0]
        if inj:
            δ = lcdname.split('_')[-1].split('.p')[0]
            stage = origstage + '_' + str(δ)
        lcd, loadfailed = ir.load_lightcurve_data(kicid, stage=stage)
        if loadfailed:
            continue
        allq, loadfailed = ir.load_allq_data(kicid, stage=stage)
        if loadfailed:
            continue
        fblserr, results = irra.write_search_result(lcd, allq, inj=inj,
                stage=stage)

    if inj:
        irra.summarize_injrecov_result()
    else:
        irra.summarize_realsearch_result()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='This is a short period EB injection-recovery machine '+\
                    '(injection is optional).')
    parser.add_argument('-irtest', '--injrecovtest', action='store_true',
        help='Inject and recover periodic transits for a small number of '+\
             'trial stars. Must specify N.')
    parser.add_argument('-N', '--Nstars', type=int, default=None,
        help='RNG seed for injrecov. Also, if injrecovtest, is int number of'+\
        ' stars to inject/recov on. ')
    parser.add_argument('-ir', '--injrecov', action='store_true',
        help='Inject and recover periodic transits for KEBC stars.'+\
             'Must specify kicid, and unique N.')
    parser.add_argument('-p', '--pkltocsv', action='store_true',
        help='Process ur pkl files to csv results. Needs inj arg.')
    parser.add_argument('-inj', '--inj', type=int, default=None,
        help='1 if u want to process inj&recov results, 0 if real results.')
    parser.add_argument('-frd', '--findrealdips', action='store_true',
        help='Search real short period contact EBs for transiting planets')
    parser.add_argument('-c', '--cluster', action='store_true', default=False,
        help='Use this flag if you are running on a cluster. Requires kicid.')
    parser.add_argument('-kicid', '--kicid', type=int, default=None,
        help='KIC ID of the system you want to load.')
    parser.add_argument('-nw', '--nworkers', type=int, default=None,
        help='Number of workers for MPI.')
    parser.add_argument('-q', '--quicklcd', action='store_true',
        help='if you need a quick `lcd` to play with, this option returns it'+\
             ' (useful in IPython, to easily explore the data structures)')

    args = parser.parse_args()

    if (args.injrecovtest and args.findrealdips) or (args.injrecov and
            args.findrealdips):
        parser.error('Choose either (injection&recovery) XOR findrealdips')
    if (args.quicklcd and (args.findrealdips or args.injrecovtest)):
        parser.error('quicklcd must be run without any other options')
    if (args.injrecovtest and not isinstance(args.Nstars,int)):
        parser.error('The --injrecovtest argument requires -N')
    if (args.injrecov and (not args.kicid or not isinstance(args.Nstars,int))):
        parser.error('--injrecov argument requires --kicid and -N.')
    if (args.pkltocsv and not isinstance(args.inj,int)):
        parser.error('The --pkltocsv argument requires --inj.')
    if (args.cluster and (not args.kicid or not args.nworkers)):
        parser.error('The --cluster argument requires --kicid.')
    if (args.kicid and (not args.cluster or not args.nworkers)):
        parser.error('The kicid arg currently should only be used on clusters')
    if isinstance(args.inj,int):
        if args.inj not in [0,1]:
            parser.error('--inj must be given as 0 or 1.')
    if (args.injrecovtest and args.injrecov):
        parser.error('can only do either injrecovtest XOR injrecov')

    if args.quicklcd:
        lcd, allq = get_lcd(stage='realsearch')

    if args.injrecov or args.injrecovtest:
        makeplots = True if args.injrecovtest else False
        injrecov(inj=True, N=args.Nstars, stage='dipsearch', ds=makeplots,
            whitened=makeplots, nwhiten_max=10, nwhiten_min=1, rms_floor=5e-4,
            kicid=args.kicid, nworkers=args.nworkers, injrecovtest=False)

    if args.findrealdips:
        recov(inj=False, stage='realsearch', ds=True, whitened=True,
            nwhiten_max=10, nwhiten_min=1, rms_floor=5e-4, min_pf_SNR=3.,
            kicid=args.kicid, nworkers=args.nworkers)

    if args.pkltocsv:
        pkls_to_results_csvs(inj=args.inj)
