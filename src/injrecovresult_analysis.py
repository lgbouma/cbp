'''
Routines for analyzing the results of the injection recovery experiments.
(Where injection is optional).

>>> python inj_recovresultanalysis.py
:: Runs summarize_injrecov_result, completeness_top1_scatterplots, and
   completeness_top1_heatmap.

write_search_result:
    Append the result of a search (i.e. what best dips did it find?
    What are their parameters?) to csv files (if it's realsearch, writes nans
    in appropriate places).

summarize_injrecov_result:
    Summarizes csv files from injection/recovery into a text file

summarize_realsearch_result:
    Summarizes csv files from search for dips into a candidate list

completeness_top1_scatterplots:
    Turns csv files into pdf plots (stored in ../results/injrecovresult/plots).
    The points on these plots are individual inj/recov expts.

completeness_top1_heatmap:
    csv files -> plots (stored in ../results/real_search/plots if approriate
    arg is passed, else in ../results/injrecovresult/plots).
'''
import pandas as pd, numpy as np, os
import time, logging
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb
import subprocess
from itertools import product

#############
## LOGGING ##
#############

# setup a logger
LOGGER = None

def set_logger_parent(parent_name):
    globals()['LOGGER'] = logging.getLogger('%s.kepler' % parent_name)

def LOGDEBUG(message):
    if LOGGER:
        LOGGER.debug(message)
    elif DEBUG:
        print('%sZ [DBUG]: %s' % (datetime.utcnow().isoformat(), message))

def LOGINFO(message):
    if LOGGER:
        LOGGER.info(message)
    else:
        print('%sZ [INFO]: %s' % (datetime.utcnow().isoformat(), message))

def LOGERROR(message):
    if LOGGER:
        LOGGER.error(message)
    else:
        print('%sZ [ERR!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGWARNING(message):
    if LOGGER:
        LOGGER.warning(message)
    else:
        print('%sZ [WRN!]: %s' % (datetime.utcnow().isoformat(), message))

def LOGEXCEPTION(message):
    if LOGGER:
        LOGGER.exception(message)
    else:
        print(
            '%sZ [EXC!]: %s\nexception was: %s' % (
                datetime.utcnow().isoformat(),
                message, format_exc()
                )
            )

##################
# RESULT WRITING #
##################
def run_script(script, stdin=None):
    '''
    Run a bash script specified by the `script` string.
    Returns (stdout, stderr), raises error on non-zero return code. Spaces,
    quotes, and newlines should not raise problems (subprocess.Popen parses
    everything).
    '''
    proc = subprocess.Popen(['bash', '-c', script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    return stdout, stderr


def summarize_injrecov_result(substr=None):
    '''
    Read csvs from write_injrecov_result, and write a summary text file to
    ../results/injrecovresult/summary.txt

    The file includes: total completeness %. Rp=4Re completeness %. And
    whatever errors came up in the logs, under a particular substring key.

    Args:
        substr (str): the substring specific to src/LOGS that identifies
        whichever set of logs you want to see errors & warnings from.
    '''
    csvdir = '../results/injrecovresult/'
    csvnames = os.listdir(csvdir)
    summpath = '../results/injrecovresult/summary.txt'
    f = open(summpath, 'w')

    # First Q: what % do we recov in top 1?
    top1s = np.sort([csvdir+n for n in csvnames if 'top1' in n])

    outstrs = []
    for top1 in top1s:
        df = pd.read_csv(top1)

        findrate = len(df[df['foundinj']==True])/len(df)

        outstr = '{:s}, find rate: {:.3g}%, N={:d}'.format(
            top1, findrate*100., len(df))
        outstrs.append(outstr)

        # What % of ~4Re recovered? (depth of 1/16/100.)
        findrate = len(df[(df['foundinj']==True)&\
                          (np.isclose(df['depth_inj'],1/16./100.,atol=1e-6))]
                      ) / \
                   float(len(df[np.isclose(df['depth_inj'],1/16./100.,atol=1e-6)]))
        outstr = '{:s}, find rate ~4Re: {:.3g}%, N={:d}'.format(
            top1, findrate*100., len(df))
        outstrs.append(outstr)



    # Second: what if we allow the next-best 5 peaks?
    allNs = np.sort([csvdir+n for n in csvnames if 'allN' in n])

    for allN in allNs:
        #(iterate over apertures)
        nbestpeaks = 5
        df = pd.read_csv(allN)

        findrate = len(df[df['foundinj']==True])/(len(df)/nbestpeaks)

        outstr = '{:s}, find rate: {:.3g}%, N={:d}'.format(
            allN, findrate*100., int(len(df)/nbestpeaks))
        outstrs.append(outstr)

        # What % of ~4Re recovered? (depth of 1/16/100.)
        findrate = len(df[(df['foundinj']==True)&\
                          (np.isclose(df['depth_inj'],1/16./100.,atol=1e-6))]
                      ) / \
                   float(
                   len(df[np.isclose(df['depth_inj'],1/16./100.,atol=1e-6)])
                   /nbestpeaks)

        outstr = '{:s}, find rate ~4Re: {:.3g}%, N={:d}'.format(
            top1, findrate*100., int(len(df)/nbestpeaks))
        outstrs.append(outstr)

    # Get warnings
    wrnerrexc = []
    if isinstance(substr,str):
        wrnout, wrnerr = run_script('cat LOGS/*'+substr+'* | grep WRN')
        errout, errerr = run_script('cat LOGS/*'+substr+'* | grep ERR')
        excout, excerr = run_script('cat LOGS/*'+substr+'* | grep EXC')
        for out in [wrnout, errout, excout]:
            lines = out.decode('utf-8').split('\n')[:-1]
            wrnerrexc.append('\n')
            for l in lines:
                wrnerrexc.append(l)
    else:
        wrnerrexc.append(['WARNING! CURRENTLY NOT PARSING WARNINGS & ERRORS!'])


    writestr = ''
    now = time.strftime('%c')
    writestr = writestr + now + '\n'
    for outstr in outstrs:
        writestr=writestr+outstr+'\n'

    # Write warnings
    if isinstance(substr,str):
        writestr = writestr + '\n{:d} warnings, {:d} errs, {:d} exceptn:\n'.\
                format(
                len(wrnout.decode('utf-8').split('\n')[:-1]),
                len(errout.decode('utf-8').split('\n')[:-1]),
                len(excout.decode('utf-8').split('\n')[:-1]))
    for wee in wrnerrexc:
        writestr=writestr+wee+'\n'

    print(writestr)
    f.write(writestr)
    f.close()


def summarize_realsearch_result(substr=None, N=None):
    '''
    Summarizes csv files from search for dips into a candidate list.  Also
    write errors that came up in the logs to a text file.

    Args:
        substr (str): the substring specific to src/LOGS that identifies
        whichever set of logs you want to see errors & warnings from.
        E.g., "findrealdips", or "170327",
        N (int): number of dipsearchplot symlinks to make
    '''
    csvdir = '../results/real_search/'
    errpath = '../results/real_search/errors.txt'
    f = open(errpath, 'w')
    outstrs = []

    ####################################
    # CANDIDATE LIST BY TOP BLS RESULT #
    ####################################
    top1_path = '../results/real_search/irresult_sap_top1.csv'
    df = pd.read_csv(top1_path)

    # Sort candidate list by phase-folded SNR.
    out = df.sort_values('SNR_rec_pf', ascending=False)
    out['P_rec_by_P_EB'] = out['P_rec']/out['kebc_period']

    # Look through things w/ phase-folded SNR of at least 3, and with recovered
    # periods not within 0.001*P_EB, and recovered
    # periods not within 0.001*(5/2*P_EB)
    outind = out['SNR_rec_pf']>3
    # First term: P_rec is slightly above a multiple of P_EB. Second term:
    # P_rec is slightly below a multiple of P_EB.
    outind &= ~(((out['P_rec_by_P_EB']%1) < 5e-3) | \
                ((1-(out['P_rec_by_P_EB']%1)) < 5e-3))
    # P_rec is slightly above [1.5,2.5,3.5,...]*P_EB, and slightly below.
    outind &= ~(((out['P_rec_by_P_EB']%0.5) < 5e-3) | \
                ((0.5-(out['P_rec_by_P_EB']%0.5)) < 5e-3))
    # P_rec is [1.333,1.6666,2.3333,2.6666,3.3333,...]*P_EB
    outind &= ~(((out['P_rec_by_P_EB']%(1/3)) < 5e-3) | \
                (((1/3)-(out['P_rec_by_P_EB']%(1/3))) < 5e-3))

    wo = out[outind][['P_rec_by_P_EB','kicid','SNR_rec_pf','depth_rec','P_rec']]
    writedir = '../results/real_search/'
    wo.to_csv(writedir+'candidates_sort_top1.csv', index=False)

    #############################################
    # CANDIDATE LIST BY ALL (TOP 5) BLS RESULTS #
    #############################################
    allN_path = '../results/real_search/irresult_sap_allN.csv'
    df = pd.read_csv(allN_path)
    out = df.sort_values('SNR_rec_pf', ascending=False)
    out['P_rec_by_P_EB'] = out['P_rec']/out['kebc_period']
    outind = out['SNR_rec_pf']>3
    outind &= ~(((out['P_rec_by_P_EB']%1) < 5e-3) | \
                ((1-(out['P_rec_by_P_EB']%1)) < 5e-3))
    outind &= ~(((out['P_rec_by_P_EB']%0.5) < 5e-3) | \
                ((0.5-(out['P_rec_by_P_EB']%0.5)) < 5e-3))
    outind &= ~(((out['P_rec_by_P_EB']%(1/3)) < 5e-3) | \
                (((1/3)-(out['P_rec_by_P_EB']%(1/3))) < 5e-3))

    # Only keep the highest SNR reported for any KICID.
    _ = out[outind][['P_rec_by_P_EB','kicid','SNR_rec_pf','depth_rec','P_rec']]
    dupind = _.duplicated(subset='kicid', keep='first')
    wo = _[~dupind]

    writedir = '../results/real_search/'
    wo.to_csv(writedir+'candidates_sort_allN.csv', index=False)

    ################################################
    # Make symlinks to the best N dipsearch plots. #
    ################################################
    if isinstance(N, int):
        kicids = np.array(wo.head(n=N)['kicid'])
        for kicid in kicids:
            srcpath = '/home/luke/Dropbox/proj/cbp/results/dipsearchplot/real/'+\
                    str(kicid)+'_saprealsearch_real.png'
            dstpath = '/home/luke/Dropbox/proj/cbp/results/real_search/'+\
                    'best_dipsearch_symlinks/'+str(kicid)+'.png'
            os.symlink(srcpath, dstpath)
        print('\nSymlinked top {:d} dipsearchplots!'.format(N))

    #################
    # Get warnings. #
    #################
    wrnerrexc = []
    if isinstance(substr,str):
        wrnout, wrnerr = run_script('cat LOGS/*'+substr+'* | grep WRN')
        errout, errerr = run_script('cat LOGS/*'+substr+'* | grep ERR')
        excout, excerr = run_script('cat LOGS/*'+substr+'* | grep EXC')
        for out in [wrnout, errout, excout]:
            lines = out.decode('utf-8').split('\n')[:-1]
            wrnerrexc.append('\n')
            for l in lines:
                wrnerrexc.append(l)
    else:
        wrnerrexc.append('WARNING! CURRENTLY NOT PARSING WARNINGS & ERRORS!')

    writestr = ''
    now = time.strftime('%c')
    writestr = writestr + now + '\n'
    for outstr in outstrs:
        writestr=writestr+outstr+'\n'

    ###################
    # Write warnings. #
    ###################
    if isinstance(substr,str):
        writestr = writestr + '\n{:d} warnings, {:d} errs, {:d} exceptn:\n'.\
                format(
                len(wrnout.decode('utf-8').split('\n')[:-1]),
                len(errout.decode('utf-8').split('\n')[:-1]),
                len(excout.decode('utf-8').split('\n')[:-1]))
    for wee in wrnerrexc:
        writestr=writestr+wee+'\n'

    print(writestr)
    f.write(writestr)
    f.close()


def write_search_result(lcd, allq, inj=None, stage=None):
    '''
    Append the result of this search (i.e. what the best dips it found were,
    and what their parameters are) to csv files.

    If inj==True (i.e. it's an injection-recovery experiment) a system is
    discovered if its "fine" (rather than coarse) best period and best transit
    epoch agree with the injected ones to some precision.

    There are two files:
    csv1: rows are results with only best periodogram period for each system.
    csv2: rows are results with all 5 "best guesses" at period for each system.

    These csv files are aperture specific.
    '''

    kicid = lcd[list(lcd.keys())[0]]['objectinfo']['keplerid']
    kebc_period = float(lcd[list(lcd.keys())[0]]['kebwg_info']['period'])
    morph = float(lcd[list(lcd.keys())[0]]['kebwg_info']['morph'])
    im = allq['inj_model'] if inj else np.nan
    P_inj = im['params'].per if inj else np.nan
    t0_inj = im['params'].t0 if inj else np.nan
    δ = (im['params'].rp)**2 if inj else np.nan
    ecc = im['params'].ecc if inj else np.nan
    a = im['params'].a if inj else np.nan
    cosi = np.cos(np.deg2rad(im['params'].inc)) if inj else np.nan

    csvdir = '../results/injrecovresult/' if inj else '../results/real_search/'

    # by default, write output as nans. Relevant in cases where there are no
    # good periods found by BLS.
    results = pd.DataFrame({
            'kicid':np.nan,
            'kebc_period':np.nan,
            'morph':np.nan,
            'ap':np.nan,
            'P_inj':np.nan,
            'P_rec':np.nan,
            'coarseperiod':np.nan,
            't0_inj':np.nan,
            't0_rec':np.nan,
            'foundinj':np.nan,
            'rms_biased':np.nan,
            'depth_inj':np.nan,
            'SNR_inj_pf':np.nan,
            'depth_rec':np.nan,
            'SNR_rec_pf':np.nan,
            'baseline':np.nan,
            'Ntra_inj':np.nan,
            'Ntra_rec':np.nan,
            'ecc':np.nan,
            'a_by_Rstar':np.nan,
            'cosi':np.nan
            }, index=['0'])

    for ap in ['sap']:
        csv1name = 'irresult_'+ap+'_top1.csv'
        csv2name = 'irresult_'+ap+'_allN.csv'

        # Get minimum time for epoch zero-point.
        times = allq['dipfind']['tfe'][ap]['times']
        baseline = np.max(times) - np.min(times)
        lc = allq['dipfind']['tfe'][ap]
        min_time = np.min(times)
        fluxs = lc['fluxs']
        meanflux = np.mean(fluxs)
        rms_biased = float(np.sqrt(np.sum((fluxs-meanflux)**2) / len(fluxs)))

        # Recover best period, and corresponding BLS depth.
        pgdc = allq['dipfind']['bls'][ap]['coarsebls']
        pgdf = allq['dipfind']['bls'][ap]['finebls']
        fblserr = True if len(pgdf)==0 else False
        if fblserr:
            break
        # Use keys because the full coarseperiod list includes wrong-sign
        # transit depths. We want the nbestperiods that were selected here, in
        # BLS-power sorted order. Start with an unsorted list of tuples:
        periods_powers = [(k, max(pgdf[k]['serialdict']['lspvals'])) \
                for k in list(pgdf.keys())]
        cnbestperiods = [per for (per,power) in sorted(periods_powers,
                key=lambda pair:pair[1], reverse=True)]
        # The coarse nbestperiods are now sorted by BLS power.
        cbestperiod = cnbestperiods[0]
        fnbestperiods = [pgdf[cnbp]['serialdict']['bestperiod']
                for cnbp in cnbestperiods]
        fbestperiod = pgdf[cbestperiod]['serialdict']['bestperiod']
        bestperiod = fbestperiod

        results_list = []
        for ix, ffoldperiod in enumerate(fnbestperiods):

            cfoldperiod = cnbestperiods[ix]
            fdepth = pgdf[cfoldperiod]['serialdict']['blsresult']['transdepth']
            fbls = allq['dipfind']['bls'][ap]['finebls'][cfoldperiod]
            φ_0 = fbls['φ_0']

            t0_rec = min_time + φ_0*ffoldperiod
            P_rec = ffoldperiod

            # If the recovered period is within +/- 0.1 days of the injected
            # period, and (recovered epoch modulo recovered period) is within 
            # +/-5% of of (injected epoch modulo injected period).
            if inj:
                atol = 0.1 # days
                rtol = 0.05
                reldiff = abs((t0_rec % P_rec) - (t0_inj % P_inj)) / (t0_inj % P_inj)
                if (abs(P_inj - P_rec) < atol) and (reldiff < rtol):
                    foundinj = True
                else:
                    foundinj = False
            else:
                foundinj = np.nan

            # Estimate number of transits for pf-SNR estimate based on
            # recovered period.
            Ntra_rec = baseline/P_rec
            # Estimate number of transits for pf-SNR estimate based on
            # injected period.
            Ntra_inj = baseline/P_inj

            results = pd.DataFrame({
                    'kicid':kicid,
                    'kebc_period':kebc_period,
                    'morph':morph,
                    'ap':ap,
                    'P_inj':P_inj,
                    'P_rec':P_rec,
                    'coarseperiod':cfoldperiod,
                    't0_inj':t0_inj,
                    't0_rec':t0_rec,
                    'foundinj':foundinj,
                    'rms_biased':rms_biased,
                    'depth_inj':δ,
                    'SNR_inj_pf':δ/rms_biased*np.sqrt(Ntra_inj),
                    'depth_rec':fdepth,
                    'SNR_rec_pf':fdepth/rms_biased*np.sqrt(Ntra_rec),
                    'baseline':baseline,
                    'Ntra_inj':Ntra_inj,
                    'Ntra_rec':Ntra_rec,
                    'ecc':ecc,
                    'a_by_Rstar':a,
                    'cosi':cosi
                    }, index=['0'])

            # Write csv1 (appending if the csv file already exists)
            if ffoldperiod == bestperiod:
                if not os.path.isfile(csvdir+csv1name):
                    results.to_csv(csvdir+csv1name,
                            header=True,
                            index=False)
                else:
                    results.to_csv(csvdir+csv1name,
                            header=False,
                            index=False,
                            mode='a')

            # Write csv2
            if not os.path.isfile(csvdir+csv2name):
                results.to_csv(csvdir+csv2name,
                        header=True,
                        index=False)
            else:
                results.to_csv(csvdir+csv2name,
                        header=False,
                        index=False,
                        mode='a')

            results_list.append(results)

            LOGINFO('Wrote KIC-{:d} result to {:s} ({:s},{:.3g}day)'.format(
                kicid,csvdir,ap,ffoldperiod))

    # pd.concat(results_list) gets written to /results/injrecov_summ/, and it
    # contains each BLS result.

    if not fblserr:
        return fblserr, pd.concat(results_list)
    elif fblserr:
        return fblserr, np.nan

#########
# PLOTS #
#########

def completeness_top1_heatmap(realsearch=None):
    '''
    csv files -> plots (stored in ../results/real_search/plots if approriate
    arg is passed, else in ../results/injrecovresult/plots).

    args:
        realsearch (bool): whether you want "real search" points overplotted
        (as scatter).
    '''
    plt.style.use('utils/fancy.mplstyle')
    assert realsearch == True or realsearch == False

    fpath = '../results/injrecovresult/irresult_sap_top1.csv'
    df = pd.read_csv(fpath)

    ##############################
    # SNR (per transit) vs P_CBP #
    ##############################
    # get ones and zeros for found/not found.
    df['foundinj'][pd.isnull(df['foundinj'])] = False
    df['found'] = list(map(int, df['foundinj']))
    df['SNR'] = df['depth_inj']/df['rms_biased']
    # for half-log bins:
    #Pgrid = np.logspace(-1,3,9)
    #SNRgrid = np.logspace(-1,2,7)
    # for third-log bins:
    Pgrid = np.logspace(-1,3,13)
    SNRgrid = np.logspace(-1,3,13)
    Pbins = [(float(Pgrid[i]), float(Pgrid[i+1])) \
                    for i in range(len(Pgrid)-1)]
    SNRbins = [(float(SNRgrid[i]), float(SNRgrid[i+1])) \
                    for i in range(len(SNRgrid)-1)]

    P = np.array(df['P_inj'])
    SNR = np.array(df['SNR'])
    found = np.array(df['found'])
    results = []

    for (Pmin, Pmax), (SNRmin, SNRmax) in product(Pbins, SNRbins):
            # get inj/recov expts in this bin
            sel = (P > Pmin) & (P < Pmax)
            sel &= (SNR > SNRmin) & (SNR < SNRmax)

            thisP, thisSNR, thisFound = P[sel], SNR[sel], found[sel]
            thisDenom = len(found[sel])
            thisNum = len(thisFound[thisFound==1])

            if thisDenom > 0:
                thisFrac = thisNum/thisDenom
            else:
                thisFrac = 0.

            results.append(thisFrac)

    results = np.reshape(results, (len(Pbins),len(SNRbins)))

    plt.close('all')
    f, ax = plt.subplots()
    im = ax.pcolor(Pgrid, SNRgrid, results.T, cmap='binary')

    if realsearch:
        fpath = '../results/real_search/irresult_sap_top1.csv'
        rs = pd.read_csv(fpath)
        rs['SNR_rec'] = rs['depth_rec']/rs['rms_biased']
        ax.scatter(rs['P_rec'], rs['SNR_rec'], c='red', lw=0, marker='o',
                label='$\mathrm{real\ search\ data}$', s=5, alpha=0.6)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('$\delta/\mathrm{RMS}\ (\mathrm{per\ transit})$')
    ax.set_xlabel('$P_\mathrm{CBP}\ [\mathrm{days}]$')
    ax.set_xlim([1e-1, 1e3])
    ax.set_ylim([1e-1, 1e3])
    cb = f.colorbar(im, orientation='vertical')
    cb.set_label('$\mathrm{percent\ recovered}$')
    if realsearch:
        lg = ax.legend(loc='best', scatterpoints=1)
        lg.get_frame().set_facecolor('none')
        lg.get_frame().set_linewidth(0.)
    f.tight_layout()
    plotdir = '../results/injrecovresult/plots/' if not realsearch else \
              '../results/real_search/plots/'
    plotname = 'completeness_heatmap_top1_SNR_vs_periodcbp.pdf'
    f.savefig(plotdir+plotname)

    ###############################
    # SNR (phase-folded) vs P_CBP #
    ###############################
    Pgrid = np.logspace(-1,3,13)
    SNRgrid = np.logspace(0,3,13)
    Pbins = [(float(Pgrid[i]), float(Pgrid[i+1])) \
                    for i in range(len(Pgrid)-1)]
    SNRbins = [(float(SNRgrid[i]), float(SNRgrid[i+1])) \
                    for i in range(len(SNRgrid)-1)]

    P = np.array(df['P_inj'])
    SNR = np.array(df['SNR_rec_pf'])
    found = np.array(df['found'])
    results = []

    for (Pmin, Pmax), (SNRmin, SNRmax) in product(Pbins, SNRbins):
            # get inj/recov expts in this bin
            sel = (P > Pmin) & (P < Pmax)
            sel &= (SNR > SNRmin) & (SNR < SNRmax)

            thisP, thisSNR, thisFound = P[sel], SNR[sel], found[sel]
            thisDenom = len(found[sel])
            thisNum = len(thisFound[thisFound==1])

            if thisDenom > 0:
                thisFrac = thisNum/thisDenom
            else:
                thisFrac = 0.

            results.append(thisFrac)

    results = np.reshape(results, (len(Pbins),len(SNRbins)))

    plt.close('all')
    f, ax = plt.subplots()
    im = ax.pcolor(Pgrid, SNRgrid, results.T, cmap='binary')

    if realsearch:
        fpath = '../results/real_search/irresult_sap_top1.csv'
        rs = pd.read_csv(fpath)
        ax.scatter(rs['P_rec'], rs['SNR_rec_pf'], c='red', lw=0, marker='o',
                label='$\mathrm{real\ search\ data}$', s=5, alpha=0.6)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$\delta/\mathrm{RMS}\times\sqrt{N_\mathrm{tra}}$')
    ax.set_xlabel('$P_\mathrm{CBP}\ [\mathrm{days}]$')
    ax.set_xlim([10**(-0.5), 1e3])
    ax.set_ylim([1e0, 10**(3)])
    cb = f.colorbar(im, orientation='vertical')
    cb.set_label('$\mathrm{percent\ of\ injections\ recovered}$')
    if realsearch:
        lg = ax.legend(loc='best', scatterpoints=1)
        lg.get_frame().set_facecolor('none')
        lg.get_frame().set_linewidth(0.)
    f.tight_layout()
    plotdir = '../results/injrecovresult/plots/' if not realsearch else \
              '../results/real_search/plots/'
    plotname = 'completeness_heatmap_top1_pfSNR_vs_periodcbp.pdf'
    f.savefig(plotdir+plotname)

    ###############
    # Rp vs P_CBP #
    ###############
    # for third-log bins:
    Pgrid = np.logspace(-1,3,13)
    Rgrid = np.logspace(0,2,15)
    Pbins = [(float(Pgrid[i]), float(Pgrid[i+1])) \
                    for i in range(len(Pgrid)-1)]
    Rbins = [(float(Rgrid[i]), float(Rgrid[i+1])) \
                    for i in range(len(Rgrid)-1)]

    # compute radius
    import astropy.units as u
    import astropy.constants as c

    P = np.array(df['P_inj'])
    found = np.array(df['found'])
    R = (np.array(np.sqrt(df['depth_inj']))*1.5*u.Rsun).to(u.Rearth).value
    results = []

    for (Pmin, Pmax), (Rmin, Rmax) in product(Pbins, Rbins):
            # get inj/recov expts in this bin
            sel = (P > Pmin) & (P < Pmax)
            sel &= (R > Rmin) & (R < Rmax)

            thisP, thisR, thisFound = P[sel], R[sel], found[sel]
            thisDenom = len(found[sel])
            thisNum = len(thisFound[thisFound==1])

            if thisDenom > 0:
                thisFrac = thisNum/thisDenom
            else:
                thisFrac = 0.

            results.append(thisFrac)

    results = np.reshape(results, (len(Pbins),len(Rbins)))

    plt.close('all')
    f, ax = plt.subplots()
    im = ax.pcolor(Pgrid, Rgrid, results.T, cmap='binary')

    if realsearch:
        fpath = '../results/real_search/irresult_sap_top1.csv'
        rs = pd.read_csv(fpath)
        rs['R_rec'] = (np.array(np.sqrt(rs['depth_rec']))*1.5*u.Rsun\
                  ).to(u.Rearth).value
        ax.scatter(rs['P_rec'], rs['R_rec'], c='red', lw=0, marker='o',
                label='$\mathrm{real\ search\ data}', s=5, alpha=0.6)

    cb = f.colorbar(im, orientation='vertical')
    cb.set_label('$\mathrm{percent\ of\ injections\ recovered}$')
    if realsearch:
        lg = ax.legend(loc='best', scatterpoints=1)
        lg.get_frame().set_facecolor('none')
        lg.get_frame().set_linewidth(0.)

    ax.set_xlabel('$P_\mathrm{CBP}\ [\mathrm{days}]$')
    ax.set_ylabel('$R_p\ [R_\oplus]$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('$\mathrm{Assume}\ R_\star = 1.5R_\odot$')
    ax.set_xlim([1e-1, 1e3])
    ax.set_ylim([1e0, 1e2])

    f.tight_layout()
    plotdir = '../results/injrecovresult/plots/' if not realsearch else \
              '../results/real_search/plots/'
    plotname = 'completeness_heatmap_top1_Rp_vs_periodcbp.pdf'
    f.savefig(plotdir+plotname)

    ########################
    # Rp by Rstar vs P_CBP #
    ########################
    # for third-log bins:
    Pgrid = np.logspace(-1,3,13)
    Rgrid = np.logspace(-3,0,13)
    Pbins = [(float(Pgrid[i]), float(Pgrid[i+1])) \
                    for i in range(len(Pgrid)-1)]
    Rbins = [(float(Rgrid[i]), float(Rgrid[i+1])) \
                    for i in range(len(Rgrid)-1)]

    # get Rp by Rstar
    P = np.array(df['P_inj'])
    found = np.array(df['found'])
    R = np.array(np.sqrt(df['depth_inj']))
    results = []

    for (Pmin, Pmax), (Rmin, Rmax) in product(Pbins, Rbins):
            sel = (P > Pmin) & (P < Pmax)
            sel &= (R > Rmin) & (R < Rmax)

            thisP, thisR, thisFound = P[sel], R[sel], found[sel]
            thisDenom = len(found[sel])
            thisNum = len(thisFound[thisFound==1])

            if thisDenom > 0:
                thisFrac = thisNum/thisDenom
            else:
                thisFrac = 0.

            results.append(thisFrac)

    results = np.reshape(results, (len(Pbins),len(Rbins)))

    plt.close('all')
    f, ax = plt.subplots()
    im = ax.pcolor(Pgrid, Rgrid, results.T, cmap='binary')

    if realsearch:
        fpath = '../results/real_search/irresult_sap_top1.csv'
        rs = pd.read_csv(fpath)
        rs['R_rec'] = (np.array(np.sqrt(rs['depth_rec']))*1.5*u.Rsun\
                  ).to(u.Rearth).value
        ax.scatter(rs['P_rec'], rs['R_rec'], c='red', lw=0, marker='o',
                label='$\mathrm{real\ search\ data}', s=5, alpha=0.6)

    cb = f.colorbar(im, orientation='vertical')
    cb.set_label('$\mathrm{percent\ of\ injections\ recovered}$')
    if realsearch:
        lg = ax.legend(loc='best', scatterpoints=1)
        lg.get_frame().set_facecolor('none')
        lg.get_frame().set_linewidth(0.)

    ax.set_xlabel('$P_\mathrm{CBP}\ [\mathrm{days}]$')
    ax.set_ylabel('$R_p / R_\star$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([1e-1, 1e3])
    ax.set_ylim([4e-3, 4e-1])

    f.tight_layout()
    plotdir = '../results/injrecovresult/plots/' if not realsearch else \
              '../results/real_search/plots/'
    plotname = 'completeness_heatmap_top1_RpbyRstar_vs_periodcbp.pdf'
    f.savefig(plotdir+plotname)




def completeness_top1_scatterplots():
    fpath = '../results/injrecovresult/irresult_sap_top1.csv'
    df = pd.read_csv(fpath)

    # δ vs P_CBP
    f, ax = plt.subplots()

    ax.scatter(df['P_inj'][df['foundinj']==True],
               df['depth_inj'][df['foundinj']==True],
               c='green', lw=0, alpha=0.9)
    ax.scatter(df['P_inj'][df['foundinj']==False],
               df['depth_inj'][df['foundinj']==False],
               c='red', lw=0, alpha=0.9)

    ax.set(xlabel='$P_\mathrm{CBP}\ [\mathrm{days}]$',
           ylabel='$(R_p/R_\star)^2$',
           xscale='log',
           yscale='log')

    f.tight_layout()
    f.savefig('../results/injrecovresult/plots/completeness_top1_depth_vs_periodcbp.pdf',
            bbox_inches='tight')
    plt.close('all')

    # SNR per tra vs P_CBP
    f, ax = plt.subplots()

    ax.scatter(df['P_inj'][df['foundinj']==True],
               df['depth_inj'][df['foundinj']==True]/df['rms_biased'][df['foundinj']==True],
               c='green', lw=0, alpha=0.9)
    ax.scatter(df['P_inj'][df['foundinj']==False],
               df['depth_inj'][df['foundinj']==False]/df['rms_biased'][df['foundinj']==False],
               c='red', lw=0, alpha=0.9)

    ax.set(xlabel='$P_\mathrm{CBP}\ [\mathrm{days}]$',
           ylabel='$\delta_\mathrm{inj}/\mathrm{RMS\ (per\ transit)}$',
           xscale='log',
           yscale='log')

    f.tight_layout()
    f.savefig('../results/injrecovresult/plots/completeness_top1_SNR_vs_periodcbp.pdf',
            bbox_inches='tight')
    plt.close('all')

    # SNR phase-folded (ish) vs P_CBP
    f, ax = plt.subplots()

    ax.scatter(df['P_inj'][df['foundinj']==True],
               df['SNR_inj_pf'][df['foundinj']==True],
               c='green', lw=0, alpha=0.9)
    ax.scatter(df['P_inj'][df['foundinj']==False],
               df['SNR_inj_pf'][df['foundinj']==False],
               c='red', lw=0, alpha=0.9)

    ax.set(xlabel='$P_\mathrm{CBP}\ [\mathrm{days}]$',
           ylabel=r'$\delta_\mathrm{inj}/\mathrm{RMS}\times\sqrt{N_\mathrm{tra}}$',
           xscale='log',
           yscale='log')

    f.tight_layout()
    f.savefig('../results/injrecovresult/plots/completeness_top1_pfSNR_vs_periodcbp.pdf',
            bbox_inches='tight')
    plt.close('all')

    # Rp/Rstar vs P_CBP
    f, ax = plt.subplots()

    ax.scatter(df['P_inj'][df['foundinj']==True],
               np.sqrt(df['depth_inj'][df['foundinj']==True]),
               c='green', lw=0, alpha=0.9)
    ax.scatter(df['P_inj'][df['foundinj']==False],
               np.sqrt(df['depth_inj'][df['foundinj']==False]),
               c='red', lw=0, alpha=0.9)

    ax.set(xlabel='$P_\mathrm{CBP}\ [\mathrm{days}]$',
           ylabel='$R_p/R_\star$',
           xscale='log',
           yscale='log')

    f.tight_layout()
    f.savefig('../results/injrecovresult/plots/completeness_top1_RpbyRs_vs_periodcbp.pdf',
            bbox_inches='tight')
    plt.close('all')

    # Rp [R_earth] vs P_CBP
    import astropy.units as u
    import astropy.constants as c

    f, ax = plt.subplots()

    δdet = np.array(np.sqrt(df['depth_inj'][df['foundinj']==True]))*1.5*u.Rsun
    δnotdet = np.array(np.sqrt(df['depth_inj'][df['foundinj']==False]))*1.5*u.Rsun

    ax.scatter(df['P_inj'][df['foundinj']==True],
               δdet.to(u.Rearth),
               c='green', lw=0, alpha=0.9)
    ax.scatter(df['P_inj'][df['foundinj']==False],
               δnotdet.to(u.Rearth),
               c='red', lw=0, alpha=0.9)

    ax.set(xlabel='$P_\mathrm{CBP}\ [\mathrm{days}]$',
           ylabel='$R_p\ [R_\oplus]$',
           title='$\mathrm{Assume}\ R_\star = 1.5R_\odot$',
           xscale='log',
           yscale='log')

    f.tight_layout()
    f.savefig('../results/injrecovresult/plots/completeness_top1_Rp_vs_periodcbp.pdf',
            bbox_inches='tight')
    plt.close('all')

    # Morph vs RMS
    import astropy.units as u
    import astropy.constants as c

    f, ax = plt.subplots()

    ax.scatter(df['rms_biased'][df['foundinj']==True],
               df['morph'][df['foundinj']==True],
               c='green', lw=0, alpha=0.9)
    ax.scatter(df['rms_biased'][df['foundinj']==False],
               df['morph'][df['foundinj']==False],
               c='red', lw=0, alpha=0.9)

    ax.set(xlabel='$\mathrm{RMS}$',
           ylabel='$\mathrm{morph}$',
           xscale='log',
           yscale='linear')

    f.tight_layout()
    plt.savefig('../results/injrecovresult/plots/completeness_top1_morph_vs_RMS.pdf',
            bbox_inches='tight')
    plt.close('all')

if __name__ == '__main__':
    summarize_injrecov_result(substr='completeness_test')
    summarize_realsearch_result(substr='findrealdips_170327', N=20)

    completeness_top1_scatterplots()
    completeness_top1_heatmap(realsearch=False)

    completeness_top1_heatmap(realsearch=True)

