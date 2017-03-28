'''
Routines for analyzing the results of the injection recovery experiments.

>>> python inj_recovresultanalysis.py
:: Runs summarize_injrecov_result and completeness_top1_plots.

write_search_result:
    Append the result of a search (i.e. what best dips did it find?
    What are their parameters?) to csv files (if it's realsearch, writes nans
    in appropriate places).

summarize_injrecov_result:
    Summarizes csv files into a text file

completeness_top1_plots:
    Turns csv files into pdf plots (stored in ../results/injrecovresult/plots)

'''
import pandas as pd, numpy as np, os
import time, logging
from datetime import datetime
import matplotlib.pyplot as plt
import pdb
import subprocess
plt.style.use('utils/lgb.mplstyle')

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

    wrnout, wrnerr = run_script('cat LOGS/*'+substr+'* | grep WRN')
    errout, errerr = run_script('cat LOGS/*'+substr+'* | grep ERR')
    excout, excerr = run_script('cat LOGS/*'+substr+'* | grep EXC')
    wrnerrexc = []
    for out in [wrnout, errout, excout]:
        lines = out.decode('utf-8').split('\n')[:-1]
        wrnerrexc.append('\n')
        for l in lines:
            wrnerrexc.append(l)

    writestr = ''
    now = time.strftime('%c')
    writestr = writestr + now + '\n'
    for outstr in outstrs:
        writestr=writestr+outstr+'\n'
    writestr = writestr + '\n{:d} warnings, {:d} errs, {:d} exceptn:\n'.format(
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

    csvdir = '../results/injrecovresult/' if inj else '../results/real_search/'

    for ap in ['sap']:
        csv1name = 'irresult_'+ap+'_top1.csv'
        csv2name = 'irresult_'+ap+'_allN.csv'

        # Get minimum time for epoch zero-point.
        lc = allq['dipfind']['tfe'][ap]
        min_time = np.min(lc['times'])
        fluxs = lc['fluxs']
        meanflux = np.mean(fluxs)
        rms_biased = float(np.sqrt(np.sum((fluxs-meanflux)**2) / len(fluxs)))

        # Recover best period, and corresponding BLS depth.
        pgdc = allq['dipfind']['bls'][ap]['coarsebls']
        pgdf = allq['dipfind']['bls'][ap]['finebls']
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

        for ix, ffoldperiod in enumerate(fnbestperiods):

            cfoldperiod = cnbestperiods[ix]
            fdepth = pgdf[cfoldperiod]['serialdict']['blsresult']['transdepth']
            fbls = allq['dipfind']['bls'][ap]['finebls'][cfoldperiod]
            φ_0 = fbls['φ_0']

            t0_rec = min_time + φ_0*ffoldperiod
            P_rec = ffoldperiod

            # If the recovered period is within +/- 0.1 days of the injected
            # period, and (recovered epoch modulo injected period) is within 
            # +/-5% of of (injected epoch modulo recovered period).
            if inj:
                atol = 0.1
                rtol = 0.05
                reldiff = abs((t0_rec % P_rec) - (t0_inj % P_inj)) / (t0_inj % P_inj)
                if (abs(P_inj - P_rec) < atol) and (reldiff < rtol):
                    foundinj = True
                else:
                    foundinj = False
            else:
                foundinj = np.nan

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
                    'depth_rec':fdepth
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

            LOGINFO('Wrote KIC-{:d} result to {:s} ({:s})'.format(
                kicid,csvdir,ap))


#########
# PLOTS #
#########

def completeness_top1_plots():
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

    # SNR vs P_CBP
    f, ax = plt.subplots()

    ax.scatter(df['P_inj'][df['foundinj']==True],
               df['depth_inj'][df['foundinj']==True]/df['rms_biased'][df['foundinj']==True],
               c='green', lw=0, alpha=0.9)
    ax.scatter(df['P_inj'][df['foundinj']==False],
               df['depth_inj'][df['foundinj']==False]/df['rms_biased'][df['foundinj']==False],
               c='red', lw=0, alpha=0.9)

    ax.set(xlabel='$P_\mathrm{CBP}\ [\mathrm{days}]$',
           ylabel='$\delta/\mathrm{RMS\ (per\ transit)}$',
           xscale='log',
           yscale='log')

    f.tight_layout()
    f.savefig('../results/injrecovresult/plots/completeness_top1_SNR_vs_periodcbp.pdf',
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
    completeness_top1_plots()
