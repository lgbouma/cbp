import pandas as pd, numpy as np, os
import time

def summarize_injrecov_result():
    '''
    Write a summary text file to ../results/injrecovresult/summary.txt
    '''
    csvdir = '../results/injrecovresult/'
    csvnames = os.listdir(csvdir)
    summpath = '../results/injrecovresult/summary.txt'
    f = open(summpath, 'w')

    # First Q: what % do we recov in top 1?
    top1s = np.sort([csvdir+n for n in csvnames if 'top1' in n])

    outstrs = []
    for top1 in top1s:
        #(iterate over apertures)
        df = pd.read_csv(top1)

        findrate = len(df[df['foundinj']==True])/len(df)

        outstr = '{:s}, find rate: {:.3g}%, N={:d}'.format(
            top1, findrate*100., len(df))
        print(outstr)
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
        print(outstr)
        outstrs.append(outstr)

    writestr = ''
    now = time.strftime('%c')
    writestr = writestr + now + '\n'
    for outstr in outstrs:
        writestr=writestr+outstr+'\n'

    f.write(writestr)
    f.close()


def write_injrecov_result(lcd, allq, stage=None):
    '''
    Append the result of this injection-recovery experiment (i.e. whether it
    was successful, what the basic parameters of the EB and the injected system
    were, what the basic parameters of the LC were) to csv files.

    A system is discovered if its "fine" (rather than coarse) best period and
    best transit epoch agree with the injected ones to some precision.

    There are two files:
    csv1: rows are results with only best periodogram period for each system.
    csv2: rows are results with all 5 "best guesses" at period for each system.

    These csv files are aperture specific.
    '''

    kicid = lcd[list(lcd.keys())[0]]['objectinfo']['keplerid']
    kebc_period = float(lcd[list(lcd.keys())[0]]['kebwg_info']['period'])
    morph = float(lcd[list(lcd.keys())[0]]['kebwg_info']['morph'])
    im = allq['inj_model']
    P_inj = im['params'].per
    t0_inj = im['params'].t0
    δ = (im['params'].rp)**2

    csvdir = '../results/injrecovresult/'

    for ap in ['sap','pdc']:
        csv1name = 'irresult_'+ap+'_top1.csv'
        csv2name = 'irresult_'+ap+'_allN.csv'

        # Get minimum time for epoch zero-point
        lc = allq['dipfind']['tfe'][ap]
        min_time = np.min(lc['times'])
        fluxs = lc['fluxs']
        meanflux = np.mean(fluxs)
        rms_biased = float(np.sqrt(np.sum((fluxs-meanflux)**2) / len(fluxs)))

        # Recover best period
        pgdc = allq['dipfind']['bls'][ap]['coarsebls']
        pgdf = allq['dipfind']['bls'][ap]['finebls']
        cnbestperiods = np.sort(pgdc['nbestperiods'])
        cbestperiod = pgdc['bestperiod']
        fnbestperiods = np.sort([pgdf[cnbp]['serialdict']['bestperiod']
                for cnbp in cnbestperiods])
        fbestperiod = pgdf[cbestperiod]['serialdict']['bestperiod']
        bestperiod = fbestperiod

        for ix, ffoldperiod in enumerate(fnbestperiods):

            cfoldperiod = cnbestperiods[ix]
            fbls = allq['dipfind']['bls'][ap]['finebls'][cfoldperiod]
            φ_0 = fbls['φ_0']

            t0_rec = min_time + φ_0*ffoldperiod
            P_rec = ffoldperiod

            # If the recovered epoch and period are within +/- 0.1 days of the
            # injected epoch and period, we "recovered" the injected signal.
            tol = 0.1
            if (abs(P_inj - P_rec) < tol) and (abs(t0_inj - t0_rec) < tol):
                foundinj = True
            else:
                foundinj = False

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
                    'depth':δ
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

