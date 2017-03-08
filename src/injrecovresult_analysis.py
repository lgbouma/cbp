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

if __name__ == '__main__':
    summarize_injrecov_result()
