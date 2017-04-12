'''
Plotting errors in the main run led to a few things being flagged as pf-SNR>3
objects, but not getting dipsearchplots.

Once manually identified (and labelled "wtf") during
`/src/utils/look_at_results.py`, this script tries to make their dipsearchplots
again.

From /src/:
>>> python -m utils.wtf_ids_to_retry_dipsearchplot.py
'''

import inj_recov as ir
import inj_recov_plots as irp
import numpy as np
import os

# paths are relative to launch dir, which is src
wtf_ids = list(map(str,list(map(int,
    np.loadtxt('../results/real_search/wtf_ids.txt')))))
# apparently we stored these as strings...

for kicid in wtf_ids:
    print('{:s}'.format(kicid))
    stage = 'realsearch_real' # really really real
    predir = 'real'
    inj = False

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

    doneplots = os.listdir('../results/dipsearchplot/'+predir)
    plotmatches = [f for f in doneplots if f.startswith(kicid) and
            stage in f]
    if len(plotmatches)>0:
        print('\nFound dipsearchplot, continuing.\n')
    elif 'realsearch' in stage:
        irp.dipsearchplot(lcd, allq, ap='sap', stage=stage, inj=inj)

print('done!')
