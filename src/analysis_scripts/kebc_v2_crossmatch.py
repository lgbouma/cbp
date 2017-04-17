'''
Cross match things labelled "dips" in real_search to things in KEBC, v2.

>>> cd $CBP_DIR/src/analysis_scripts
>>> python kebc_v2_crossmatch.py
'''
import os
import pandas as pd, numpy as np

deepdivedir = '../../results/real_search/dip_deepdive'
cand_ids = [f.split('.')[0] for f in os.listdir(deepdivedir) if
        f.endswith('.txt')]

kebc_v2 = pd.read_csv('../../data/kepler_eb_catalog_v2.csv')

for cand_id in cand_ids:
    match = kebc_v2[kebc_v2['KIC'] == int(cand_id)]
    if len(match) == 1:
        got_match = True
    else:
        got_match = False
    assert len(match) <= 1, 'should have at most one crossmatch'

    msg = ''
    if not got_match:
        msg = '* KEBC_v2: no match.'
    if got_match:
        msg = '* KEBC_v2: match. Info:\n'
        for k in np.sort(list(match.keys())):
            if k not in ['KIC','BJD_0']:
                msg += '\t{:s}: {:s}\n'.format(k, str(np.array(match[k])[0]))


    writefile = deepdivedir+'/'+cand_id+'.txt'
    f = open(writefile, 'a')
    f.write(msg)
    f.close()

    print(msg)
    print('wrote to {:s}'.format(writefile))

