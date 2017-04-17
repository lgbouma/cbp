'''
Cross match anything appearing in Table 3 of Uytterhoeven et al (2011) with the
candidates. (Of more general interest: what if we did this for the entire
KEBC?)

>>> cd $CBP_DIR/src/analysis_scripts
>>> python uytterhoeven_2011_table3_crossmatch.py
'''
import os
import pandas as pd, numpy as np
from astropy.table import Table
import astropy.units as u
import pdb

deepdivedir = '../../results/real_search/dip_deepdive'
cand_ids = [f.split('.')[0] for f in os.listdir(deepdivedir) if
        f.endswith('.txt')]

tab3 = Table.read('../../data/Uytterhoeven_2011_Table3.vot', format='votable')
df = tab3.to_pandas()

for cand_id in cand_ids:
    match = df[df['KIC'] == np.int32(cand_id)]
    if len(match) == 1:
        got_match = True
    else:
        got_match = False
    assert len(match) <= 1, 'should have at most one crossmatch'

    msg = ''
    if not got_match:
        msg = '* Uytterhoeven11 table3: no match.'
    if got_match:
        classificaton = np.array(match['Class'])[0]
        freq = np.array(match['f'])[0]
        period = (1/(freq/u.day)).to(u.hr).value
        msg = '* Uytterhoeven11 table3: match. Info:\n'
        msg += '\tclass: {:s}\n\tfreq (1/day): {:s}\n\tperiod (hr): {:s}'.format(
                str(classificaton),
                str(freq),
                str(period)
                )

    writefile = deepdivedir+'/'+cand_id+'.txt'
    f = open(writefile, 'a')
    f.write(msg)
    f.close()

    print(msg)
    print('wrote to {:s}'.format(writefile))

