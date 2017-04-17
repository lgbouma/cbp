'''
Cross match anything appearing in Table 3 of Uytterhoeven et al (2011) with
KEBC v3.

>>> cd $CBP_DIR/src/analysis_scripts
>>> python uytterhoeven_2011_table3_crossmatch_with_KEBC.py
'''
import os
import pandas as pd, numpy as np
from astropy.table import Table
from astropy.io import ascii
import astropy.units as u
import pdb

tab3 = Table.read('../../data/Uytterhoeven_2011_Table3.vot', format='votable')
tab3 = tab3.to_pandas()
tab3_kic_ids = tab3['KIC']

def get_kepler_ebs_info():
    keb_path = '../../data/kepler_eb_catalog_v3.csv'
    cols = 'KIC,period,period_err,bjd0,bjd0_err,morph,GLon,GLat,kmag,Teff,SC'
    cols = tuple(cols.split(','))
    tab = ascii.read(keb_path,comment='#')
    currentcols = tab.colnames
    for ix, col in enumerate(cols):
        tab.rename_column(currentcols[ix], col)
    tab.remove_column('col12') # now table has correct features

    return tab

kebc_v3 = get_kepler_ebs_info()
kebc_v3_highmorph = kebc_v3[kebc_v3['morph']>0.6]
v3_kic_ids = np.array(kebc_v3['KIC'])
v3_highmorph_ids = np.array(kebc_v3_highmorph['KIC'])

N_int = len(np.intersect1d(tab3_kic_ids, v3_kic_ids))
N_highmorph_int = len(np.intersect1d(tab3_kic_ids, v3_highmorph_ids))

print('N in Uytterhoeven 2011 Tab3 and KEBC v3: {:d}'.format(N_int))
print('N in Uytterhoeven 2011 Tab3 and KEBC v3 (morph>0.6): {:d}'.format(N_highmorph_int))
