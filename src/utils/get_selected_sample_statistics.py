'''
Get statistics (e.g., numbers of different subsets of the catalog) for the
selected sample of the Kepler EB Catalog.

From /src/utils:
>>> python get_selected_sample_statistics.py
'''

from __future__ import print_function, division
import numpy as np
import pandas as pd
from astropy.io import ascii

##########
# BASICS #
##########
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
print('N in KEBC_v3, retrieved 170204: {:d}'.format(
    len(kebc_v3)))

kebc_v2 = pd.read_csv('../../data/kepler_eb_catalog_v2.csv')
print('N in KEBC_v2, retrieved 170412: {:d}'.format(
    len(kebc_v2)))

df_v3 = kebc_v3[kebc_v3['morph']>0.6]
print('N in KEBC_v3, morph>0.6: {:d}'.format(
    len(df_v3)))

print('KEBC_v2 does not have morph parameter.\n')

##############
# CROSSMATCH #
##############
highmorph_ids_v3 = np.array(df_v3['KIC'])
kic_ids_v2 = np.array(kebc_v2['KIC'])
# Make sure these .00, .01, .02 things can be ignored.
v2_dot_ids = np.unique(list(map(int, kic_ids_v2[kic_ids_v2 % 1 != 0])))
v2_int_v3_dot_ids = np.intersect1d(v2_dot_ids, highmorph_ids_v3)
assert v2_int_v3_dot_ids.size == 0, 'the .00, .01, .02 stuff is not safe to'+\
    ' ignore! they show up in morph>0.6 sample!'

# 1. What fraction of v3 morph>0.6 were in v2?
v2_int_v3_highmorph_ids = np.intersect1d(kic_ids_v2, highmorph_ids_v3)
print('N in KEBC_v3, morph>0.6 that have crossmatches in KEBC_v2: {:d}'.format(
    len(v2_int_v3_highmorph_ids)))
print('\twhich is {:.3g}%'.format(float(
    len(v2_int_v3_highmorph_ids)/len(df_v3)*100)))

# 2. Of the v3 morph>0.6 entries that also had entries in v2, what is the
# distribution of TYPE (detached, semidetached, etc)?
is_in_v3_highmorph = []
for v2_id in kic_ids_v2:
    val = True if v2_id in highmorph_ids_v3 else False
    is_in_v3_highmorph.append(val)

kebc_v2['is_in_v3'] = is_in_v3_highmorph
highmorph_v2 = kebc_v2[kebc_v2['is_in_v3']==True]
print('Of the {:d},'.format(len(highmorph_v2)))
type_frac = {}
for TYPE in np.sort(np.unique(highmorph_v2['TYPE'])):
    print('\t{:s}:\t{:d}\t({:.3g}%)'.format(
        str(TYPE),
        len(highmorph_v2[highmorph_v2['TYPE']==TYPE]),
        len(highmorph_v2[highmorph_v2['TYPE']==TYPE])/len(highmorph_v2)*100,
        ))
    type_frac[TYPE] = \
        len(highmorph_v2[highmorph_v2['TYPE']==TYPE])/len(highmorph_v2)

# 3. What number of overcontact binaries is in our v3 sample if the same rates
# apply? (Unclear if should be expected, but this is a 0th order guess).

print('\nAssuming the same rates apply to the v3 catalog, we get' )
print('Of the {:d},'.format(len(highmorph_ids_v3)))
for TYPE in np.sort(list(type_frac.keys())):
    print('\t{:s}:\t{:d}\t({:.3g}%)'.format(
        str(TYPE),
        int(type_frac[TYPE]*len(highmorph_ids_v3)),
        type_frac[TYPE]*100))

# 4. What does the sini distribution of overcontact binaries from v2 look like?
import seaborn as sns
import matplotlib.pyplot as plt
plt.close('all')
ax = sns.distplot(kebc_v2[kebc_v2['TYPE']=='OC']['sini'], kde=False)
ax.set_title('KEBC, v2, things labelled \"overcontact\"\n(types and sini\'s were later retracted(?))')
writepath = '../../doc/170412_overcontact_sini_distribution.pdf'
plt.savefig(writepath)

print('\nWrote overcontact sini distribn to {:s}'.format(writepath))
