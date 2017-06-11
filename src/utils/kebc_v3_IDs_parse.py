# run directly from cmd line

from astropy.io import ascii
import numpy as np

def get_kepler_ebs_info():
    '''
    Read in the nicely formatted astropy table of the Kepler Eclipsing
    Binary Catalog information. (Includes morphology parameter, kepler
    magnitudes, whether short cadence data exists, and periods).
    See Prsa et al. (2011) and the subsequent papers in the KEBC series.
    '''

    #get Kepler EB data (e.g., the period)
    keb_path = '../../data/kebc_v3_170611.csv'
    cols = 'KIC,period,period_err,bjd0,bjd0_err,pdepth,sdepth,pwidth,swidth,'+\
           'sep,morph,GLon,GLat,kmag,Teff,SC,'
    cols = tuple(cols.split(','))
    tab = ascii.read(keb_path,comment='#')
    currentcols = tab.colnames
    for ix, col in enumerate(cols):
        tab.rename_column(currentcols[ix], col)

    tab.remove_column('') # now table has correct features

    return tab

if __name__ == '__main__':
    tab = get_kepler_ebs_info()
    sel = tab[(tab['period'] < 3) | (tab['morph'] > 0.6)]
    sel_KIC_ID, indices = np.unique(np.array(sel['KIC']), return_index=True)
    oldsel = sel
    sel = sel[indices]

    # write to numbers.tex
    N_kep_EBs_tot = len(tab)
    N_sel_EBs = len(sel_KIC_ID)
    N_sel_morph_gt_pt6 = len(sel[sel['morph']>0.6])
    N_sel_per_lt_3 = len(sel[sel['period']<3])

    f = open('../../paper/numbers.tex', 'a+')
    f.write(r'%following commands written by utils/kebc_v3_IDs_parse.py')
    f.write('\n')
    f.write(r'\newcommand{\numkepebstot}{'+'{:d}'.format(N_kep_EBs_tot)+r'}')
    f.write('\n')
    f.write(r'\newcommand{\numselectedebs}{'+'{:d}'.format(N_sel_EBs)+r'}')
    f.write('\n')
    f.write(r'\newcommand{\numselectedmorphgtptsix}{'+'{:d}'
            .format(N_sel_morph_gt_pt6)+r'}')
    f.write('\n')
    f.write(r'\newcommand{\numselectedperltthree}{'+'{:d}'
            .format(N_sel_per_lt_3)+r'}')
    f.write('\n')
    f.close()

    # write KIC IDs to use for MAST download
    np.savetxt(
        '../../data/morph_gt_0.6_OR_per_lt_3_ids.txt',
        sel_KIC_ID,
        fmt='%d',
        header='written by utils/kebc_v3_IDs_parse.py before MAST upload')

    # notice that there are duplicates
    import pandas as pd
    df = pd.DataFrame({'kic':np.array(oldsel['KIC'])})
    print('found duplicates:')
    print(pd.concat(g for _,g in df.groupby('kic') if len(g) > 1))
