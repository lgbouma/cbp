'''
$ cd $CBPDIR/src/utils
$ python human_labelled_to_symlinks.py

Turn `../../results/real_search/human_labelled_candidates.txt` to symlinks
of images to look through (for both "dips" and "maybes").

This also makes a directory: /results/real_search/dip_deepdive, with symlinks
to everything, and text files for everything.
'''

import pandas as pd, numpy as np, os
import pickle

def make_labelled_symlinks():

    cand_path = '../../results/real_search/human_labelled_candidates.txt'
    df = pd.read_csv(cand_path, names=['kicid','human_label'], delimiter=', ',
            engine='python')

    assert set(np.unique(df['human_label'])) == {'noise','dip','maybe_dip'}, \
            'human_label should only be noise, dip, or maybe_dip'

    # Make symlinks for "dips", then for 'maybes"
    for id_type in ['dip', 'maybe_dip']:
        dipids = np.array(df[df['human_label']==id_type]['kicid'])

        for dipid in dipids:
            srcpath = '/home/luke/Dropbox/proj/cbp/results/dipsearchplot/real/'+\
                    str(dipid)+'_saprealsearch_real.png'
            substr = 'dip' if id_type=='dip' else 'maybe'
            dstpath = '/home/luke/Dropbox/proj/cbp/results/real_search/'+\
                    'labelled_{:s}_symlinks/'.format(substr)+str(dipid)+'.png'
            if not os.path.exists(dstpath):
                try:
                    os.symlink(srcpath, dstpath)
                except:
                    print('symlink failed for {:s}'.format(str(dipid)))

        print('\nSymlinked human-labelled {:s}!'.format(id_type))

    # Make a list of wtfs
    wtfids = np.array(df[df['human_label']=='wtf']['kicid'])
    np.savetxt('../../results/real_search/wtf_ids.txt', wtfids, fmt='%d')


def make_deepdive_dir():

    cand_path = '../../results/real_search/human_labelled_candidates.txt'
    df = pd.read_csv(cand_path, names=['kicid','human_label'], delimiter=', ',
            engine='python')

    # Make symlinks for "dips", both dipsearchplot and whitened diagnostic.
    dipids = np.array(df[df['human_label']=='dip']['kicid'])

    for dipid in dipids:
        srcpath = '/home/luke/Dropbox/proj/cbp/results/dipsearchplot/real/'+\
                str(dipid)+'_saprealsearch_real.png'
        dstpath = '/home/luke/Dropbox/proj/cbp/results/real_search/dip_deepdive/'+\
                'dipsearchplot_symlinks/'+str(dipid)+'.png'
        if not os.path.exists(dstpath):
            try:
                os.symlink(srcpath, dstpath)
            except:
                print('symlink failed for {:s}'.format(str(dipid)))
    print('\nSymlinked dipsearchplots for human-labelled \'dips\'!')

    for dipid in dipids:
        srcpath = '/home/luke/Dropbox/proj/cbp/results/whitened_diagnostic/real/'+\
                str(dipid)+'_saprealsearch_real.png'
        dstpath = '/home/luke/Dropbox/proj/cbp/results/real_search/dip_deepdive/'+\
                'whiteneddiagnostic_symlinks/'+str(dipid)+'.png'
        if not os.path.exists(dstpath):
            try:
                os.symlink(srcpath, dstpath)
            except:
                print('symlink failed for {:s}'.format(str(dipid)))
    print('\nSymlinked whitened_diagnostic for human-labelled \'dips\'!')

    # Make files to read available info.
    madefiles = False
    for dipid in dipids:
        txtpath = '/home/luke/Dropbox/proj/cbp/results/real_search/dip_deepdive/'+str(dipid)+'.txt'
        if not os.path.exists(txtpath):
            f = open(txtpath, 'a')
            f.close()
            madefiles = True

    if madefiles:
        print('\nMade text files for info-dumping.')


if __name__ == '__main__':
    make_labelled_symlinks()
    make_deepdive_dir()
