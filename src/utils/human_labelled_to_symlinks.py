import pandas as pd, numpy as np, os
'''
Turn `../../results/real_search/human_labelled_candidates.txt` to symlinks
of images to look through (for both "dips" and "maybes").

The symlinks are in ../../results/real_search/labelled_dip_symlinks and
../../results/real_search/labelled_maybe_symlinks
'''

def make_symlinks():

    cand_path = '../../results/real_search/human_labelled_candidates.txt'
    df = pd.read_csv(cand_path, names=['kicid','human_label'], delimiter=', ',
            engine='python')

    # Make symlinks for "dips", then for 'maybes"
    for id_type in ['dip', 'maybe_dip']:
        dipids = np.array(df[df['human_label']==id_type]['kicid'])

        for dipid in dipids:
            srcpath = '/home/luke/Dropbox/proj/cbp/results/dipsearchplot/real/'+\
                    str(dipid)+'_saprealsearch_real.png'
            substr = 'dip' if id_type=='dip' else 'maybe'
            dstpath = '/home/luke/Dropbox/proj/cbp/results/real_search/'+\
                    'labelled_{:s}_symlinks/'.format(substr)+str(dipid)+'.png'
            try:
                os.symlink(srcpath, dstpath)
            except:
                print('symlink failed for {:s}'.format(str(dipid)))

        print('\nSymlinked human-labelled {:s}!'.format(id_type))


if __name__ == '__main__':
    make_symlinks()
