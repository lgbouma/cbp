'''
A script for the real search that finds whichever Kepler IDs have not been
searched, and then writes them to a text file.

Usage:
>>> open the script, select your writeout file name
>>> (bash): python make_remaining_list.py
'''
import numpy as np
import os

dones = [int(f.split('_')[0]) for f in
        os.listdir('/media/luke/LGB_tess_data/170705_realsearch/') if
        f.endswith('allq_realsearch_real.p') ]

want = np.genfromtxt('../data/morph_gt_0.6_OR_per_lt_3_ids.txt', dtype=int)

mask = np.in1d(want, dones)

done = want[mask]
print('len done: {:d}'.format(len(done)))

todo = want[~mask]
print('len todo: {:d}'.format(len(todo)))

np.savetxt('../data/170710_11am_notdone.txt', todo, fmt='%d')
