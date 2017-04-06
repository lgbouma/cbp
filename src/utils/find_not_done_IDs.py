'''
Write a list of which KIC IDs are left to process in realsearch.
'''

import os
import numpy as np

ids = []
with open('../../data/morph_gt_0.6_ids.txt') as f:
    ids.append(f.readlines())
ids = list(map(int, [i.split('\n')[0] for i in ids[0]]))

done_ids = os.listdir('../../data/injrecov_pkl/real/')
_ = [fn for fn in done_ids if 'allq' not in fn]
done_ids  = list(map(int, [f.split('_')[0] for f in _]))

not_done_ids = list(set(ids).symmetric_difference(done_ids))

np.savetxt('../../data/not_done_ids.txt', np.array(not_done_ids), fmt='%d')
