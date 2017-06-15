'''
Make a list (to be used during slurm script submission), via:
>>> python -m utils.make_N_to_KICID.py

range(0,20320)      kicid
------              -------
0                   id0
1                   id0
2                   id0
...                 ...
20                  id1
21                  id1
...                 ...
'''

import inj_recov as ir
import numpy as np

kebc = ir.get_kepler_ebs_info()
kebc = kebc[(kebc['morph']>0.6) | (kebc['period']<3)]
kic_ids = np.array(kebc['KIC'])
N_injrecovs_per_LC = 20

N_and_KICID = []
N, kicid_ind = 0, 0
while N < (len(kic_ids)*N_injrecovs_per_LC):
    N_and_KICID.append((N, kic_ids[kicid_ind]))
    if (N+1) % N_injrecovs_per_LC == 0:
        kicid_ind += 1
    N += 1

# make the first few a case with only 2 quarters of data for early error raise
N_extra = 10
header = np.array([range(N_extra), np.ones_like(range(N_extra))*4936680]).T

N_and_KICID = np.concatenate([header, np.array(N_and_KICID)])

N_and_KICID[N_extra:, 0] += N_extra

np.savetxt('../data/N_to_KICID.txt', N_and_KICID, fmt='%d')
