'''
A script that, when launched, iteratively prompts user for classification
based on the results of real_search ("CR":=cmd line return).

Each input is then appended to a readout file.

(bash)
>>> python look_at_results.py
Run from /src/utils, and that must be the only open window with "utils" in its
name.

Requires installation of `wmctrl` (a program that enables interaction with X
Window Manager).
'''

import pandas as pd, numpy as np
import os
import subprocess
import time

candlist = '../../results/real_search/candidates_sort.csv'
df = pd.read_csv(candlist)

ids_labels = []

# Create list data file, otherwise append.
writepath = '../../results/real_search/human_labelled_candidates.txt'
if os.path.exists(writepath):
    delete = input('delete existing labelled candidates? [y/n]')
    if delete == 'y':
        os.remove(writepath)
    if delete == 'n':
        print('OK, appending')
writefile = open(writepath, 'w+')
writefile.close()

for kicid in np.array(df['kicid']):

    pathdir = '../../results/dipsearchplot/real/'
    fname = str(kicid)+'_saprealsearch_real.png'
    fpath = pathdir+fname

    eogcall = 'eog --fullscreen '+fpath+' &'
    p = subprocess.Popen(eogcall.split(), stdout=subprocess.PIPE)

    # Assume: shell is only open window with "utils" in its name. Takes ~2 secs
    # for the fullscreen image load.
    time.sleep(2)
    subprocess.Popen('wmctrl -a utils'.split(), stdout=subprocess.PIPE)

    human_label = input('dip [j+CR], maybe dip [k+CR], noise [CR], wtf [w+CR] ')

    p.terminate()

    if human_label == '':
        human_label = 'noise'
    elif human_label == 'j':
        human_label = 'dip'
    elif human_label == 'k':
        human_label = 'maybe_dip'
    elif human_label == 'w':
        human_label = 'wtf'

    with open(writepath, 'a') as writefile:
        writefile.write('{:d}, {:s}\n'.format(int(kicid), human_label))

    ids_labels.append([(kicid, human_label)])


print(ids_labels)

