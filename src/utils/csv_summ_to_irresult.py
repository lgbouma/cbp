'''
after injection recovery, to turn the injrecov_summ result csvs into a big
summary csv file, run this script.
'''
import pandas as pd
import numpy as np
import os

readdir = '../../results/injrecov_summ/'
fs = np.sort([f for f in os.listdir(readdir) if f.endswith('.csv')])

for ix, f in enumerate(fs):
    if ix % 1000 == 0:
        print(ix)
    if ix == 0:
        allN = pd.read_csv(readdir+f)
        continue
    allN = pd.concat([allN, pd.read_csv(readdir+f)])

allN = allN.drop('SNR_inj_pf') # code error in july run; fixable in post
top1 = allN[allN.index == 0]

allN.to_csv('../../results/real_search/irresult_sap_allN.csv', index=False)
top1.to_csv('../../results/real_search/irresult_sap_top1.csv', index=False)
