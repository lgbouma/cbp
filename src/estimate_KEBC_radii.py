'''
How many Gaia DR1 stars overlap with the sample given by Armstrong+ 2013? (&
how many Armstrong+ 2013 stars overlap w/ KEBC v3?)

Rephrased: how many stars can we get ~ok radii estimates for?

Usage:
>>> python estimate_KEBC_radii.py
'''
import os
import pandas as pd, numpy as np
from astropy.table import Table
import astropy.units as u
import pdb

# cf. Sec 5.2.2 of Armstrong+ 2013: his method did pretty well at binary solid
# angles, but not individual radii.
data_dir = '../data/'
tab = Table.read(data_dir+'Armstrong_2013_KEBC_temperatures.vot',
        format='votable')
adf = tab.to_pandas()
adf['kepid'] = np.array(adf['KIC'], dtype=np.int64)
adf['bin_solid_angle'] = adf['R1_D']**2 * (1 + adf['R2_R1']**2)

# Find all KEBC v3 (Kirk+ 2016, 2875 entries, so a few less than most-modern)
# catalogs that match with TGAS.  Uses: DFM's gaia_kepler package, which runs
# off Jo Bovy's gaia_tools.
# The crossmatch is done by coordinate matching in gaia_tools.xmatch, which
# wraps around astropy.coordinates' SkyCoord.match_coordinates_sky module. That
# module uses kd-trees to do the matching quickly.
from gaia_kepler import data, tgas_match
kic = data.KICatalog().df
ebs = pd.merge(data.EBCatalog().df, kic[['kepid', 'ra', 'dec']], on='kepid')
matched = tgas_match(ebs)
N_Gaia_KEBCv3_xmatch = len(matched)

# Get the Armstrong+ 2013 / Gaia / KEBC_v3 crossmatches
allmatch = pd.merge(matched, adf[['kepid', 'bin_solid_angle']], on='kepid')
N_allmatch = len(allmatch)

print('{:d} Gaia TGAS to KEBCv3 xmatches'.format(N_Gaia_KEBCv3_xmatch))
print('{:d} Gaia TGAS to KEBCv3 to Armstrong 2013 xmatches'.format(N_allmatch))

# Use distances reported by Bailer-Jones in 2016ApJ...833..119A
# Naive "1/parallax" gets things wrong in a biased way.
tgas_distances = data.TGASDistancesCatalog().df
allmatch = pd.merge(allmatch, tgas_distances, left_on='tgas_source_id',
        right_on='SourceId')

# Take distances to be the 50th percentile (i.e. the median) of the posterior,
# using "Milky Way" prior.
# (http://www2.mpia-hd.mpg.de/homes/calj/tgas_distances/README.txt)
dist = np.array(allmatch['r50MW'])*u.pc
bin_solid_angle = np.array(allmatch['bin_solid_angle'])

effective_radius = np.sqrt(dist**2 * bin_solid_angle)

allmatch['R_eff'] = effective_radius.to(u.Rsun)

# See which of these are overcontact.
kebc_v2 = pd.read_csv('../data/kepler_eb_catalog_v2.csv')
kebc_v2['kepid'] = np.array(kebc_v2['KIC'], dtype=np.int64)

allmatch = pd.merge(allmatch, kebc_v2[['kepid','TYPE','sini','P_0']], on='kepid')

overcontact = allmatch[allmatch['TYPE']=='OC']

#In [4]: overcontact[['R_eff','sini','P_0']].describe()
#Out[4]: 
#               R_eff       sini        P_0
#      count  47.000000  47.000000  47.000000
#      mean    3.683552   0.631008   0.816799
#      std     3.793451   0.238093   0.584381
#      min     1.231325   0.322070   0.133474
#      25%     2.212401   0.399155   0.437094
#      50%     2.892126   0.578490   0.731513
#      75%     3.847760   0.857260   1.030456
#      max    25.306283   0.997540   3.397683
import matplotlib.pyplot as plt
f, ax = plt.subplots()

ax.scatter(overcontact['P_0'],overcontact['R_eff'],c='k')

ax.set_xlabel('period [day]',fontsize=8)
ax.set_ylabel('R_eff [R_sun]', fontsize=8)
ax.set_title('R_eff from solid angles reported in Armstrong+ 2013,'+\
             '\ndistances from TGAS, labelled OC by KEBC_v2', fontsize=8)
f.tight_layout()
f.savefig('../results/stellar_properties/period_vs_radius_xmatch.pdf')
ax.set(xlim=[0,2],ylim=[0,6])
f.tight_layout()
f.savefig('../results/stellar_properties/period_vs_radius_xmatch_smaller_lims.pdf')


