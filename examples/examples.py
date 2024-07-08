import sys
from tqdm import tqdm
import inspect
import os
import numpy as np
import wget
import pandas as pd

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

import astropy

import miletos
import tdpy
from tdpy.util import summgene
import ephesos
import nicomedia

import assos


def cnfg_ggal():
    
    #path = os.environ['DATA'] + '/general/TOI1233/HD108236_PFS_20220627.vels'
    #print('Reading from %s...' % path)
    
    dictchalinpt = dict()
    dictchalinpt['sherextr'] = 0.3
    dictchalinpt['sangextr'] = 0.3
    
    dictchalinpt['xposhost'] = np.array([0.])
    dictchalinpt['yposhost'] = np.array([0.])
    dictchalinpt['ellphost'] = np.array([0.1])
    dictchalinpt['beinhost'] = np.array([1.])
    
    dictchalinpt['xpossubh'] = np.array([1.])
    dictchalinpt['ypossubh'] = np.array([1.])
    dictchalinpt['defssubh'] = np.array([1.])
    dictchalinpt['ascasubh'] = np.array([1.])
    dictchalinpt['acutsubh'] = np.array([1.])

    dictchaloutp = assos.eval_emislens( \
                                           dictchalinpt=dictchalinpt, \
                                          )
    

globals().get(sys.argv[1])(*sys.argv[2:])

