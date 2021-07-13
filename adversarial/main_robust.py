#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# MIT License
# Copyright (c) 2020 Christa Cuchiero, Wahid Khosrawi, Josef Teichmann

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""main_robust.py:
This file implements the robust/adversarial version of the neural lsv calibration
"""

from finModels.helpers_finModels import fin_option
from runfile_cali import main as main_caliLSV
from runfile_sabr_cali import main as main_sabr_cali
from runfile_localVol import main as main_locvol
import datetime
from time import time
import pickle
import tensorflow as tf
import scipy as sp
import numpy as np
import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def serve_fin_option(S0):
    maturities = [.15, 0.25, 0.5, 1.]
    # STRIKES ARE LISTS
    strikes = []
    strikes.append(np.linspace(np.exp(-.1) * S0, np.exp(0.1) * S0, 20))
    strikes.append(np.linspace(np.exp(-.2) * S0, np.exp(0.2) * S0, 20))
    strikes.append(np.linspace(np.exp(-.3) * S0, np.exp(0.3) * S0, 20))
    strikes.append(np.linspace(np.exp(-.5) * S0, np.exp(0.5) * S0, 20))

    return (fin_option(maturities, strikes))


def one_sim(version, option, para_locvol, para_MC):
    # Each data requires the parameters below to be randomly drawn.
    # order of parameters:
    
    lv_sig = []
  
    unif_factor = 0.005

    # We draw 4 sets of ground truth assumption parameters
    for i in range(4):
        lv_sig.append( {
            "p0": 0.473468 + 0.5*np.random.uniform(-unif_factor,unif_factor) ,
            "p1": 0.677544 + 0.5*np.random.uniform(-unif_factor,unif_factor),
            "sig0": 1.601103 + 2*np.random.uniform(-unif_factor,unif_factor),
            "sig1": 0.349448 + 2*np.random.uniform(-unif_factor,unif_factor),
            "sig2": 0.561196 + 2*np.random.uniform(-unif_factor,unif_factor),
            "gamma1": 1.1,
            "gamma2": 20.0,
            "beta1": 0.005,
            "beta2": 0.001,
            "kappa": 0.5,
            "lam1": 10.0,
            "lam2": 10.0,
            "eps_t": 0.1,
            "const_else_factor": 0.4
        })
    
    para_locvol.update({
        "alpha_loc_vol_0": 0.5,
        "version": version,
        "lv_sig": lv_sig
    } )

    

    # If already stored, we can comment out the pM computation part
    finsurf = main_locvol(para_locvol, option, para_MC)

    return(finsurf)


def cali_sabr(finsurf, para_MC):
    # This part returns the calibrated three sabr parameters
    main_sabr_cali(finsurf, para_MC)


if __name__ == "__main__":

    with open('log.txt', 'w') as file:
        file.writelines(['Log file for the statistical test:',
                         3*'\n'])

    para_locvol = {'S0': 100.}

    para_MC = {'delta_t': 0.01,
               'N_mc_data': 10**6,
               'N_mc_inner': 10**5,
               'N_mc_train': 400,          # How many trajectories per Iteration
               'N_mc_sabr_cali_iter': 2000,  # How many training steps
               'N_mc_lsv_cali_iter': 12000   # How many training steps
               }

    para_MC['N_mc_runs'] = int(
        np.rint(para_MC['N_mc_data']/para_MC['N_mc_inner']))

    option = serve_fin_option(para_locvol['S0'])

    # We only loop once, but this can be extended if one wants to repeat the adversarial
    # calibration multiple times 
    for i in range(0, 1):
        tstart = time()
        with open('log.txt', 'a') as file:
            file.writelines([20*'--',
                             '\nI start with step {}'.format(i), 3*'\n'])

    
        version = str(i).zfill(3)
        tf.reset_default_graph()

        finsurfL = one_sim(version, option, para_locvol, para_MC)
        with open('caliRes/'+version+'/'+'finsurf_locVol'+'.pkl', 'wb') as f:
            pickle.dump(finsurfL, f)

        with open('data/locVolPrices/finsurf_locVol_{}.pkl'.format(version), 'wb') as f:
            pickle.dump(finsurfL, f)

        tf.reset_default_graph()

        
        cali_sabr(finsurfL[0], para_MC)
        tf.reset_default_graph()

        
        # Now calibrate the SLV model using the previously calibrated sabr parameters
        # Have everything stored in finsurf at the end, including times etc. Using pickle
        # We can then store it
        finsurf = main_caliLSV(finsurfL, para_MC)

        # We could now store (pickle) finsurf to plot stuff later here, skip for now
        print('Step {:4} success:\nComputation time  {}'.format(
            i, str(datetime.timedelta(seconds=np.rint(time() - tstart)))))
        print(50*'=')

        with open('log.txt', 'a') as file:
            file.writelines(['\n',
                                'Step: {:4} success:\n'.format(i),
                                'Time for whole iteration: {}\n'.format(str(datetime.timedelta(seconds=np.rint(time() - tstart))))])


        with open('caliRes/'+version+'/'+'finsurf'+'.pkl', 'wb') as f:
            pickle.dump(finsurf, f)
    

        

        with open('log.txt', 'a') as file:
            file.writelines([20*'--', '\n', 20*'--'])
