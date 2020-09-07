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

"""stat_test.py:
This file implements the statistical test as specified in the paper. The main 
settings are defined here and can be modified as needed, e.g.
  - the gpu to use 
  - whether data is supposed to be generated or loaded from precomputed samples
  - the range of data that is supposed to be calibrated
  - ...
"""

# OS and utility imports
import os, sys
import pickle
from time import time
import datetime

# This is so that the gpu id is the same as the nvidia-smi ones 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# specify which gpu should be used.
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# ML imports
import numpy as np
import tensorflow as tf

# Imports for the stat. test such as calibrations etc
from runfile_localVol import main as main_locvol
from runfile_sabr_cali import main as main_sabr_cali
from runfile_cali import main as main_caliLSV

# Utility for handling maturities strikes etc
from finModels.helpers_finModels import fin_option


# This defines the range of iV surfaces that will be calibrated 
VERSIONS = np.arange(1,201)


def serve_fin_option(S0):
    """Defines the maturities and strikes we consider for calibration

    Args:
        S0 (float): Current price of stock/ risky asset

    Returns:
        fin_option instance: stores strikes ans maturities defined inside the
        function
    """
    maturities = [.15, 0.25, 0.5 , 1.]
    # STRIKES ARE LIST
    strikes = []
    strikes.append(np.linspace(np.exp(-.1) * S0, np.exp(0.1) * S0, 20) )
    strikes.append(np.linspace(np.exp(-.2) * S0, np.exp(0.2) * S0, 20) )
    strikes.append(np.linspace(np.exp(-.3) * S0, np.exp(0.3) * S0, 20) )
    strikes.append(np.linspace(np.exp(-.5) * S0, np.exp(0.5) * S0, 20) )

    return (fin_option(maturities, strikes)  )


def one_sim(version, option, para_locvol, para_MC, compute_prices=True):
    """Draws one set of random parameters for the ground truth assumption.

    Unless specified diffently, corresponding prices are computed and stored 
    inside a fin_surf instance and additionally inside the data folder.
    Otherwise, precomputed prices will be loaded from the data folder.

    Args:
        version (string): version for implied vola data
        option (fin_option): fin_option instance for this run 
        para_locvol (dict): parameter dict in which loc-vol parameters
                            will be stored 
        para_MC (dict): Dict with the MC-Euler specifications

    Returns:
        finsurf (fin_surface): An instance of the fin_surface class containing 
                               all the previous loc-vol computations
    """   


    lv_sig = {
        'p0':       np.random.uniform(0.4,0.5),
        'p1':       np.random.uniform(0.5,0.7),
        'sig0':     np.random.uniform(0.5,1.7),
        'sig1':     np.random.uniform(0.2,0.4),
        'sig2':     np.random.uniform(0.5,1.7),
        'gamma1':   1.1,
        'gamma2':   20.0,
        'beta1':    0.005,
        'beta2':    0.001,
        'kappa':    0.5,
        'lam1':     10.0,
        'lam2':     10.0,
        'eps_t':    0.1,
        'const_else_factor': 0.4
    }
    para_locvol['alpha_loc_vol_0'] = 0.5
    para_locvol['lv_sig'] = lv_sig
    para_locvol['version'] = version


    finsurf = main_locvol(para_locvol, option, para_MC, compute_prices)

    return(finsurf)


def cali_sabr(finsurface, para_MC):
    '''Wrapper for the SABR calibration part'''
    main_sabr_cali(finsurf,para_MC)


if __name__ == "__main__":
    # Make sure the dest folders exists
    if not os.path.exists('data'):
        os.makedirs('data')
    # Make sure the dest folder exists
    if not os.path.exists('data/locVolPrices'):
        os.makedirs('data/locVolPrices')

    with open('log.txt','w') as file:
                file.writelines(['Log file for the statistical test:',
                3*'\n'])


    para_locvol = {'S0': 100.}
    
    para_MC = {'delta_t': 0.01,
               'N_mc_data': 10**7, # Number of trajectories to compute data 
               'N_mc_inner': 10**6,  # Number of trajectories in a single session run
               'N_mc_train': 400, # How many trajectories per Iteration 
               'N_mc_sabr_cali_iter': 2000,  # How many training steps (SABR)
               'N_mc_lsv_cali_iter': 12000   # How many training steps (LSV)
               }
  
    para_MC['N_mc_runs']= int(np.rint(para_MC['N_mc_data']/para_MC['N_mc_inner']))
    
    option = serve_fin_option(para_locvol['S0'])
    

    for i in VERSIONS:
        tstart = time()
        with open('log.txt','a') as file:
            file.writelines([20*'--',
            '\nI start with step {}'.format(i), 3*'\n'])
        
        try:
            version = str(i).zfill(3)
            tf.reset_default_graph()

            finsurf = one_sim(version, option, para_locvol, para_MC)
            with open('caliRes/'+version+'/'+'finsurf_locVol'+'.pkl', 'wb') as f:
                pickle.dump(finsurf, f)

            with open('data/locVolPrices/finsurf_locVol_{}.pkl'.format(version), 'wb') as f:
                pickle.dump(finsurf, f) 

            tf.reset_default_graph()

            cali_sabr(finsurf, para_MC)
            tf.reset_default_graph()
            
            # Now calibrate the SLV model using the previously calibrated sabr parameters
            # Have everything stored in finsurf at the end, including times etc.
            finsurf = main_caliLSV(finsurf,para_MC)
            print('Step {:4} success:\nComputation time  {}'.format(i, str(
                datetime.timedelta(seconds = np.rint(time()- tstart) ))))
            print(50*'=')

            with open('log.txt','a') as file:
                file.writelines(['\n',
                'Step: {:4} success:\n'.format(i),
                'Time for whole iteration: {}\n'.format(
                    str(datetime.timedelta(seconds = np.rint(time()- tstart))
                    ))])

            # Store the data when successfull with np.savez
            was_success = True
            
            iV_data     = finsurf.lsv_cali['iVdata']
            p_data      = finsurf.prices
            iV_model    = finsurf.lsv_cali['iVmodel']
            p_model     = finsurf.lsv_cali['pM']
            iV_sabr     = finsurf.sabr_cali['iV']
            p_sabr      = finsurf.sabr_cali['prices']
            calitime = finsurf.lsv_cali['cali_time']
            np.savez('caliRes/'+version+'/'+'histData', was_success = was_success,
                    iV_data = iV_data, p_data = p_data, iV_model = iV_model,
                    p_model = p_model, iV_sabr = iV_sabr,  p_sabr = p_sabr,
                    calitime = calitime)

            # Also save finsurf via pickle
            with open('caliRes/'+version+'/'+'finsurf'+'.pkl', 'wb') as f:
                pickle.dump(finsurf, f) 


        except KeyboardInterrupt:
            print('\n')
            print(40*'=')
            print(40*'=')
            print(10*'='+ '  The user aborted'  + 10*'=' )
            sys.exit()
        except Exception as e:
            was_success = False
            np.savez('caliRes/'+version+'/'+'histData', was_success=was_success)
            print(e)
            print('Step {:4} failed'.format(i))
            with open('log.txt','a') as file:
                file.writelines(['\n',
                'Step: {:4} failed. I continue with the next...\n'.format(i)])

        with open('log.txt','a') as file:
            file.writelines([20*'--','\n',20*'--'])


