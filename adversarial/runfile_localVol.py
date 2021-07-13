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

"""runfile_localVol.py:
This file implements the the functionality for computing and plotting local-vol 
implied volatilities for the sampled parameters given as input and options also
given as input. Starting point is the main function.
"""



import matplotlib.pyplot as plt
import numpy as np, scipy as sp, tensorflow as tf
# convenience
from numpy import log, exp # pylint: disable=E0611
# system
import os, sys, argparse
from optparse import OptionParser
from time import time
from time import sleep
import datetime

from tensorflow.python import data
from compute_pM import MC_pM_locvol
from neural_nets_hedging import sigma_SLV_true
from finModels.BlackScholes import get_impl_volaSurface
from finModels.helpers_finModels import fin_surface

from contextlib import redirect_stdout
import pickle

# Env variables
HOSTNAME = os.uname()[1]

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# User packages
from finModels.BlackScholes import  BS_price
import helpers


def makePrices(para_locVol, option, para_MC, just_ops = False):
    # Compute data if user specified so, don't forget to store the data
    version = para_locVol['version']
    if not os.path.exists('caliRes/'+version):
        os.mkdir('caliRes/'+version)
    
    Batchsize_tf = tf.placeholder(dtype=tf.int32, shape=())

    strikes_tf   = [tf.placeholder(dtype=tf.float32, shape=(option.nK[i]) ) for i in range(option.nT) ]

    pM_tf = MC_pM_locvol(para_locVol, option, para_MC, Batchsize_tf, strikes_tf)

    if just_ops:
        return(pM_tf, Batchsize_tf, strikes_tf)
    

    N_mc_data = para_MC['N_mc_data']
    N_mc_inner = para_MC['N_mc_inner']

    
    helpers.init_folders_and_readme(mode='create_data', version=version)

    print('I run this script on host: {}'.format(HOSTNAME), version)


    # Check if Data version already exists
    if os.path.isfile('caliRes/pM_'+version+'.pkl'):
        if version != 'test':
            raise Exception('Data file already exists. Give new name to store it.\n\n')




    # Compute the number of MC rounds we need to get to Batchsize
    N_MC_outer = int(np.trunc(N_mc_data/N_mc_inner))

    D = dict(zip(strikes_tf, option.K) )
    with tf.Session() as sess:
        print('\nData computation progress:')
        pML = sess.run(
            pM_tf,
            feed_dict={**D,Batchsize_tf: N_mc_inner}
        )
        
        for i in np.arange(N_MC_outer - 1):
            prog = 100*i /(N_MC_outer)
            print('\rMC simulation {:7} | progress: {:10.4}%  '.format(i,prog), end='')
            pM_auxL = sess.run(pM_tf, feed_dict={**D,Batchsize_tf: N_mc_inner})

            for it_pM, pM in enumerate(pML):
                for it_mat in range(len(pM)):
                    pML[it_pM][it_mat] = (  pML[it_pM][it_mat]
                                          + pM_auxL[it_pM][it_mat]  )

        for it_pM, pM in enumerate(pML):
                for it_mat in range(len(pM)):
                    pML[it_pM][it_mat] = pML[it_pM][it_mat] / N_MC_outer
                                          
    
    # Save the prices to disk for later calibration
    with open('data/locVolPrices/pML_'+version+'.pkl', 'wb') as f:
       pickle.dump(pML, f) 

    



def plot_IV(para, option):
    log_m = option.log_m_list(para['S0'])
    version = para['version']

    with open('data/locVolPrices/pML_'+version+'.pkl', 'rb') as f:
       pML = pickle.load(f)



    # Constuct the objects that handles conversion
    data_handler = [
        fin_surface(mats=option.T, strikes=option.K, spot=para['S0'], version = version)
        for pM in pML
    ]
    
    for d, pM in zip(data_handler, pML):
        d.paralocVol = para
        d.feed_prices(prices= pM  )
        d.convert(direction='price->iV')


    if not os.path.exists('caliRes/'+version):
        os.mkdir('caliRes/'+version)
    
    
    # Create the plots
    for itmat in range(len(pML[0])):
        for it_locV, data in enumerate(data_handler):
            plt.plot(log_m[itmat], data.iV[itmat])
        plt.savefig('caliRes/'+version+'/plot_{}.png'.format(str(itmat+1).zfill(3)))
        plt.close()


    return(data_handler)


def main(para_locVol, option, para_MC):

    makePrices(para_locVol, option, para_MC)
    finsurf = plot_IV(para_locVol, option)
    
    return(finsurf)


if __name__ == "__main__":
    pass