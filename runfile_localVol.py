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
import numpy as np, tensorflow as tf
import os

from compute_pM import MC_pM_locvol
from finModels.helpers_finModels import fin_surface

import pickle
import helpers

# Env variables
HOSTNAME = os.uname()[1]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 



def makePrices(para_locVol, option, para_MC, just_ops = False):
    '''Compute locVol prices for the given parameters. '''
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
        pM = sess.run(
            pM_tf,
            feed_dict={**D,Batchsize_tf: N_mc_inner}
        )
        prog = 100 /(N_MC_outer)
        print('\rMC simulation {:7} | progress: {:10.4}%  '.format(1,prog), end='')
        for i in np.arange(2, N_MC_outer+1):
            prog = 100*i /(N_MC_outer)
            pM_aux = sess.run(pM_tf, feed_dict={**D,Batchsize_tf: N_mc_inner})

            pM = [ pM[iteraux] +   pM_aux[iteraux] for iteraux in range( len(pM) )     ]
            
            print('\rMC simulation {:7} | progress: {:10.4}%  '.format(i,prog), end='')

        pM = [pM[iteraux] / N_MC_outer for iteraux in range( len(pM) )  ]

    # Save the prices to disk for later calibration
    with open('data/locVolPrices/pM_'+version+'.pkl', 'wb') as f:
       pickle.dump(pM, f) 

def plot_IV(para, option):
    '''load computed prices and create fin_surf instance. Also do all needed conversions'''

    log_m = option.log_m_list(para['S0'])
    version = para['version']

    with open('data/locVolPrices/pM_'+version+'.pkl', 'rb') as f:
       pM = pickle.load(f)

    # Constuct the object that handles conversion
    data_handler = fin_surface(mats=option.T, strikes=option.K, spot=para['S0'], version = version)
    data_handler.paralocVol = para
    data_handler.feed_prices(prices= [p for p in pM]  )    
    data_handler.convert(direction='price->iV')

    for i in range(len(pM)):
        plt.plot(log_m[i], data_handler.iV[i] )
        if not os.path.exists('caliRes/'+version):
            os.mkdir('caliRes/'+version)
        plt.savefig('caliRes/'+version+'/plot_{}.png'.format(str(i+1).zfill(3)) )
        plt.close()

    return(data_handler)


def main(para_locVol, option, para_MC, compute_prices):

    '''wrapper function. Depending on compute_prices data is computed or just loaded.'''
    if compute_prices:
        makePrices(para_locVol, option, para_MC)
    
    finsurf = plot_IV(para_locVol, option)
    return(finsurf)


if __name__ == "__main__":
    pass