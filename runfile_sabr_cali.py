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

"""runfile_sabar_cali.py:
This file implements the SABR calibration part.
"""




# science
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, scipy as sp, tensorflow as tf
# convenience
from numpy import log, exp # pylint: disable=E0611
import os

from compute_pM import MC_errors_SABR
from neural_nets_hedging import sigma_SLV
from finModels.BlackScholes import get_impl_volaSurface

# Env variables
HOSTNAME = os.uname()[1]

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# User packages
import helpers




def main(finsurf, paraMC):
    '''Entry point for the SABR calibration part.'''

    print('\n\n\nCalibrating SABR Parameters for version {}'.format(finsurf.version))
    N_MC_inner = paraMC['N_mc_inner']
    N_MC_runs = paraMC['N_mc_runs']
    N_MC_train = paraMC['N_mc_train']
    N_mc_sabr_cali_iter = paraMC['N_mc_sabr_cali_iter']

    helpers.init_folders_and_readme(mode='calibrate SABR', version=finsurf.version)


    Batchsize_tf = tf.placeholder_with_default(input=10, shape=())


    cost_all_tf, sabr_paras, prices_tf = MC_errors_SABR(finsurf.T_K, Batchsize_tf, paraMC, finsurf.spot, finsurf.prices,finsurf.iV)

    # Create the optimizers
    optim = tf.train.AdamOptimizer(.001).minimize(cost_all_tf )
    init  = tf.global_variables_initializer()

    with tf.Session() as sess:
        print('I start with the pure SABR calibration')
        sess.run(init)
        cost_now, alpha0, nu, rho = sess.run([cost_all_tf, sabr_paras['alpha_0'], sabr_paras['nu'], sabr_paras['rho']], feed_dict={Batchsize_tf: N_MC_train  })
        print('\n\nThe initial parameters are:')
        print('COST: {:15.4e}\nrho: {:15.3e} | nu: {:15.3e} | alpha0: {:15.3e}'.format(cost_now[0],rho[0],nu[0],alpha0[0]))
        print(20*'+')
        print('I start with the training:')
        for i in range(1,N_mc_sabr_cali_iter+1):
            sess.run(optim, feed_dict={Batchsize_tf: N_MC_train  })
            if i%500==0:
                cost_now, alpha0, nu, rho = sess.run([cost_all_tf, sabr_paras['alpha_0'], sabr_paras['nu'], sabr_paras['rho']],
                                                      feed_dict={Batchsize_tf: N_MC_train  })
                print('STEP: {:10} - COST: {:15.4e}\nrho: {:15.3e} | nu: {:15.3e} | alpha0: {:15.3e}'.format(i,cost_now[0],rho[0],nu[0],alpha0[0]))
                print(20*'-')
                N_MC_train *= 2
                print(3*'\n')
                print(30*'=')
                print('Ntrain: {}'.format(N_MC_train))
                print(30*'=')
                print(3*'\n')
                if N_MC_train > 1000:
                    N_MC_train = 1000


        prices = [np.zeros([iter]) for iter in finsurf.T_K.nK ]
        for i in range(N_MC_runs):
            prices_aux = sess.run(prices_tf, feed_dict={Batchsize_tf: N_MC_inner})
            print('\rRun {:4}/{:4}     '.format(1+i,N_MC_runs), end='')
            for i in range(len(prices)):
                prices[i] += prices_aux[i]

        for i in range(len(prices)):
            prices[i] = prices[i]/N_MC_runs
        print('\n'+20*'=')
        # I now have the calibrated sabr parameters, I store them in the finsurf together with prices
        finsurf.feed_sabr_cali(rho=rho[0],alpha0=alpha0[0],nu=nu[0], prices=prices)
        finsurf.convert(direction='price_sabr->iV_sabr')
        print('SABR parameters stored')
        print(3*'\n')
        
        for iter_mat in range(finsurf.T_K.nT):
            log_m = finsurf.log_m_scale_list[iter_mat]
            matu_now = finsurf.T_K.T[iter_mat]

            implVola_m = finsurf.sabr_cali['iV'][iter_mat]
            implVola_d = finsurf.iV[iter_mat]

            plt.plot( log_m ,implVola_d, label='data')
            plt.plot( log_m ,implVola_m, label='sabr')
            plt.legend()
            plt.title('impl. vol.sabr for T = {:4.4}'.format(matu_now))
            plt.savefig('caliRes/'+finsurf.version+'/''mat_'+str(iter_mat+1)+'_sabr_'+'implVola.png',dpi = 300)
            plt.close()

if __name__ == "__main__":
    pass