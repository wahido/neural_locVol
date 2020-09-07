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

"""runfile_cali.py:
This file implements the networks for training the leverage function,
the Black Scholes hedges and the ground truth assumption.

ToDo: Add documentation
"""


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, tensorflow as tf

from scipy.stats import norm as normal_handler
import os
from time import time
import datetime
from compute_pM import hedge_error
from neural_nets_hedging import sigma_SLV
from finModels.helpers_finModels import fin_surface
from contextlib import redirect_stdout

# Env variables
HOSTNAME = os.uname()[1]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def main(finsurf, paraMC, just_ops = False):
    assert isinstance(finsurf, fin_surface)
    version = finsurf.version

    with open('caliRes/'+version+'/log.txt', 'a') as f:
            with redirect_stdout(f):
                print('\nON HOST {}\nThe SABR Parameters we use are:\n\n'.format(HOSTNAME))
                print('{:10}: {:10.4e}'.format('alpha_0', finsurf.sabr_alpha0) )
                print('{:10}: {:10.4e}'.format('nu', finsurf.sabr_nu))
                print('{:10}: {:10.4e}'.format('rho', finsurf.sabr_rho))
    
    
    print(10*'='+' cali on host {} '.format(HOSTNAME)+10*'=',end='\n\n')
    

    print('I start with calibrating version {}'.format(version))


    option = finsurf.T_K
    strikes_tf   = [tf.placeholder(dtype=tf.float32, shape=(option.nK[i]) ) for i in range(option.nT) ]


    maturities, strikes = finsurf.T_K.T, finsurf.T_K.K
    log_m = finsurf.log_m_scale_list
    lenT  = finsurf.nT
    t_grid_nn  = np.append(0,maturities[:-1])
    sig = sigma_SLV(t_grid_nn)

    Batchsize_tf = tf.placeholder_with_default(input=100, shape=()) 
    weights_default = [np.ones(dtype=np.float32, shape=[lK]) for lK in finsurf.nK  ]
    
    weights_tf_list = [ tf.placeholder_with_default(input=w, shape=w.shape) for w in weights_default   ]

    if just_ops:
        pricesList_tfCall, pricesList_tfPut= hedge_error(
        finsurf=finsurf, para_MC=paraMC, Batchsize_tf=Batchsize_tf, sig=sig, use_hedges=True, weights_list=weights_tf_list, strikes = strikes_tf,
        just_ops = just_ops
        )
        x_sig_tf   = tf.placeholder(tf.float32, shape=[None,1])
        sig_SLV_op = []
        for t in t_grid_nn:
            sig_SLV_op.append(sig(t, x_sig_tf) )

        return(pricesList_tfCall,
               pricesList_tfPut,
               Batchsize_tf,
               strikes_tf,
               x_sig_tf,
               sig_SLV_op
               )
   
    else:
        cost, pricesList_tf= hedge_error(
            finsurf=finsurf, para_MC=paraMC, Batchsize_tf=Batchsize_tf, sig=sig, use_hedges=True, weights_list=weights_tf_list, strikes = strikes_tf,
            just_ops = just_ops
            )

    D = dict(zip(strikes_tf, option.K) )
    
    # Contruct the lists for the different optimizers
    list_vars = tf.trainable_variables()
    lis_vars_perMat = []
    for i in range(lenT): lis_vars_perMat.append([])

    while len(list_vars)>0:
        list_var_el = list_vars.pop()
        FLAG_TMP = 0
        for idx_mat in range(lenT):
            if ('_timegrid_'+str(idx_mat)) in list_var_el.name:
                lis_vars_perMat[idx_mat].append(list_var_el)
                FLAG_TMP = 1
        assert FLAG_TMP == 1


    print('\n I now create the optimizers')
    optim_list = []

    init_list  = []
    for i in range(lenT):
        print('I am creating optimizer {:4} of {:4}'.format(i+1, lenT))
        list_to_opt = lis_vars_perMat[i]
        optim_aux = tf.train.AdamOptimizer(.001)
                                  
        optim_list.append(optim_aux.minimize(cost[i],
            var_list=list_to_opt ) )
        init_list.append( tf.variables_initializer(var_list = list_to_opt ) )
    print('\n Done...')


    print('\n I now create the eval ops for plotting...\n\n')

    # Define the op that plots the trained sigma, plot at strikes
    x_sig_tf   = tf.placeholder(tf.float32, shape=[None,1])
    
    sig_SLV_op = []
    for t in t_grid_nn:
        sig_SLV_op.append(sig(t, x_sig_tf) )
    print('Done!\n\n')


    init = tf.global_variables_initializer()

    x_sig_list = []
    pM_list = []
    cali_time = []
    sigma_vals = []
    trainSteps = []
    iVmodel    = []
    iVdata     = []


    with tf.Session() as sess:
        saver0 = tf.train.Saver()
        t_big_bang = time()
        sess.run(init)
        print('\nI start with the training session... \n\n')
        print(3*'\n'+10*'- + ',end='\n\n')
        with open('caliRes/'+version+'/log.txt', 'a') as f:
            with redirect_stdout(f):
                print('\nI start with the training session... \n\n')
                print(3*'\n'+10*'- + ',end='\n\n')
        
        for iter_mat, matu_now in enumerate(maturities):
            
            threshhold = 0.0045

            N_MC_train = paraMC['N_mc_train']
            N_outer_simMC = paraMC['N_mc_runs']
                
            alphavega = finsurf.iV[iter_mat]
            d1 = np.log(finsurf.spot/strikes[iter_mat]  )   + 0.5 * (alphavega)**2 * maturities[iter_mat]
            d1 /= alphavega * np.sqrt(maturities[iter_mat])
            vega = finsurf.spot * normal_handler.pdf(d1) * np.sqrt(maturities[iter_mat])
            weights_step =   (1/vega) 
            weights_step /= np.sum(weights_step)

            t_big_bang_slice = time()
            print('I start training for maturity {:4} of {:4}'.format(iter_mat+1, lenT))
            with open('caliRes/'+version+'/log.txt', 'a') as f:
                with redirect_stdout(f):
                    print('I start training for maturity {:4} of {:4}'.format(iter_mat+1, lenT))

            
            train = 0
            trainReal = 0
            FLAG_keepTraining = True

            print(3*'\n'+5*'='+' Using vega, the weights are :  ' + 5*'=' )
            print(weights_step)
            print(35*'=')
            print('\n\n')

            while FLAG_keepTraining:

                train += 1
                trainReal += 1
                if trainReal >= paraMC['N_mc_lsv_cali_iter']:
                    FLAG_keepTraining = False
                    print('\n'+20*'=')
                    print('STOPPING training since MAX iter reached')
                    print('\n'+20*'=')

                
                if train == 500:
                    N_MC_train = 2000
                
                elif train == 1500:
                    N_MC_train = 10000
                elif train == 4000 :
                    N_MC_train = 50000
                        
                if train >= 5000 and (train%1000)==0:       

                    pM_val = sess.run(pricesList_tf[iter_mat],
                    feed_dict={**D,Batchsize_tf: paraMC['N_mc_inner']})
                    print('\rCompleted {:5} of {:5} MC rounds'.format(1, N_outer_simMC), end='')
                    for run in range(N_outer_simMC-1):
                        pM_val += sess.run(pricesList_tf[iter_mat], feed_dict={**D, Batchsize_tf: paraMC['N_mc_inner']}  )
                        print('\rCompleted {:5} of {:5} MC rounds'.format(run+2, N_outer_simMC), end='')
                    pM_val *= 1/N_outer_simMC 

                    iV_lsv_slice = finsurf.get_model_iv_slice_mixed(model_price=pM_val, slice=iter_mat)


                    implVola_d_slice = finsurf.iV[iter_mat]
                    ivErrorRaw = np.abs(iV_lsv_slice - implVola_d_slice)

                    # Update the weights
                    weights_step = weights_step + 1.0 * ivErrorRaw
                    weights_step = weights_step/np.sum(weights_step)

                    if ivErrorRaw.max() <= threshhold:
                        FLAG_keepTraining = False
                        print('\n'+20*'=')
                        print('STOPPING training since SUCCESS')
                        print('\n'+20*'=')

                    elif (ivErrorRaw.max() > 0.01 and train == 3000):
                        if FLAG_keepTraining:
                            sess.run( init_list[iter_mat] )
                            N_MC_train = paraMC['N_mc_train']
                            
                            train = 0
                            print(2*'\n')
                            print(20*'=')
                            print(7*'='  + '  RESTARTING  SLICE  '  + 7*'=' )
                            with open('caliRes/'+version+'/log.txt', 'a') as f:
                                with redirect_stdout(f):
                                    print(15*'=')
                                    print(7*'='  + '  RESTARTING  SLICE  '  + 7*'=' )
                                    print('\n\n')


                    print(30*'=')
                    print('\n\niVErrorMax at step {}: {}\n\n'.format(train,ivErrorRaw.max()))
                    print(30*'=')

                    strike_low  = finsurf.log_m_scale_list[iter_mat][0]
                    strike_high = finsurf.log_m_scale_list[iter_mat][-1]

                    x_sig_feed = np.linspace(strike_low, strike_high,100)                    
                    x_sig_feed_shaped = np.reshape(x_sig_feed, [len(x_sig_feed), 1])

                    sig_SLV_val = sess.run(sig_SLV_op[iter_mat], feed_dict={x_sig_tf: x_sig_feed_shaped  })
                    sig_SLV_val = np.reshape(sig_SLV_val, [-1])
    
                    # Now make all the plots
                    train_for_string = str(  int(np.rint(trainReal/10) ) ).zfill(5)
                    
                    # 1) sig**2
                    # Plot sig_square
                    plt.plot(x_sig_feed, sig_SLV_val**2, label='trained lev. func. $L^2(t,x)$')
                    plt.xlabel('log-price $x$')
                    plt.title('Training after {} steps, maturity {} of {}'.format(trainReal,iter_mat+1, lenT))
                    plt.legend()
                    plt.savefig('caliRes/'+version+'/'+'mat_'+str(iter_mat+1)+'__'+train_for_string+'sig_square_trained.png',dpi = 300)
                    plt.close()

                    # 2) iv model and iv data
                    # Compute the iV of the calibrated model. The prices of the slice are
                    # pM_val, the current slice is iter_mat. 
                    implVola_d_slice = finsurf.iV[iter_mat]
                    log_m   =  finsurf.log_m_scale_list[iter_mat]
                    plt.plot( log_m, implVola_d_slice, label='data')
                    plt.plot( log_m, iV_lsv_slice, label='model lsv')
                    plt.legend()
                    plt.title('impl. vol. for T = {:4.4}'.format(matu_now))
                    plt.savefig('caliRes/'+version+'/''mat_'+str(iter_mat+1)+'__'+train_for_string+'implVola.png',dpi = 300)
                    plt.close()

                    # 3) iv errors
                    # # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- # #
                    plt.plot( log_m, implVola_d_slice - iV_lsv_slice, label='data - model_LSV')
                    plt.title('impl. vol. error for T = {:4.4}'.format(matu_now))
                    plt.legend()
                    plt.savefig('caliRes/'+version+'/'+'mat_'+str(iter_mat+1)+'__'+train_for_string+'implVola_err.png',dpi = 300)
                    plt.close()
                

                if not FLAG_keepTraining:
                    sigma_vals.append(sig_SLV_val)
                    pM_list.append(pM_val)
                    x_sig_list.append(x_sig_feed)
                    iVmodel.append(iV_lsv_slice)
                    iVdata.append(implVola_d_slice)
                    trainSteps.append(trainReal)
                        
                    print('\n'+7*'- + = + -')
                    print(7*'- + = + -')
                    print('')
                    time_cali_slice = time()-t_big_bang_slice
                    cali_time.append(time_cali_slice)
                    print('Training / Calibration time: {:15} for {:8} steps'.format( str(datetime.timedelta(seconds = np.rint(time_cali_slice ) )), trainReal )
                    )

                    with open('caliRes/'+version+'/log.txt', 'a') as f:
                        with redirect_stdout(f):
                            print(3*'\n')
                            print(15*'- + = + -')
                            print('Training / Calibration time: {:15} for {:8} steps\n\n\n'.format( str(datetime.timedelta(seconds = np.rint(time_cali_slice ) )), trainReal )
                    )

                    with open('caliRes/'+version+'/log.txt', 'a') as f:
                        with redirect_stdout(f):
                            print(15*'=')
                            print('Training / Calibration time: {:15} for {:8} steps'.format( str(datetime.timedelta(seconds = np.rint(time_cali_slice ) )), trainReal )
                    )
                            print('\n\n')
                            print(35*'=')
                            print(10*'='+'  END OF SLICE '+10*'=')
                            print(35*'=')
                    break
                    
                if (train)%500 == 0:
                    print(1*'\n')
                    print(30*'=')
                    print('Ntrain: {}'.format(N_MC_train))
                    print(30*'=')

                    tstart = time()
                    # Get the current cost
                    _,cost_now = sess.run([ optim_list[iter_mat] , cost[iter_mat] ] , 
                            feed_dict={**D,Batchsize_tf: N_MC_train, weights_tf_list[iter_mat]: weights_step })                
                    
                    tend = time()
                    print(24*'- + ')
                    print('Maturity {:3}/{:3} | train step: {:6}  |  cost: {:10.4e} |  elapsed time:{:5.2f} | overall time: '.format( iter_mat+1,lenT,  trainReal, cost_now, tend-tstart)     +str(datetime.timedelta(seconds = np.rint(tend-t_big_bang) )), end='\n')
                    print('Phase step: {:6} '.format(train), end='\n')
                    
                    
                    with open('caliRes/'+version+'/log.txt', 'a') as f:
                        with redirect_stdout(f):
                            print(24*'- + ')
                            print('Maturity {:3}/{:3} | train step: {:6}  |  cost: {:10.4e} |  elapsed time: {:5.2f} | overall time: '.format( iter_mat+1,lenT,  trainReal, cost_now, tend-tstart)+str(datetime.timedelta(seconds = np.rint(tend-t_big_bang) )), end='\n')
                            print('Phase step: {:6} '.format(train), end='\n')
                else:
                    
                    sess.run(optim_list[iter_mat], feed_dict={**D,Batchsize_tf: N_MC_train , weights_tf_list[iter_mat]: weights_step } )
            
        # All values for all slices have been computed now. Next, insert the values into finsurf,
        # compute the iV and store it.

        # Store results in finsurf
        finsurf.store_lsv_cali(pMlist=pM_list, x_sig_list=x_sig_list,
                              cali_time=cali_time, sigma_vals=sigma_vals, trainSteps=trainSteps,
                              iVmodel = iVmodel, iVdata = iVdata)

        # the model iV per slice is available via finsurf.get_iV_lsv_m(slice=...) which also returns
        # the corresponding log moneyness

        # store training (weights) to disk via saver0
        saver0.save(sess, 'caliRes/'+version+'/train_tf')

    print('\nI have saved the files')
    return(finsurf)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
