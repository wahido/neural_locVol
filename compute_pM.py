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

"""compute_pM.py:
This file implements the tensorflow op creation functionality.

ToDo: Add documentation
"""



import numpy as np
import scipy as sp
import tensorflow as tf
import os, sys
from neural_nets_hedging import sigma_SLV_true, sigma_SLV, strategy_SLV, strategy_LV
from numpy import log, exp

import time

# THIS FOR CALIBRATING THE SABR PARAMETERS
def MC_errors_SABR(option,Batchsize_tf, para_MC, S0, pM_data, iV_data):
    # Define the model variables we wish to calibrate
    with tf.variable_scope("pure_SABR_cali", reuse=tf.AUTO_REUSE):
        rho_tf = tf.get_variable("rho", [1], initializer=tf.constant_initializer(value=-0.5) )
        nu_tf = tf.get_variable("nu", [1],  initializer=tf.constant_initializer(value=0.1) )
        alpha_0_tf = tf.get_variable("alpha0", [1], initializer=tf.constant_initializer(value=0.7) )
    sabr_paras = {'rho':rho_tf,'nu':nu_tf,'alpha_0': alpha_0_tf}

    strikes = option.K # this should be a list of strikes indexed by maturities
    num_mats = option.nT
    maturities = option.T

    delta_t = para_MC['delta_t']
    N_t_step = option.N_t_step(para_MC)
    N_t_final = N_t_step[-1]
    T_final = maturities[-1]

    taux = np.linspace(0,T_final,num=(N_t_final+1), dtype=np.float32 ) 

    lenS = [ len(strikes[i]) for i in range(num_mats)   ]
    

    Z_incr = tf.random_normal([Batchsize_tf, N_t_final], mean=0.0, stddev=np.sqrt(delta_t))
    Z_aux = tf.random_normal([Batchsize_tf, N_t_final], mean=0.0, stddev=np.sqrt(delta_t))


    # this needs to be a list too
    strikes_tf = [tf.constant(strikes[i], dtype = tf.float32, shape=[lenS[i]]) for i in range(num_mats) ]    # For weights
    K_tf = [tf.constant(strikes[i], dtype=tf.float32, shape=[1, lenS[i]])  for i in range(num_mats) ]     # For costs

    t_vec = tf.constant(taux, dtype=tf.float32)
    
    W_incr = rho_tf * Z_incr + tf.sqrt(1- rho_tf**2) * Z_aux
    W = tf.concat([tf.zeros([Batchsize_tf,1]), tf.cumsum(W_incr, axis=1  )], axis = 1)
   
    # Compute the solution vector for alpha
    alpha_vec = alpha_0_tf * tf.exp( (-0.5 * nu_tf**2 * t_vec + nu_tf * W), name='sim_alpha')

     # Now we make the Euler-iterations.
    X =  tf.zeros(shape=[Batchsize_tf,1], dtype=tf.float32)

    cost_tf = []
    prices_tf = []
    mat_counter = -1

    StochInt = [tf.zeros([Batchsize_tf,lenSthismat]) for lenSthismat in lenS]
    h_SABR = strategy_LV(strikes, maturities)

    for step in np.arange(N_t_final):

        alpha = tf.reshape(alpha_vec[:,step], shape = [-1,1] )
        
        # Make one MC time step
        Z_i = tf.reshape(Z_incr[:,step], shape = [-1,1] )
        print('\rX-step: {:4} of {:4} , t = {:4.4}   '.format(step+1, N_t_final, taux[step+1]), end='')

        Delta_X = (- 0.5* tf.square(alpha) *  delta_t + alpha* Z_i )
        
        for matiter in range(mat_counter+1,num_mats):
            StochInt[matiter] = StochInt[matiter] + h_SABR(taux[step],S0* tf.exp(X), alpha, matiter ) * S0*(tf.exp(X+Delta_X)-tf.exp(X) )

        # Update X
        X += Delta_X
        # X has now the updated value after one time step

        # Here we check if the current time step corresponds to a time to maturity. If
        # yes, we compute the prices for all strikes and store the prices 
        if (step+1) in N_t_step:
            mat_counter += 1
            prices_data_slice_tf = tf.constant(pM_data[mat_counter],
                                shape=[1,lenS[mat_counter]], dtype=tf.float32)

            prices_data_slice_tf_batched = tf.tile(prices_data_slice_tf, [Batchsize_tf,1])

            F = S0*tf.exp(X)
            Fmat = tf.tile(F,(1,lenS[mat_counter]))
            Kmat = tf.tile(K_tf[mat_counter], (Batchsize_tf,1))
            
            payMat = Fmat - Kmat
            payMat = tf.maximum(payMat, 0)
            
            hedge_error = payMat - prices_data_slice_tf_batched - tf.stop_gradient(StochInt[mat_counter])
            price_slice = payMat - tf.stop_gradient(StochInt[mat_counter])
            
            alpha_vega = np.float32(iV_data[mat_counter]) 
    
            d1 = tf.log(S0 /strikes_tf[mat_counter]) + 0.5 * tf.square(alpha_vega)*(maturities[mat_counter] )
            d1 = d1 / (alpha_vega * tf.sqrt(np.float32(maturities[mat_counter] ) ))
            
            norm_dist_tf = tf.distributions.Normal(loc=0., scale=1.)
            vega = S0 * norm_dist_tf.prob(d1) * np.sqrt(maturities[mat_counter] )

            weights = tf.reciprocal(vega)
            weights = weights/ tf.reduce_sum(weights)
            weights = tf.reshape(weights, [-1, lenS[mat_counter]])



            price_slice = tf.reduce_mean(price_slice, axis = 0)
            prices_tf.append(price_slice)
            
            cost = tf.reduce_mean(hedge_error, axis = 0)
            cost = tf.square(cost)
            cost = cost * weights
            cost = tf.reduce_sum(cost)

            cost_tf.append(cost)

    # We only calibrate to the first slice but this can be changed by uncommenting
    # below
    JUSTFIRSTSLICE = True

    if JUSTFIRSTSLICE:
        cost_all_slices_tf = cost_tf[0] 
    else:
        for it_cost, costslice in enumerate(cost_tf):
            if it_cost == 0:
                cost_all_slices_tf = costslice
            else:
                cost_all_slices_tf += costslice

    # Add a penalty to ensure valid values for rho
    cost_all_slices_tf += tf.reciprocal(1.0 -  tf.maximum(0.9,sabr_paras['rho']**2)    )
   
    return(cost_all_slices_tf, sabr_paras, prices_tf)


# For the loc vol data generation
def MC_pM_locvol(para, option, para_MC, Batchsize_tf, strikes_tf):
    sigsquare = sigma_SLV_true()
    sigsquare.setconstants(**para['lv_sig'], version=para['version'])

    strikes = strikes_tf
    num_mats = option.nT
    maturities = option.T
    assert len(maturities)==num_mats 

    alpha0 = para['alpha_loc_vol_0']
    S0     = para['S0']

    delta_t = para_MC['delta_t']
    N_t_step = option.N_t_step(para_MC)
    N_t_final = N_t_step[-1]
    T_final = maturities[-1]

    taux = np.linspace(0,T_final,num=(N_t_final+1), dtype=np.float32 ) 

    # lenS should be a list indexed by maturity that returns the 
    # number of strikes available at that maturity
    lenS = option.nK
        
    K_tf = [tf.reshape(K, (1,-1)) for K in strikes]
   
    Z_incr = tf.random_normal([Batchsize_tf, N_t_final], mean=0.0, stddev=np.sqrt(delta_t))
    
    # Now we make the Euler-iterations.
    X =  tf.zeros(shape=[Batchsize_tf,1], dtype=tf.float32)

    pM_tf = []
    mat_counter = -1

    StochInt = [tf.zeros([Batchsize_tf,lenSthismat]) for lenSthismat in lenS]
    h_LV = strategy_LV(strikes, maturities)

    for step in np.arange(N_t_final):

        # Make one MC time step
        Z_i = tf.reshape(Z_incr[:,step], shape = [-1,1] )
        print('\rX-step: {:4} of {:4} , t = {:4.4}   '.format(step+1, N_t_final, taux[step+1]), end='')

        Delta_X = ( - 0.5 * alpha0**2 * sigsquare(taux[step], X) * delta_t 
            + alpha0 * sigsquare.vola(taux[step], X) * Z_i
            )

        # Update the stochastic integrals
        for matiter in range(num_mats):
            StochInt[matiter] = StochInt[matiter] + h_LV(taux[step], S0*tf.exp(X), alpha0 * sigsquare.vola(taux[step], X), matiter ) * S0*(tf.exp(X+Delta_X)-tf.exp(X) )


        X += Delta_X
        # X has now the updated value after one time step

        # Here we check if the current time step corresponds to a time to maturity. If
        # yes, we compute the prices for all strikes and store the prices 
        if (step+1) in N_t_step:
            mat_counter += 1
            F = S0*tf.exp(X)
            Fmat = tf.tile(F,(1,lenS[mat_counter]))
            Kmat = tf.tile(K_tf[mat_counter], (Batchsize_tf,1))
            
            payMat = Fmat - Kmat
            payMat = tf.maximum(payMat, 0)
            payMat = payMat - StochInt[mat_counter]
            price_one_mat = tf.reduce_mean(payMat, axis = 0)

            pM_tf.append(price_one_mat)

    assert (1+mat_counter) == num_mats
    return pM_tf


# This for the LSV calibration
def hedge_error(*,finsurf, para_MC, Batchsize_tf, sig, use_hedges, weights_list, strikes, just_ops):

    mat_counter = -1
    cost_tf = []

    if just_ops:
        prices_tfCall = []
        prices_tfPut  = []
    else:
        prices_tf = []
    
    pM_data = finsurf.serve_mixed_prices()

    atmInd  = [finsurf.find_atm_index(i) for i in range(finsurf.nT)]
    call_putt_hedge_vec = []
    for auxIterMat in range(finsurf.nT):
        aux     = np.ones([finsurf.nK[auxIterMat]]  , dtype=np.float32)
        for auxIterATM in range(atmInd[auxIterMat]):
            aux[auxIterATM] = np.float32(0.0)

        call_putt_hedge_vec.append(tf.constant(  aux   , dtype=tf.float32) )

    maturities = finsurf.maturities
    K_tf = [tf.reshape(K, (1,-1)) for K in strikes]


    num_mats = finsurf.nT

    alpha0 = finsurf.sabr_alpha0
    nu     = finsurf.sabr_nu
    rho    = finsurf.sabr_rho
    S0     = finsurf.spot

    delta_t = para_MC['delta_t']
    N_t_step = finsurf.N_t_step(para_MC)
    N_t_final = N_t_step[-1] 
    T_final = maturities[-1] 

    taux = np.linspace(0,T_final,num=(N_t_final+1), dtype=np.float32 )
    
    lenS = finsurf.T_K.nK

    # StochInt as list conatainer
    StochInt = [tf.zeros([Batchsize_tf,lenSthismat], dtype=tf.float32) for lenSthismat in lenS]
    if just_ops:
        StochInt2 = [tf.zeros([Batchsize_tf,lenSthismat],
                                dtype=tf.float32)  for  lenSthismat in lenS]

    Z_incr = tf.random_normal([Batchsize_tf, N_t_final], mean=0.0, stddev=np.sqrt(delta_t), dtype=tf.float32)
    Z_aux = tf.random_normal([Batchsize_tf, N_t_final], mean=0.0, stddev=np.sqrt(delta_t), dtype=tf.float32)


    t_vec = tf.constant(taux, dtype=tf.float32)
    
    W_incr = rho * Z_incr + np.sqrt(1- rho**2) * Z_aux
    W = tf.concat([tf.zeros([Batchsize_tf,1]), tf.cumsum(W_incr, axis=1  )], axis = 1)
   
    # Compute the solution vector for alpha
    alpha_vec = alpha0 * tf.exp( (-0.5 * nu**2 * t_vec + nu * W), name='sim_alpha')

    X =  tf.zeros(shape=[Batchsize_tf,1], dtype=tf.float32)

    h2 = strategy_SLV('strat_X_mat_' , strikes, maturities)
    
    print('Starting with the MC steps')
    print(20*'-')
    for step in np.arange(N_t_final):

        # DO the steps here
        alpha = tf.reshape(alpha_vec[:,step], shape = [-1,1] )
        # Make one MC time step
        Z_i = tf.reshape(Z_incr[:,step], shape = [-1,1] )
        print('\rX and hedge -step: {:4} of {:4} , t = {:4.4}   '.format(step+1, N_t_final, taux[step+1]), end='')
        Delta_X = (- 0.5* tf.square(alpha) * tf.square( sig(taux[step], X)) * delta_t + alpha * sig(taux[step], X) * Z_i)

        if just_ops:
            for matiter in range(mat_counter+1,num_mats):
                StochInt[matiter] = StochInt[matiter] + (h2(taux[step], S0*tf.exp(X), alpha * tf.abs(sig(taux[step], X)) , matiter ) )* S0*(tf.exp(X+Delta_X)-tf.exp(X))

                StochInt2[matiter] = StochInt[matiter] + (h2(taux[step], S0*tf.exp(X), alpha * tf.abs(sig(taux[step], X)) , matiter ) - 1 )* S0*(tf.exp(X+Delta_X)-tf.exp(X)) 
        else:
            for matiter in range(mat_counter+1,num_mats):
                StochInt[matiter] = StochInt[matiter] + (h2(taux[step], S0*tf.exp(X), alpha * tf.abs(sig(taux[step], X)) , matiter ) - call_putt_hedge_vec[matiter] )* S0*(tf.exp(X+Delta_X)-tf.exp(X) ) 


        X = X + Delta_X

        # CHECK IF now is a maturity
        if (step+1) in N_t_step:
            mat_counter += 1
            prices_data_slice_tf = tf.constant(pM_data[mat_counter],
                                shape=[1,lenS[mat_counter]], dtype=tf.float32)

            #prices_data_slice_tf_batched = tf.tile(prices_data_slice_tf, [Batchsize_tf,1])
            # Next we compute the values we want to store in lists and return
            F = S0*tf.exp(X)
            Fmat = tf.tile(F,(1,lenS[mat_counter]))
            Kmat = tf.tile(K_tf[mat_counter], (Batchsize_tf,1))
            
            
            if just_ops:
                payMat1 = Fmat - Kmat
                payMat2 = - payMat1

                payMat1  = tf.maximum(payMat1, 0.)
                payMat2  = tf.maximum(payMat2, 0.)

                price_sliceCall = tf.reduce_mean( payMat1 - StochInt[mat_counter], axis=0)
                price_slicePut  = tf.reduce_mean( payMat2 - StochInt2[mat_counter], axis = 0)

                prices_tfCall.append(price_sliceCall)
                prices_tfPut.append( price_slicePut )
                
            else:
                payMat1 = Fmat[:,:atmInd[mat_counter]+1] - Kmat[:,:atmInd[mat_counter]+1  ]
                payMat2 = Kmat[:,atmInd[mat_counter]+1 : ] - Fmat[:,atmInd[mat_counter]+1 : ]
                payMat  = tf.concat([payMat1,payMat2] , axis =1)
                payMat  = tf.maximum(payMat, 0.)

                price_slice = payMat - StochInt[mat_counter]
                price_slice = tf.reduce_mean(price_slice, axis = 0)
                prices_tf.append(price_slice)

                hedge_err =  payMat - tf.stop_gradient(StochInt[mat_counter] )
                hedge_err = tf.reduce_mean(hedge_err, axis = 0, keep_dims=True)
                hedge_err = (hedge_err - prices_data_slice_tf)
                                
                cost = weights_list[mat_counter] * tf.square(  hedge_err )            
                cost = tf.reduce_sum(cost)
                cost_tf.append(cost)

    if just_ops:
        return(prices_tfCall, prices_tfPut)
    else:
        return(cost_tf, prices_tf)





