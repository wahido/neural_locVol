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

"""neural_nets_hedging.py:
This file implements the networks for training the leverage function,
the Black Scholes hedges and the ground truth assumption.

ToDo: Add class descriptions
"""

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class sigma_SLV_true(object):
    '''TODO'''

    def __init__(self):
        self.initialized = False
        self.regular1    = 0.01

    def setconstants(self,*, p0, p1, sig0, sig1, sig2,
                             gamma1, gamma2, beta1, beta2, kappa, lam1, lam2,
                             eps_t, const_else_factor,
                             version):
        self.p = np.zeros([1,3])
        self.p[:] = p0, p1, (1 - p1 - p0)

        self.sig  = np.zeros([1,3])
        self.sig[:] = sig0, sig1, sig2

        self.gamma1 = gamma1
        self.gamma2 = gamma2 
        self.beta1  = beta1 
        self.beta2  = beta2 
        self.kappa  = kappa 
        self.lam1   = lam1 
        self.lam2   = lam2 
        self.eps_t  = eps_t
        self.const_else_factor = const_else_factor

        self.initialized = True
        self.version     = version
        self.plot_func()
    
    def __call__(self, tinput, x):
        
        if tinput == 0.0:
          t = 0.01
        else:
            t = tinput

        val1 =    (self.p * self.sig)  * tf.exp((-tf.square(x) / ( 2.0 * t * self.sig**2 ) ) - t * self.sig**2 * 0.125)
        val11aux = tf.reduce_sum(val1, axis=1, keepdims=True)

        val11 = val11aux + \
            tf.minimum(
                tf.pow(
                    (self.gamma1 * tf.maximum(x - self.beta1, 0 ) + self.gamma2 * tf.maximum( - x - self.beta2, 0 )  ) ,
                    self.kappa
                ),
                self.lam1
            ) *  ((tinput <= self.eps_t) / (1+(.1*tinput) ))**self.lam2
        if t > self.eps_t:
            val11 *= self.const_else_factor

        
        val2 = (self.p / self.sig)  * tf.exp((- tf.square(x) / ( 2.0 * t * self.sig**2 ) ) - t * self.sig**2 * 0.125 )
        val22 = tf.reduce_sum(val2, axis= 1, keepdims=True)

        val = val11 /(val22 + self.regular1 )
        val = tf.abs(val)
        return(tf.minimum(val,2))

    def vola(self,tinput, x):
        return(  tf.sqrt(self(tinput, x))  )

    def _test_t(self,tinput):
        x = np.linspace(-2,2,50)
        x = np.reshape(x, [len(x),1])

        if tinput == 0.0:
          t = 0.01
        else:
            t = tinput
       
        val1 =    (self.p * self.sig)  * np.exp((-np.square(x) / ( 2.0 * t * self.sig**2 ) ) - t * self.sig**2 * 0.125 )
        val11aux = np.sum(val1, axis=1, keepdims=True)

        val11 = val11aux + \
            np.minimum(
                np.power(
                    (self.gamma1 * np.maximum(x - self.beta1, 0 ) + self.gamma2 * np.maximum( - x - self.beta2, 0 )  ) ,
                    self.kappa
                ),
                self.lam1
            ) *  ((tinput <= self.eps_t) / (1+(.1*tinput) ))**self.lam2
        if t > self.eps_t:
            val11 *= self.const_else_factor


        val2 = (self.p / self.sig)  * np.exp((- np.square(x) / ( 2.0 * t * self.sig**2 ) ) - t * self.sig**2 * 0.125 )
        val22 = np.sum(val2, axis= 1, keepdims=True)

        val = val11 /(val22 + self.regular1 )
        val = np.abs(val)
        
        return(np.minimum(val,2))

    def plot_func(self):
        time = np.linspace(0,1,100)
        sig_val = np.zeros([50,100])
        for iter, t in enumerate(time):
            sig_val[:,iter] = np.reshape(self._test_t(t),[50])



        X, T = np.meshgrid(np.linspace(-3,3,50), time)
        Z = np.zeros(shape=[100,50])
        for i in range(100):
            Z[i,:] = sig_val[:,i]

        ax = plt.axes(projection='3d')
        ax.set_xlabel('log')
        ax.set_ylabel('t')
        ax.set_zlabel('sig')
        ax.plot_wireframe(X,T,np.minimum(Z, 5))
        plt.savefig('caliRes/{}/locaVol.png'.format(self.version))
        plt.close()

class sigma_SLV(object):
    '''Short explanation of the class'''

    def __init__(self, time_grid):
        self.time_grid = time_grid
        self.LAYERLENGTH = 64
        self.NUMHIDDENLAYERS =  3

    def __call__(self, time, inputs):
        # Find the corresponding times on the grid
        #right = np.where(self.time_grid >= time)[0][0]
        left = np.where(self.time_grid <= time)[0][-1]
        result_left = self._sigma_t(left, inputs)

        return result_left
        
    def _sigma_t(self, t_grid, inputs):
        
        with tf.variable_scope("sigma_SLV_timegrid_"+str(t_grid)+"_", reuse=tf.AUTO_REUSE):
            

            nn = tf.layers.dense(inputs ,
                            self.LAYERLENGTH,
                            activation = tf.nn.leaky_relu,
                            bias_initializer=tf.zeros_initializer(),
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
                            use_bias=True)

            for i in range(self.NUMHIDDENLAYERS-1): # pylint: disable=W0612

                nn = tf.layers.dense(nn ,
                            self.LAYERLENGTH,
                            activation=tf.nn.leaky_relu,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
                            bias_initializer=tf.zeros_initializer(),
                            use_bias=True)

            
            nn = tf.layers.dense(nn ,
                            self.LAYERLENGTH,
                            activation=tf.nn.tanh,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
                            bias_initializer=tf.zeros_initializer(),
                            use_bias=True)

            
            nn = tf.layers.dense(nn, 1, activation=None, use_bias=True,
                                bias_initializer=tf.zeros_initializer(),
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.05))

            return (1 + nn) 




class strategy_LV(object):
    '''Short explanation of the class'''

    def __init__(self, strikes, maturity):
        self.strikes = strikes
        if not type(strikes) == list:
            for strike in strikes:
                if len(strike.shape) != 1:
                    raise('Strikes need to be a list of 1-D arrays')    
        
        self.mat = maturity.astype(np.float32)

        self.dist_norm =  tf.distributions.Normal(loc=0., scale=1.)

        
    def __call__(self, t, x, vola, mat_considered): 

        d1 = tf.log(x /self.strikes[mat_considered]) + 0.5 * tf.square(vola)*(self.mat[mat_considered] -t)
        d1 = d1 / (vola * tf.sqrt(self.mat[mat_considered] - t))

        hedge_BS = self.dist_norm.cdf(d1)

        return(hedge_BS )


class strategy_SLV(object):
    '''Short explanation of the class'''

    def __init__(self, identifier, strikes, maturity):
        self.identifier = identifier
        self.LAYERLENGTHSTRATEGY =  64
        self.NUMHIDDENLAYERSTRATEGY = 3

        self.strikes = strikes
        if not isinstance(strikes, list):
            raise('Strikes need to be a list')
        
        self.strikes_tf = [tf.reshape(K, (1,-1)) for K in strikes]
        
        self.mat = maturity

        self.dist_norm =  tf.distributions.Normal(loc=0., scale=1.)
    
    def __call__(self, t, x, vola, mat_considered):
        
        
        d1 = tf.log(x /self.strikes_tf[mat_considered]) + 0.5 * tf.square(vola)*(self.mat[mat_considered] -t)
        d1 = d1 / (vola * np.sqrt(self.mat[mat_considered] - t))

        hedge_BS = self.dist_norm.cdf(d1)

        return(hedge_BS )




                            
