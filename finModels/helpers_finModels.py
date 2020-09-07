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

"""helpers_finModels.py:
This file implements some basic option and implied vola utility as well as a 
helpfull way of storing calibration results.

ToDo: Documentations
"""


import numpy as np
from finModels.BlackScholes import BS_price
from py_vollib.black_scholes.implied_volatility import implied_volatility as implVola



class fin_option():
    '''Class descroption (ToDo) '''

    def __init__(self, maturities, Strikes):
        # Check correct inputs
        if type(maturities)==list:
            mat = np.array(maturities)
        else:
            mat = maturities

        assert len(mat.shape)    == 1
        assert type(mat)         == np.ndarray
        assert mat.dtype == np.float


        assert type(Strikes)            == list
        # We allow for 32 bit values here
        for i in Strikes: assert isinstance(i, np.ndarray  )

        self.T = mat
        self.K = Strikes
        assert len(self.T) == len(self.K)
        
    def __call__(self):
        return([self.T, self.K] )

    def __getitem__(self, key):
        return [self.T[key], self.K[key]]
    

    @property
    def t_grid_nn(self):
        return(
        np.append(0, self.T[:-1])
        )

    @property
    def nT(self):
        return(len(self.T) )

    @property
    def nK(self):
        nK = [len(i) for i in self.K ]
        return nK

    def log_m_mat(self, iter_mat, S0):
        val = np.log(self.K[iter_mat]/S0)
        return (val)
        
    def log_m_list(self, S0):
        return(
            [self.log_m_mat(it_T, S0) for it_T in range(len(self.T))  ]
        )

    def N_t_step(self, para_MC):
        maturities = self.T
        delta_t    = para_MC['delta_t']

        for t in maturities:
            if not np.isclose(t/delta_t, int(np.rint(t/delta_t )), rtol=1e-6):
                raise('delta_t is not true fraction of some maturity')
        
        N_t_step = [int(np.rint(t_in_mat / delta_t )) for t_in_mat in maturities]
        if np.isclose(N_t_step[0], 0.0):
            raise Exception('The smallest time to maturities are too small for given delta_t')
        return (N_t_step)

    

class fin_surface():
    # Implements a financial surface object that handles conversions
    # etc and stores in list format, each entry -> one mat

    def __init__(self,*, mats, strikes, spot, version ):
        self.T_K = fin_option(mats, strikes)
        self.spot = spot
        self.logSpot = np.log(spot)
        self._iV = None
        self._prices = None
        self.version = version
        self.calibrated_SLV_iV = None 
        self.leveragegraph     = None
        self.sabr_cali = {}
        self.lsv_cali  = {}

    
    def find_atm_index(self,mat):
        return(  np.where(self.log_m_scale_list[mat] <= 0)[0][-1]) 


    def store_lsv_cali(self, pMlist, x_sig_list, cali_time, sigma_vals, trainSteps, iVmodel, iVdata):
        self.lsv_cali['pM'] = pMlist 
        self.lsv_cali['xsig'] = x_sig_list
        self.lsv_cali['cali_time'] = cali_time
        self.lsv_cali['sig_of_xsig'] = sigma_vals
        self.lsv_cali['iVdata'] = iVdata
        self.lsv_cali['iVmodel'] = iVmodel

    def store_lsv_cali_mixed(self, pMlist, x_sig_list, cali_time, sigma_vals, trainSteps, iVmodel, iVdata):
        self.lsv_cali['pM'] = pMlist 
        self.lsv_cali['xsig'] = x_sig_list
        self.lsv_cali['cali_time'] = cali_time
        self.lsv_cali['sig_of_xsig'] = sigma_vals
        self.lsv_cali['iVdata'] = iVdata
        self.lsv_cali['iVmodel'] = iVmodel
    
    

    def N_t_step(self, para_MC):
        return(self.T_K.N_t_step(para_MC))

    def check_data(self, iV_prices, T_K):
        # Check format etc
        assert len(iV_prices) == T_K.nT
        for i in range(T_K.nT):
            assert isinstance(iV_prices[i], np.ndarray)
            assert isinstance(iV_prices[i][0], (np.float32,np.float64 ))
            assert iV_prices[i].shape == T_K.K[i].shape

    def feed_iV(self,*, iV):
        self.check_data(iV, self.T_K)
        self._iV = iV
    def feed_prices(self,*,  prices):
        self.check_data(prices, self.T_K)
        self._prices = prices
    def feed_sabr_cali(self,*,rho,alpha0,nu, prices):
        self.sabr_cali['rho'] = rho
        self.sabr_cali['alpha_0'] = alpha0
        self.sabr_cali['nu'] = nu
        self.sabr_cali['prices'] = prices

    @property
    def sabr_rho(self):
        return(self.sabr_cali['rho'])
    @property
    def sabr_alpha0(self):
        return(self.sabr_cali['alpha_0'])
    @property
    def sabr_nu(self):
        return(self.sabr_cali['nu'])
    
    @property
    def nT(self):
        return(self.T_K.nT)
    @property
    def nK(self):
        return(self.T_K.nK)
        
    @property
    def prices(self):
        if self._prices == None:
            raise ValueError('Prices have not been computed/filled yet')
        return(self._prices)
    @property
    def strikes(self):
        return(self.T_K.K)
    
    @property
    def maturities(self):
        return(self.T_K.T)

    @property
    def iV(self):
        if self._iV == None:
            raise ValueError('IV has not been computed yet')
        return(self._iV)

    def clear_iV(self):
        self._iV = None
    
    # Conversion function for all slices
    def convert_p_2_iV(self,*,prices):
        iV = []
        # Check if prices math the mats and strikes dims
        assert len(prices) == self.nT
        for iter, p in enumerate(prices):
            assert len(p) == self.nK[iter]
        
        for iter_mat, p in enumerate(prices):
            iV.append( self.get_model_iv_slice(model_price=p, slice=iter_mat))
        return(iV)

    # This is used by the wrapper convert_p_2_iV, computes per slice
    def get_model_iv_slice(self,*, model_price, slice):
        mat, K_slice = self.T_K[slice]
    
        iV_m_slice = np.zeros(K_slice.shape)
        for iter, K in enumerate(K_slice):
            iV_m_slice[iter] = implVola(model_price[iter], self.spot, K, mat, 0.0, 'c')
        return(iV_m_slice)

    def get_model_iv_slice_mixed(self,*, model_price, slice):
        atmInd = self.find_atm_index(slice)

        mat, K_slice = self.T_K[slice]

        iV_m_slice = np.zeros(K_slice.shape)
        for iter, K in enumerate(K_slice):
            if iter <= atmInd:
                iV_m_slice[iter] = implVola(model_price[iter], self.spot, K, mat, 0.0, 'c')
            else:
                iV_m_slice[iter] = implVola(model_price[iter], self.spot, K, mat, 0.0, 'p')
        return(iV_m_slice)

    def serve_mixed_prices(self):
        prices = [np.zeros(iVmat.shape) for iVmat in self.iV ]
        for i_t, t, K_slice in zip(range(self.T_K.nT), *self.T_K()):
                atmInd = self.find_atm_index(i_t)
                for i_K, K in enumerate(K_slice):
                    if i_K <= atmInd:
                        prices[i_t][i_K] = BS_price(self.iV[i_t][i_K], K, t, self.spot,
                                                        OptionType='call')
                    else:
                        prices[i_t][i_K] = BS_price(self.iV[i_t][i_K], K, t, self.spot,
                                                        OptionType='put')
        return(prices)

    def convert(self,*, direction):
        if direction=='iV->price':
            self.prices = [np.zeros(iVmat.shape) for iVmat in self.iV ]
            for i_t, t, K_slice in zip(range(self.T_K.nT), *self.T_K()):
                for i_K, K in enumerate(K_slice):
                    self.prices[i_t][i_K] = BS_price(self.iV[i_t][i_K], K, t, self.spot,
                                                    OptionType='call')

        elif direction=='price->iV' or direction=='price_sabr->iV_sabr':

            if direction=='price->iV':
                prices_aux = self.prices
            elif direction == 'price_sabr->iV_sabr':
                prices_aux = self.sabr_cali['prices']
            
            iV_aux = [np.zeros(Pmat.shape) for Pmat in prices_aux ]
        
            for i_t, t, K_slice in zip(range(self.T_K.nT), *self.T_K()):
                for i_K, K in enumerate(K_slice):
                    iV_aux[i_t][i_K] = implVola(prices_aux[i_t][i_K], self.spot,
                                                K, t, 0.0, 'c')
            if direction=='price->iV':
                self._iV = iV_aux 
            elif direction=='price_sabr->iV_sabr':
                self.sabr_cali['iV'] = iV_aux
            else:
                raise('This should not happen')
        else:
            raise ValueError("unsupported direction")    

    def get_iV_lsv_m_slice(self,*, slice):
        iV = self.lsv_cali['iV'][slice]
        log_m = self.log_m_scale_list[slice]
        return(log_m, iV)


    @property
    def log_m_scale_list(self):
        return(self.T_K.log_m_list(self.spot))

    def plotIV(self):
        pass

    
def find_mat_positions(taux, maturities):
    
    lenMat = len(maturities)
    positions = np.zeros( lenMat, dtype=int)

    for i in np.arange(lenMat):
        tau = maturities[i]

        pos_low = np.where( taux <= tau)[0][-1]
        pos_high = pos_low +1
        if pos_low == (len(taux)-1):
            positions[i] = pos_low
            continue
        
        elif (np.abs( tau - taux[pos_low] ) < np.abs( tau - taux[pos_high] )):
            positions[i] = pos_low
            continue
        else:
            positions[i] = pos_high
            continue
    
    return(positions)



if __name__ == "__main__":
    pass    