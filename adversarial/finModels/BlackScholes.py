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

"""BlackScholes.py:
This file implements some core Black Scholes related functionality for conversion
of prices and implied volas for several strikes and maturities as used in our
fin_surface class using the library: py_vollib
"""

import numpy as np
from scipy.stats import norm
from numpy import sqrt, log # pylint: disable=E0611
from py_vollib.black_scholes.implied_volatility import implied_volatility as implVola


def BS_price(vola, Strike, maturity, spot, OptionType="call"):
    # Strike and maturity must be single here
    
    if type(Strike) != np.float64:
        raise ValueError("The Strike val is not correct")
    elif type(maturity) != np.float64:
        raise ValueError("The maturity val is not correct")

    if OptionType == "call":
        
        d1 = (log(spot/Strike) + 0.5 * vola**2 * maturity)/(vola * sqrt(maturity) )
        d2 = d1 -vola*sqrt(maturity)
        
        phi_d1 = norm.cdf(  d1 )
        phi_d2 = norm.cdf(  d2 )
        
        price = spot * phi_d1 - Strike * phi_d2

    elif OptionType == "put":
        d1 = (log(spot/Strike) + 0.5 * vola**2 * maturity)/(vola * sqrt(maturity) )
        d2 = d1 -vola*sqrt(maturity)
        
        phi_m_d1 = norm.cdf(  -d1 )
        phi_m_d2 = norm.cdf(  -d2 )
        
        price = Strike * phi_m_d2  - spot * phi_m_d1 

    else:
        raise Exception("This option type is not implemented yet")
    
    return price



def get_impl_volaSurface(priceMatrix, strikes, maturities, Spot):
    # prices need to be Call prices, not put or other!!!
    len_strikes = len(strikes)
    maturities = np.reshape(maturities,[-1])
    len_mats    = len(maturities)
    prices  = np.reshape(priceMatrix, [len_strikes, len_mats])
    surface = np.zeros(prices.shape )

    iter_K, iter_tau = 0, 0
    for K in strikes:
        for tau in maturities:   

            try:
                aux =  implVola( prices[iter_K, iter_tau],
                                 Spot, K, tau, 0, 'c')

            except:
                aux = 0

            surface[iter_K, iter_tau] = aux
            iter_tau += 1
        iter_K += 1
        iter_tau = 0

    return(surface)




if __name__ == "__main__":
    pass