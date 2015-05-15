# This includes some very simple cdfs and pdf, used for fitting probability
# distributions

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 10:38:44 2015

@author: smudd
"""

import numpy as np

def weibull_cdf(x,lambda_wb,k):
    return 1-np.exp(np.power(-(x/lambda_wb),k))

def weibull_pdf(x,lambda_wb,k):
    return (k/lambda_wb)*np.power((x/lambda_wb),k-1)*np.exp(np.power((-x/lambda_wb),k))

#def lognormal_cdf(x,mu,sigma):
#    return 0.5+0.4*math.erf( (np.log(x)-mu)/np.sqrt(2)*sigma)

def lognormal_pdf(x,mu,sigma):
    return (1/(x*sigma*np.sqrt(2*np.pi)))*np.exp( -((np.log(x)-mu)*(np.log(x)-mu))/(2*sigma*sigma) )

def exponential_cdf(x,lambda_exp):
    return 1-np.exp(-x*lambda_exp)

def exponential_pdf(x,lambda_exp):
    return lambda_exp*np.exp(-x*lambda_exp)
    
def exponential_fxn(x,lambda_exp):
    return np.exp(-x*lambda_exp)

def RMSE(data,prediction):
    sum_err = 0
    n = len(data)
    for d,p in zip(data,prediction):
        sum_err = (p-d)*(p-d)
        
    RMSE = np.sqrt(sum_err/n)
    return RMSE
         