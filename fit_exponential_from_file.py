# This function is used for assimilating wind data and creating a 
# function based on the exponentail probability function. 
# The function is fit beyond a tail, so that the fit function is
# _NOT_ a pdf, since it does not integrate to one. 
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 10:38:44 2015

@author: smudd
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit # Import the curve fitting module
import simple_distributions as sd


def fit_exponential_from_file(filename):
    # State the filename
    #filename = 'c:\\Users\\smudd\\Documents\\Papers\\Tidal_paper_padova\\Wind_Data\\s_andrea_ok.txt'    
    
    # First load the wind data file. This data just has a wind speed on every line
    wind_data = np.loadtxt(filename)
    
    # make a second dataset of wind values below a threshold
    threshold = 5.0
    wind_data_thresh = [x for x in wind_data if x >= threshold]

    # get an unshifted dataset
    wind_data_thresh_unshift = wind_data_thresh
    
    # now shift the thresholded data
    wind_data_thresh[:] = [x - 5 for x in wind_data_thresh] 

    
    max_wind = wind_data.max()
    print "Maximum windspeed is: "+str(max_wind)
    
    # number of bins in the histogram
    bins=50
    
    # now build a histogram from these data
    hist, bin_edges = np.histogram(wind_data, bins, density=True) # Calculate histogram
    x_hist = bin_edges[1:] # Take the upper edges of the bins

    hist_thresh, bin_edges_thresh = np.histogram(wind_data_thresh, bins, density=True) # Calculate histogram
    x_hist_thresh = bin_edges_thresh[1:] # Take the upper edges of the bins
    
    # now do it for the unshifted data    
    hist_thresh_us, bin_edges_thresh_us = np.histogram(wind_data_thresh_unshift, bins, density=True) # Calculate histogram
    x_hist_thresh_us = bin_edges_thresh_us[1:] # Take the upper edges of the bins
    
    #now get the pdf    
    y_pdf = hist/hist.cumsum().max()
    y_pdf_thresh = hist_thresh/hist_thresh.cumsum().max()
    y_pdf_thresh_us = hist_thresh_us/hist_thresh_us.cumsum().max()
    
    # now get the cdf
    y_cdf = hist.cumsum()/hist.cumsum().max()  # Normalise the cumulative sum
    y_cdf_thresh = hist_thresh.cumsum()/hist_thresh.cumsum().max()  
    y_cdf_thresh_us = hist_thresh_us.cumsum()/hist_thresh_us.cumsum().max() 


    # fit the cdf for both the thresholded and the raw data. 
    # the raw data gets fit with a weibull, the thresholded with and exponential
    #popt_wb, pcov_wb = curve_fit(weibull_pdf, x_hist, y_pdf, p0 = [1,5] )
    popt_exp, pcov_exp = curve_fit(exponential_cdf, x_hist_thresh, y_cdf_thresh)
    popt_ln, pcov_exp = curve_fit(lognormal_pdf, x_hist, y_pdf)
    
    # also fit the data using the weibul but only for thresholded data
    popt_lnt, pcov_lnt = curve_fit(lognormal_pdf, x_hist_thresh_us, y_cdf_thresh_us)
    
    # get the fitted cdfs
    #print popt_wb
    print popt_exp
    print popt_ln
    print popt_lnt
    #y_wb_fit = weibull_cdf(x_hist,20,2)    
    y_exp_fit = exponential_cdf(x_hist_thresh,popt_exp)        
    y_ln_fit = lognormal_pdf(x_hist,popt_ln[0],popt_ln[1])   
    y_lnt_fit = lognormal_pdf(x_hist_thresh_us,popt_lnt[0],popt_lnt[1])   
    
    
    ## PLOT THE GRAPH
    plt.figure(figsize=(12,6))
    ax1=plt.subplot(221)
    ax1.plot(x_hist,y_cdf,label='cdf')
    #ax1.plot(x_hist,y_ln_fit,label='cdf, fit')
    #plt.xlim(0,150)
    plt.xlabel('x')
    plt.ylabel('cdf')
    ax1.legend(loc='lower right')
    
    #ax2=plt.subplot(223)
    #ax2.plot(x_hist_thresh,y_cdf_thresh,label='cdf, thresholded')
    #ax2.plot(x_hist_thresh,y_exp_fit,label='cdf, fit')
    #plt.xlim(0,150)
    #plt.xlabel('x')
    #plt.ylabel('cdf')
    #ax2.legend(loc='lower right')    
    #
    ax2=plt.subplot(223)
    ax2.plot(x_hist_thresh_us,y_cdf_thresh_us,label='cdf, thresholded')
    ax2.plot(x_hist_thresh_us,y_lnt_fit,label='cdf, fit')
    plt.xlim(0,150)
    plt.xlabel('x')
    plt.ylabel('cdf')
    ax2.legend(loc='lower right')       
    
    ax3=plt.subplot(222)
    ax3.plot(x_hist, y_pdf, 'ro', label='pdf')
    ax3.plot(x_hist, y_ln_fit, 'bx', label='pdf')
    ax3.plot(x_hist_thresh_us, y_lnt_fit, 'gx', label='pdf, threshold ln')
    plt.xlabel('x')
    plt.ylabel('pdf')    
    ax3.legend(loc='upper right')
    
    ax4=plt.subplot(224)
    ax4.plot(x_hist_thresh, y_pdf_thresh, 'bo', label='pdf, thresholded')
    plt.xlabel('x')
    plt.ylabel('pdf')    
    ax4.legend(loc='upper right')
    
    #ax2.plot(x,stats.lognorm.cdf(x, shape_out, loc=0, scale=scale_out), label='Fitted distribution')
    #plt.xlim(0,150)
    #ax2.set_ylim(0,1.0)
    #plt.xlabel('x')
    #plt.ylabel('y_cdf')
    #leg=ax2.legend(loc='lower right', numpoints=1)
    #results_txt="""M_in=%.2f
    #M_out=%.2f
    #s_in=%.2f
    #s_out=%.2f""" % (M, scale_out, s, np.exp(shape_out))
    #txt=plt.text(0.97, 0.3, results_txt, transform=ax2.transAxes, horizontalalignment='right', fontsize='large')
    plt.show()    

      
    
    
if __name__ == "__main__":
    #fit_weibull_from_file(sys.argv[1])
    filename = 'c:\\Users\\smudd\\Documents\\Papers\\Tidal_paper_padova\\Wind_Data\\s_andrea_ok.txt'   
    fit_exponential_from_file(filename) 
    