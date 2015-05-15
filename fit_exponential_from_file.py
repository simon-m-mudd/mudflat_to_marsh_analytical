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


def fit_exponential_from_file(filename, threshold_velocity):
    # State the filename
    #filename = 'c:\\Users\\smudd\\Documents\\Papers\\Tidal_paper_padova\\Wind_Data\\s_andrea_ok.txt'    
    
    # First load the wind data file. This data just has a wind speed on every line
    wind_data = np.loadtxt(filename)
    
    # get the maximum windspeed
    max_wind = wind_data.max()
    print "Maximum windspeed is: "+str(max_wind)
    
    # number of bins in the histogram
    bins=20
    
    # now build a histogram from these data
    hist, bin_edges = np.histogram(wind_data, bins, (0,20), density=True) # Calculate histogram
    x_hist = bin_edges[1:] # Take the upper edges of the bins
    print "bin edges are: " + str(x_hist)
    #np.histogram()

    #now get the pdf    
    y_pdf = hist/hist.cumsum().max()
    
    # now get the cdf
    y_cdf = hist.cumsum()/hist.cumsum().max()  # Normalise the cumulative sum
    
    # now get a truncated version of the data, cutting off any values less
    # than the threshold   
    truncated_x = []
    truncated_pdf = []
    truncated_cdf = []    
    for x,pdf_value,cdf_value in zip(x_hist,y_pdf,y_cdf):
        if x >= threshold_velocity:
            truncated_x.append(x)
            truncated_pdf.append(pdf_value)
            truncated_cdf.append(cdf_value)
            
    # turn the data into numpy arrays, necessary for the fitting routine below
    trunc_x = np.asarray(truncated_x)
    trunc_pdf = np.asarray(truncated_pdf)

    # fit the pdf for the thresholded data. 
    popt_exp_pdf, pcov_exp_pdf = curve_fit(sd.exponential_fxn, trunc_x, trunc_pdf)
     
    # get the fitted pdf
    print "The fitted decay coefficient is: "
    print popt_exp_pdf  
    y_exp_fit = sd.exponential_fxn(trunc_x,popt_exp_pdf)        
    
    
    RMSE =  sd.RMSE(truncated_pdf,y_exp_fit)
    print "The RMSE is: " + str(RMSE)
    
    return x_hist,y_pdf,trunc_x,trunc_pdf,popt_exp_pdf,y_exp_fit,RMSE


    

# This function plots the results from the fitting
def plot_exponential_fit(x_hist,y_pdf,trunc_x,trunc_pdf,y_exp_fit,popt_exp_pdf,threshold_velocity,RMSE):      

    ## PLOT THE GRAPH
    plt.figure(figsize=(12,6))
    
    # The first subplot is the data, and the truncated data
    ax1=plt.subplot(211)
    ax1.plot(trunc_x,trunc_pdf,'ro',label='truncated pdf')    
    ax1.plot(x_hist,y_pdf,'k.',label='raw pdf')   
    plt.xlabel('x')
    plt.ylabel('pdf')
    ax1.legend(loc='upper right')
    plt.title('velocity: ' + str(threshold_velocity))
    
    # The second subplot plots the data vs the fit
    ax2=plt.subplot(212)
    ax2.plot(trunc_x,trunc_pdf,'ro',label='truncated pdf')
    ax2.plot(trunc_x,y_exp_fit,'gx',label='fit pdf')
    #plt.xlim(0,150)
    plt.xlabel('x')
    plt.ylabel('pdf')
    ax2.legend(loc='upper right')
    plt.title('RMSE: ' + str(RMSE))
    
    plot_fname = "Fit_plot_v"+str(threshold_velocity)+".png"
    plt.savefig(plot_fname, format='png')
    plt.clf()

if __name__ == "__main__":
    #fit_weibull_from_file(sys.argv[1])
    filename = 'c:\\Users\\smudd\\Documents\\Papers\\Tidal_paper_padova\\Wind_Data\\s_leonardo_ok.txt'
    threshold_velocity = 5
    for i in range(1,6):
        threshold_velocity = i
        print "threshold velocity is: " + str(threshold_velocity)
        x_hist,y_pdf,trunc_x,trunc_pdf,popt_exp_pdf,y_exp_fit,RMSE = fit_exponential_from_file(filename,threshold_velocity) 
        plot_exponential_fit(x_hist,y_pdf,trunc_x,trunc_pdf,y_exp_fit,popt_exp_pdf,threshold_velocity,RMSE)
    

    