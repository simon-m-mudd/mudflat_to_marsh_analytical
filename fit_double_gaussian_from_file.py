# This function is used for assimilating erosion data and fitting it with 
# a double gaussian function
"""
Created on Tue Mar 31 10:38:44 2015

@author: smudd
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit # Import the curve fitting module
import simple_distributions as sd
import peak_detect as pd


def fit_double_gaussian_from_file(filename):
    # State the filename
    #filename = 'c:\\Users\\smudd\\Documents\\Papers\\Tidal_paper_padova\\Wind_Data\\s_andrea_ok.txt'    
    
    # First load the erosion data file. This data just has an erosion rate 
    # and a depth on every line
    erosion_data = np.loadtxt(filename)
    depth = erosion_data[:,0]
    erate = erosion_data[:,1]
    
    # Find some peaks
    _max, _min = pd.peakdetect(erate, depth, 20, 0.10)
    xm = [p[0] for p in _max]
    ym = [p[1] for p in _max]

    # default initial guess
    midpoint = 0
    sig = 0.1
    amplitude = 1
    spacing = 1
    
    # now construct the initial guess from these data
    n_peaks = len(xm)
    if n_peaks == 1:
        amplitude = ym[0]
        spacing = 0
        midpoint = xm[0]
    else:
        amplitude = max(ym)
        spacing = xm[1]-xm[0]
        midpoint = 0.5*(xm[1]+xm[0])
    
    # new initial guess    
    initial_guess = [midpoint,sig,amplitude,spacing]
    
    print "Initial guess: "
    print "midpoint: " + str(midpoint)
    print "sigma: " + str(sig)
    print "amplitude: " + str(amplitude)
    print "spacing: " + str(spacing)
        


    # try to get some reasonable inital guesses based on the data
    
    
    # now fit the erosion data to a double gaussian
    popt_dg, pcov_dg = curve_fit(sd.double_gaussian, depth, erate,initial_guess)
    #popt_dg, pcov_dg = curve_fit(sd.double_gaussian, depth, erate)   
   
    # get the fitted pdf
    print "The fitted components are: "
    print popt_dg 
    y_dg_fit = sd.double_gaussian(depth,popt_dg[0],popt_dg[1],popt_dg[2],popt_dg[3])        
    
    
    RMSE =  sd.RMSE(erate,y_dg_fit)
    print "The RMSE is: " + str(RMSE)
    
    return depth,erate,popt_dg,y_dg_fit,RMSE

def generate_test_data(midpoint,sig,amplitude,spacing):
    depth = np.linspace(-2, 2, num=201)
    double_gauss = sd.double_gaussian(depth,midpoint,sig,amplitude,spacing)
    
    test_data = zip(depth,double_gauss)
    #print "The data to go out is: "
    #print test_data
    np.savetxt('test_dg.txt', test_data, fmt='%2.6f')    
    
    return depth,double_gauss 
    

# This function plots the results from the fitting
def plot_double_gaussian_fit(depth,erate,popt_dg,y_dg_fit,RMSE):      

    ## PLOT THE GRAPH
    plt.figure(figsize=(12,6))
    
    # The first subplot is the data, and the truncated data
    ax1=plt.subplot(211)
    ax1.plot(depth,erate,'ro',label='data')    
    ax1.plot(depth,y_dg_fit,'k.',label='fit data')   
    plt.xlabel('depth')
    plt.ylabel('data and fit')
    ax1.legend(loc='upper right')
    #plt.title('velocity: ' + str(threshold_velocity))
    
    # The second subplot plots the data vs the fit
    #ax2=plt.subplot(212)
    #ax2.plot(trunc_x,trunc_pdf,'ro',label='truncated pdf')
    #ax2.plot(trunc_x,y_exp_fit,'gx',label='fit pdf')
    #plt.xlabel('x')
    #plt.ylabel('pdf')
    #ax2.legend(loc='upper right')
    #plt.title('RMSE: ' + str(RMSE))
    
    plot_fname = "Fit_plot_dg.png"
    plt.savefig(plot_fname, format='png')
    plt.clf()




if __name__ == "__main__":
    #fit_weibull_from_file(sys.argv[1])
    filename = 'c:\\Users\\smudd\\Documents\\Papers\\Tidal_paper_padova\\Python_code\\test_dg.txt'
    
    midpoint = -0.1
    sig = 0.1
    amplitude = 2.6
    spacing = 0.87
    generate_test_data(midpoint,sig,amplitude,spacing)
    
    depth,erate,popt_dg,y_dg_fit,RMSE = fit_double_gaussian_from_file(filename)
    plot_double_gaussian_fit(depth,erate,popt_dg,y_dg_fit,RMSE)
    

    

    