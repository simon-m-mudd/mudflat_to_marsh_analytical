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


def fit_double_gaussian_from_file(filename):
    # State the filename
    #filename = 'c:\\Users\\smudd\\Documents\\Papers\\Tidal_paper_padova\\Wind_Data\\s_andrea_ok.txt'    
    
    # First load the erosion data file. This data just has an erosion rate 
    # and a depth on every line
    erosion_data = np.loadtxt(filename)
    depth = erosion_data[:,0]
    erate = erosion_data[:,1]
    
    print "Depth is: " 
    print depth
    print "Erosion rate is: "
    print erate
    # initial guesses
    initial_guess = [0,0.001,1,1]    
    
    # now fit the erosion data to a double gaussian
    popt_dg, pcov_dg = curve_fit(sd.double_gaussian, depth, erate,initial_guess)
     
    # get the fitted pdf
    print "The fitted components are: "
    print popt_dg 
    y_dg_fit = sd.double_gaussian(depth,popt_dg[0],popt_dg[1],popt_dg[2],popt_dg[3])        
    
    
    RMSE =  sd.RMSE(erate,y_dg_fit)
    print "The RMSE is: " + str(RMSE)
    
    return depth,erate,popt_dg,y_dg_fit,RMSE

def generate_test_data(midpoint,sig,amplitude,spacing):
    depth = np.linspace(-1, 1, num=21)
    double_gauss = sd.double_gaussian(depth,midpoint,sig,amplitude,spacing)
    
    test_data = zip(depth,double_gauss)
    print "The data to go out is: "
    print test_data
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
    
    midpoint = 0
    sig = 0.2
    amplitude = 2
    spacing = 0.5
    generate_test_data(midpoint,sig,amplitude,spacing)
    
    depth,erate,popt_dg,y_dg_fit,RMSE = fit_double_gaussian_from_file(filename)
    plot_double_gaussian_fit(depth,erate,popt_dg,y_dg_fit,RMSE)
    

    

    