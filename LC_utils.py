""" Some code I use in fitting binary star lightcurves with ellc"""

import numpy as np
import matplotlib.pyplot as plt




def WD_MR(M):
    '''calculate the radius of a zero temperature wd given a mass

    input :
    M : float or array-like
        the mass of the wd in solar units
    output : float or array-like
        the radius in solar units
    '''
    M_ch = 1.44
    M_p = 0.00057
    R = 0.0114 * (( M/M_ch )**(-2./3.)-(M/M_ch)**(2./3.))**(0.5) * (1+3.5*(M/M_p)**(-2./3.)+(M/M_p)**(-1))**(-2./3.)
    return R



def calc_binarypars(p,r1,r2,i,q,a):
    """Convert lightcurve binary parameters to physical parameters

    input: 
        p:  period in days
        r1: scale radius
        r1: scale radius
        i: inclination in degrees
        q: mass ratio M2/M1
        a: sma in solar radii
         
    output:
        M1: mass 1 in solar units
        M2: mass 2 in solar units
        R1: radius 1 in solar units
        R2: radius 2 in solar units
        logg1: surface gravity in log(cgs)
        logg2: surface gravity in log(cgs)
        K1: radial velocicty amplitude in km/s  
        K2: radial velocicty amplitude in km/s        
    """


    # calculate some numbers
    M_sun = 1.9891*10**30
    R_sun = 695800000.0
    G = 6.673e-11
    
    # calc
    Mtot = (a*R_sun)**3*(2.*np.pi/(p*3600*24))**2/G/M_sun
    M1 = 1./(1+q)*Mtot
    M2 = q/(1+q)*Mtot
    R1 = a*r1
    R2 = a*r2

    # calculate RVs
    K = 2*np.pi*a*R_sun / (p*3600*24)/1000 * np.sin(np.radians(i))
    K1 = q/(1+q)*K
    K2 = 1./(1+q)*K
    
    # loggs ou
    logg1 = np.log10(G*M1*M_sun/(R1*R_sun)**2)+2
    logg2 = np.log10(G*M2*M_sun/(R2*R_sun)**2)+2

    return M1,M2,R1,R2,logg1,logg2,K1,K2



def plotall(allpars,alldata,allmodels,grid='default',fold=False,Nbins=False):

    for n,(data,model) in enumerate(zip(alldata,allmodels)):
        pars = np.r_[allpars[:7],allpars[7+n*10:7+(n+1)*10]]
        pars = pars[:-1]
        plot(pars,data,model,grid=grid,fold=fold,Nbins=Nbins)

    plt.show()
    return None




def plotLC(pars,data,model,grid='default',fold=False,Nbins=False):
    """ A function to plot a single lightcurve

    input
        pars: an array of input variables
        data: the lightcurve in [t,y,dy]
        model: an ellc model function which takes as input (t,pars,grid)
        grid: the ellc-surface grid density
        fold: fold the lightcurve on the period (period is assumed to be the first parameter)
        Nbins: show binned data and residual

    output:
        nothing, it shows a plot

    """
    # double panel figure 
    # Three subplots sharing both x/y axes
    f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

    offset = np.floor(np.min(data[:,0]))


    p = pars[0]
    t0 = pars[1]
    m = model(data[:,0],pars,grid) 

    if fold:

        phases = (data[:,0]-t0)/p%1
        p_idx = np.argsort(phases)
        ax1.errorbar(phases[p_idx],data[p_idx,1],data[p_idx,2],fmt='k.')
        ax1.plot(phases[p_idx],m[p_idx],zorder=3,c='C3')
        ax2.errorbar(phases[p_idx],data[p_idx,1]-m[p_idx],data[p_idx,2],fmt='k.')    
        if Nbins:
            bins = np.linspace(0,1,Nbins+1)
            digitized = np.digitize(phases, bins)
            res_means = np.array([np.average(data[digitized==i,1]-m[digitized==i]) for i in range(1, len(bins))])
            plt.plot((bins[:-1]+bins[1:])/2,res_means,'bx',zorder=3)

    else:
        ax1.errorbar(data[:,0]-offset,data[:,1],data[:,2],fmt='k.')
        ax1.plot(data[:,0]-offset,m,zorder=3,c='C3')
        ax2.errorbar(data[:,0]-offset,data[:,1]-m,data[:,2],fmt='k.')
        if Nbins:
            bins = np.linspace(np.min(data[:,0]),np.max(data[:,0]),Nbins+1)
            digitized = np.digitize(data[:,0], bins)
            t_means = np.array([np.average(data[digitized==i,0]) for i in range(1, len(bins))])
            res_means = np.array([np.average(data[digitized==i,1]-m[digitized==i]) for i in range(1, len(bins))])

            ph_means = (t_means-t0)/p%1
            ax2.scatter(t_means-offset,res_means,c=ph_means,zorder=3,marker='o')

    if fold:
        plt.xlabel('Phase' %offset)
    else: 
        plt.xlabel('Time - %d (BJD)' %offset)

    ax1.set_ylabel('flux ratio')
    ax2.set_ylabel('residual')

    
    if fold:
        pass
    else:
        plt.xlim(np.min(data[:,0]-offset),np.max(data[:,0]-offset))

    chi2 = np.sum(((data[:,1]-m)/data[:,2])**2)
    plt.suptitle( "Chi2/N: %d/%d" %(chi2,np.size(data[:,1])))


