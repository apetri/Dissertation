import sys,os

import numpy as np
from scipy.optimize import leastsq
from scipy import stats
from astropy.io import fits

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

try:
	import pixelize
	pixelize = pixelize
except ImportError:
	pixelize = None

from lenstools.pipeline.simulation import LensToolsCosmology

####################
####Book keeping####
####################

data_path = "/Users/andreapetri/Documents/Columbia/spurious_shear"
cosmo_parameters = ["Om","Ol","w","ns","si"]
cosmo_legend = {"Om":"Om0","Ol":"Ode0","w":"w0","ns":"ns","si":"sigma8"}

fiducial = LensToolsCosmology(sigma8=0.798)
variations = [ LensToolsCosmology(Om0=0.29,Ode0=0.71,sigma8=0.798), LensToolsCosmology(w0=-0.8,sigma8=0.798), LensToolsCosmology(sigma8=0.850) ]

##########
#Plotting#
##########

#E/B power spectrum plotting
def ebPlot(cmd_args,fontsize=18):

	#Load data
	hdu = fits.open(os.path.join(data_path,"pixelized","pow_avg.fit"))
	data = hdu[0].data
	lmin = hdu[0].header['LMIN']

	ee = data[0,:,:]
	bb = data[1,:,:]
	eb = data[2,:,:]

	n1,n2 = ee.shape

	new_ee = np.zeros(ee.shape)
	new_ee[0:n1/2,:] = ee[n1/2:n1,:]
	new_ee[n1/2:,:] = ee[0:n1/2,:]

	new_bb = np.zeros(ee.shape)
	new_bb[0:n1/2,:] = bb[n1/2:n1,:]
	new_bb[n1/2:,:] = bb[0:n1/2,:]

	new_eb = np.zeros(ee.shape)
	new_eb[0:n1/2,:] = eb[n1/2:n1,:]
	new_eb[n1/2:,:] = eb[0:n1/2,:]

	#Set up plot
	sns.set(font_scale=1)
	fig,ax = plt.subplots(1,3)

	ax0 = ax[0].imshow(new_ee,origin="lower",extent=[0,lmin*n2,-lmin*n1/2,lmin*n1/2],norm=LogNorm(),interpolation="nearest",cmap="viridis")
	ax[0].set_xlim(200,9000)
	ax[0].set_ylim(-9000,9000)
	ax[0].set_xlabel(r"$\ell_x$",fontsize=fontsize)
	ax[0].set_ylabel(r"$\ell_y$",fontsize=fontsize)
	ax[0].set_title(r"$S^{EE}$")
	cbar0 = plt.colorbar(ax0,ax=ax[0],orientation="horizontal") 
	ticks0 = cbar0.set_ticks([1.0e-18,1.0e-14,1.0e-10])
	ax[0].set_xticks([500,4000,8000])

	ax1 = ax[1].imshow(new_bb,origin="lower",extent=[0,lmin*n2,-lmin*n1/2,lmin*n1/2],norm=LogNorm(),interpolation="nearest",cmap="viridis")
	ax[1].set_xlim(200,9000)
	ax[1].set_ylim(-9000,9000)
	ax[1].set_xlabel(r"$\ell_x$",fontsize=fontsize)
	ax[1].set_ylabel(r"$\ell_y$",fontsize=fontsize)
	ax[1].set_title(r"$S^{BB}$")
	cbar1 = plt.colorbar(ax1,ax=ax[1],orientation="horizontal")
	ticks1 = cbar1.set_ticks([1.0e-18,1.0e-14,1.0e-10]) 
	ax[1].set_xticks([500,4000,8000])

	ax2 = ax[2].imshow(np.abs(new_eb),origin="lower",extent=[0,lmin*n2,-lmin*n1/2,lmin*n1/2],norm=LogNorm(),interpolation="nearest",cmap="viridis")
	ax[2].set_xlim(200,9000)
	ax[2].set_ylim(-9000,9000)
	ax[2].set_xlabel(r"$\ell_x$",fontsize=fontsize)
	ax[2].set_ylabel(r"$\ell_y$",fontsize=fontsize)
	ax[2].set_title(r"$\mathrm{Re}\vert S^{EB} \vert$")
	cbar2 = plt.colorbar(ax2,ax=ax[2],orientation="horizontal") 
	ticks2 = cbar2.set_ticks([1.0e-21,1.0e-16,1.0e-11])
	ax[2].set_xticks([500,4000,8000])

	fig.tight_layout()
	fig.savefig("{0}/spurious_eb2D.{0}".format(cmd_args.type))
	sns.set(font_scale=2)

#Fit spurious E/B power with log-linear, log-normal
def residuals_lognormal(p,x,y,weight,n=2):
	model = (p[0]*(x**n))*np.exp(-p[2]*(x-p[1])**2)
	return (y - model)/weight

def fit_power_lognormal(x,y,weight,pguess,n=2):
	pfit = leastsq(residuals_lognormal,pguess,args=(x,y,weight,n))
	return pfit[1],pfit[0][0],pfit[0][1],pfit[0][2]

def ebFit(cmd_args,realizations=range(1,21),lmin=70.0,lmax=10000.0,Nbins=100,fontsize=22):

	#Set up plot
	fig,ax = plt.subplots()
	
	#Linear spaced bins
	lbins = np.linspace(lmin,lmax,Nbins)

	#Placeholders
	Nrealizations = len(realizations)
	Nbins = len(lbins)

	all_power_binned = np.zeros([Nrealizations,Nbins-1,3])
	binned_power_average = np.zeros([Nbins-1,3])
	binned_power_std = np.zeros([Nbins-1,3])

	#Average spurious power
	for r in realizations:
	
		power,two_d_power = pixelize.power_spectrum(os.path.join(data_path,"pixelized","eb_{0}.fit".format(r)))
		l,binned_power = pixelize.bin_power_spectrum(power,lbins)
		all_power_binned[r-1,:,:] = binned_power

	binned_power_average = all_power_binned.mean(axis=0)
	binned_power_std = all_power_binned.std(axis=0)

	#Select ell, perform fit
	selected_modes1 = ((l>0.0).astype(int) * (l<=900.0).astype(int)).astype(bool)
	ll1 = l[selected_modes1]
	ee1 = binned_power_average[selected_modes1,0]
	bb1 = binned_power_average[selected_modes1,1]
	eb1 = np.abs(binned_power_average[selected_modes1,2])

	slope1,intercept1,r_value,p_value,std_err = stats.linregress(np.log10(ll1/700.0),ll1*(ll1+1)*ee1)

	selected_modes2 = ((l>900.0).astype(int) * (l<=3300.0).astype(int)).astype(bool)
	ll2 = l[selected_modes2]
	ee2 = binned_power_average[selected_modes2,0]
	bb2 = binned_power_average[selected_modes2,1]
	eb2 = np.abs(binned_power_average[selected_modes2,2])

	slope2,intercept2,r_value,p_value,std_err = stats.linregress(np.log10(ll2/700.0),ll2*(ll2+1)*ee2)

	#Fit power spectrum with modified lognormal
	selected_modes = l>200
	l2 = l[selected_modes]
	ee2 = binned_power_average[selected_modes,0]
	w2 = binned_power_std[selected_modes,0]

	logres,amplog,loclog,scalelog = fit_power_lognormal(np.log10(l2),l2*(l2+1)*ee2,l2*(l2+1)*w2,pguess=np.array([1.0,3.3,10.0]),n=1)

	#Do the piecewise plotting
	mult=1.0e4
	lcut = 3300.0
	lL = np.arange(300,710,10)
	lM = np.arange(710,lcut,10)
	lR = np.arange(lcut,10000,100)

	ax.errorbar(l,mult*l*(l+1)*binned_power_average[:,0]/(2*np.pi),yerr=mult*l*(l+1)*binned_power_std[:,0]/((2*np.pi)*np.sqrt(Nrealizations)),label=r'$S^{EE}$',linestyle='none')
	ax.errorbar(l,mult*l*(l+1)*binned_power_average[:,1]/(2*np.pi),yerr=mult*l*(l+1)*binned_power_std[:,1]/((2*np.pi)*np.sqrt(Nrealizations)),label=r'$S^{BB}$',linestyle='none')
	ax.errorbar(l,mult*l*(l+1)*abs(binned_power_average[:,2]/(2*np.pi)),yerr=mult*l*(l+1)*binned_power_std[:,2]/((2*np.pi)*np.sqrt(Nrealizations)),label=r'$\vert\mathrm{Re}(S^{EB})\vert$',linestyle='none')
	ax.plot(lL,mult*(slope1*np.log10(lL/700.0)+intercept1)/(2*np.pi),color="black",linestyle="--",label=r"${\rm Fit}$")
	ax.plot(lM,mult*(slope2*np.log10(lM/700.0)+intercept2)/(2*np.pi),color="black",linestyle="--")
	ax.plot(lR,mult*amplog*((np.log10(lR))**1)*np.exp(-scalelog*(np.log10(lR)-loclog)**2)/(2*np.pi),color="black",linestyle="--")

	#Limits
	ax.set_xlim(250,10000)
	ax.set_ylim(0.0,0.23)
	ax.set_xscale("log")

	#Labels
	ax.set_xlabel(r'$\ell$',fontsize=fontsize)
	ax.set_ylabel(r'$10^4\times \ell(\ell+1)S_\ell/2\pi$',fontsize=fontsize)
	ax.legend(loc="upper left")

	#Save
	fig.tight_layout()
	fig.savefig("{0}/spurious_fit.{0}".format(cmd_args.type))


	

