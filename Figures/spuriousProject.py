import sys,os

import numpy as np
import pandas as pd
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

from lenstools.image.convergence import ConvergenceMap
from lenstools.pipeline.simulation import LensToolsCosmology
from lenstools.statistics.constraints import FisherAnalysis

####################
####Book keeping####
####################

data_path = "/Users/andreapetri/Documents/Columbia/spurious_shear"
cosmo_parameters = ["Om","Ol","w","ns","si"]
cosmo_legend = {"Om":"Om0","Ol":"Ode0","w":"w0","ns":"ns","si":"sigma8"}

fiducial = LensToolsCosmology(sigma8=0.798)
variations = [ LensToolsCosmology(Om0=0.29,Ode0=0.71,sigma8=0.798), LensToolsCosmology(w0=-0.8,sigma8=0.798), LensToolsCosmology(sigma8=0.850) ]

#Fill in cosmological parameter values
parameter_values = np.zeros((4,3))

parameter_values[0] = fiducial.Om0,fiducial.w0,fiducial.sigma8
for n,v in enumerate(variations):
	parameter_values[n+1] = v.Om0,v.w0,v.sigma8

#Color sequence
colors = ["pale red","medium green","denim blue","dark grey","pumpkin"]

##########
#Plotting#
##########

#Atmosphere residuals visualization
def visualize(cmd_args):

	#Set up plot
	fig,axes = plt.subplots(2,2,figsize=(16,16))

	#Load data and plot
	for n,ax in enumerate(axes.reshape(4)):
		sp = ConvergenceMap.load(os.path.join(data_path,"pixelized","conv_{0}.fit".format(n+1)))
		sp.visualize(fig=fig,ax=ax,colorbar=True)

	#Save
	fig.tight_layout()
	fig.savefig("{0}/spurious_visualize.{0}".format(cmd_args.type))


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

	#Switch off grid
	for n in (0,1,2):
		ax[n].grid(b=False)

	fig.tight_layout()
	fig.savefig("{0}/spurious_eb2D.{0}".format(cmd_args.type))
	sns.set(font_scale=2)

##################################################################################################################################

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


##################################################################################################################################

def _load_features(features,nbins):

	feature_directory = os.path.join(data_path,"systematics","output_bias2","nosys")
	fname = "{0}_{1}_200z_1smth.txt"

	#Load covariance, features
	all_covariance = np.loadtxt(os.path.join(feature_directory,fname.format("cov",fiducial.cosmo_id(cosmo_parameters,cosmo_legend))))
	all_features = list()
	for m in [fiducial] + variations:
		all_features.append(np.loadtxt(os.path.join(feature_directory,fname.format("obs",m.cosmo_id(cosmo_parameters,cosmo_legend)))))
	all_features = np.array(all_features)

	#Do the slicing
	slicing = list()
	for f in features:

		if f=="power_spectrum":
			slicing += range(nbins["power_spectrum"])
		elif f=="moments":
			slicing += range(nbins["power_spectrum"],nbins["power_spectrum"]+nbins["moments"])
		elif f=="moments1pt":
			slicing += [ nbins["power_spectrum"]+n for n in (0,2,5) ]
		elif f=="minkowski":
			slicing += range(nbins["power_spectrum"]+nbins["moments"],nbins["power_spectrum"]+nbins["moments"]+nbins["minkowski"]*3)
		elif f=="peaks":
			slicing += range(nbins["power_spectrum"]+nbins["moments"]+nbins["minkowski"]*3,nbins["power_spectrum"]+nbins["moments"]+nbins["minkowski"]*3+nbins["peaks"])

	features = all_features[:,slicing]
	covariance = all_covariance[slicing][:,slicing]

	#Return 
	return features,covariance

########################################################################################################################################################################################
########################################################################################################################################################################################

def plot_constraints(cmd_args,features=(["power_spectrum"],),parameters=["Om0","sigma8"],flabels=("ps",),plabels={"Om0":r"$\Omega_m$","w0":r"$w_0$","sigma8":r"$\sigma_8$"},bounds={"Om0":(0.26,0.29)},figname="wl_constraints",legendloc="upper right",fontsize=22):

	#Binning
	nbins = {"power_spectrum":100,"moments":9,"minkowski":100,"peaks":100}

	#Set up plot
	fig,ax = plt.subplots()

	#Ellipses
	ellipses = list()

	for n,feature in enumerate(features):

		#Load features
		f,cov = _load_features(feature,nbins)

		#Instantiate Fisher analysis, compute parameter covariance
		fisher = FisherAnalysis.from_features(f,parameters=parameter_values,parameter_index=["Om0","w0","sigma8"])
		pcov = fisher.parameter_covariance(cov)[parameters].loc[parameters].values

		#Correct for bias
		pcov/=(1.+(3.-len(cov))/(1000.-1.))

		#Plot ellipse
		center = (getattr(fiducial,parameters[0]),getattr(fiducial,parameters[1]))
		ellipse = fisher.ellipse(center=center,covariance=pcov,lw=1,fill=False,edgecolor=sns.xkcd_rgb[colors[n]])
		ax.add_artist(ellipse)
		ellipses.append(ellipse)

	#Axes labels
	ax.set_xlabel(plabels[parameters[0]],fontsize=fontsize)
	ax.set_ylabel(plabels[parameters[1]],fontsize=fontsize)
	ax.legend(ellipses,flabels,loc=legendloc,ncol=2,prop={"size":15})

	#Bounds
	ax.set_xlim(*bounds[parameters[0]])
	ax.set_ylim(*bounds[parameters[1]])

	#Save
	fig.savefig("{0}/{1}_{2}.{0}".format(cmd_args.type,figname,"-".join(parameters)))

########################################################################################################################################################################################

def constraints_single1(cmd_args):
	parameters = ["Om0","sigma8"]
	features = (["power_spectrum"],["moments"],["minkowski"],["peaks"],["moments1pt"])
	flabels = [r"$P_{\kappa\kappa}$",r"${\rm Moments}$",r"$V_{0,1,2}$",r"${\rm Peaks}$",r"${\rm Moments}$ ${\rm no}$ $\nabla$"]
	bounds = {"Om0":(0.15,0.36),"w0":(-1.8,-0.2),"sigma8":(0.6,1.0)}
	figname = "wl_constraints_single"

	plot_constraints(cmd_args,parameters=parameters,features=features,flabels=flabels,bounds=bounds,figname=figname)

def constraints_single2(cmd_args):
	parameters = ["Om0","w0"]
	features = (["power_spectrum"],["moments"],["minkowski"],["peaks"],["moments1pt"])
	flabels = [r"$P_{\kappa\kappa}$",r"${\rm Moments}$",r"$V_{0,1,2}$",r"${\rm Peaks}$",r"${\rm Moments}$ ${\rm no}$ $\nabla$"]
	bounds = {"Om0":(0.15,0.36),"w0":(-1.8,-0.05),"sigma8":(0.6,1.0)}
	figname = "wl_constraints_single"

	plot_constraints(cmd_args,parameters=parameters,features=features,flabels=flabels,bounds=bounds,figname=figname,legendloc="upper left")

########################################################################################################################################################################################

def constraints_combine1(cmd_args):
	parameters = ["Om0","sigma8"]
	features = (["power_spectrum"],["power_spectrum","moments"],["power_spectrum","minkowski"],["power_spectrum","peaks"])
	flabels = [r"$P_{\kappa\kappa}$",r"$P_{\kappa\kappa}+{\rm Moments}$",r"$P_{\kappa\kappa}+V_{0,1,2}$",r"$P_{\kappa\kappa}+{\rm Peaks}$"]
	bounds = {"Om0":(0.15,0.36),"w0":(-1.8,-0.05),"sigma8":(0.6,1.0)}
	figname = "wl_constraints_combine"

	plot_constraints(cmd_args,parameters=parameters,features=features,flabels=flabels,bounds=bounds,figname=figname)

def constraints_combine2(cmd_args):
	parameters = ["Om0","w0"]
	features = (["power_spectrum"],["power_spectrum","moments"],["power_spectrum","minkowski"],["power_spectrum","peaks"])
	flabels = [r"$P_{\kappa\kappa}$",r"$P_{\kappa\kappa}+{\rm Moments}$",r"$P_{\kappa\kappa}+V_{0,1,2}$",r"$P_{\kappa\kappa}+{\rm Peaks}$"]
	bounds = {"Om0":(0.15,0.36),"w0":(-1.8,-0.05),"sigma8":(0.6,1.0)}
	figname = "wl_constraints_combine"

	plot_constraints(cmd_args,parameters=parameters,features=features,flabels=flabels,bounds=bounds,figname=figname)

