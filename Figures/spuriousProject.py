import sys,os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from astropy.io import fits

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

def ebFit(cmd_args,fontsize=22):
	pass

