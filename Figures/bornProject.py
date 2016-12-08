import os

import numpy as np
import astropy.units as u

import matplotlib
import matplotlib.pyplot as plt

from lenstools.pipeline.simulation import SimulationBatch
from lenstools import ConvergenceMap

#Simulation batch handler
batch = SimulationBatch.current("/Users/andreapetri/Documents/Columbia/Simulations/DEBatch/environment.ini")

models = batch.models

fiducial = batch.getModel("Om0.260_Ode0.740_w-1.000_wa0.000_si0.800")
variations = ( 

	map(lambda m:batch.getModel(m),["Om0.290_Ode0.710_w-1.000_wa0.000_si0.800","Om0.260_Ode0.740_w-0.800_wa0.000_si0.800","Om0.260_Ode0.740_w-1.000_wa0.000_si0.900"]),
	map(lambda m:batch.getModel(m),["Om0.230_Ode0.770_w-1.000_wa0.000_si0.800","Om0.260_Ode0.740_w-1.200_wa0.000_si0.800","Om0.260_Ode0.740_w-1.000_wa0.000_si0.700"])

)

plab = { "Om":r"$\Omega_m$", "w0":r"$w_0$", "wa":r"$w_a$", "si8":r"$\sigma_8$" }

##########################################################################################################################

def convergenceVisualize(cmd_args,collection="c0",smooth=0.5*u.arcmin,fontsize=22):

	#Initialize plot
	fig,ax = plt.subplots(2,2,figsize=(16,16))

	#Load data
	cborn = ConvergenceMap.load(os.path.join(fiducial[collection].getMapSet("kappaBorn").home,"born_z2.00_0001r.fits"))
	cray = ConvergenceMap.load(os.path.join(fiducial[collection].getMapSet("kappa").home,"WLconv_z2.00_0001r.fits"))
	cll = ConvergenceMap.load(os.path.join(fiducial[collection].getMapSet("kappaLL").home,"postBorn2-ll_z2.00_0001r.fits"))
	cgp = ConvergenceMap.load(os.path.join(fiducial[collection].getMapSet("kappaGP").home,"postBorn2-gp_z2.00_0001r.fits"))

	#Smooth
	for c in (cborn,cray,cll,cgp):
		c.smooth(smooth,kind="gaussianFFT",inplace=True)

	#Plot
	cray.visualize(colorbar=True,fig=fig,ax=ax[0,0])
	(cray+cborn*-1).visualize(colorbar=True,fig=fig,ax=ax[0,1])
	cll.visualize(colorbar=True,fig=fig,ax=ax[1,0])
	cgp.visualize(colorbar=True,fig=fig,ax=ax[1,1])

	#Titles
	ax[0,0].set_title(r"$\kappa$",fontsize=fontsize)
	ax[0,1].set_title(r"$\kappa-\kappa^{(1)}$",fontsize=fontsize)
	ax[1,0].set_title(r"$\kappa^{(2-{\rm ll})}$",fontsize=fontsize)
	ax[1,1].set_title(r"$\kappa^{(2-{\rm gp})}$",fontsize=fontsize)

	#Switch off grids
	for i in (0,1):
		for j in (0,1):
			ax[i,j].grid(b=False) 

	#Save
	fig.tight_layout()
	fig.savefig("{0}/csample.{0}".format(cmd_args.type))

##########################################################################################################################

def excursion(cmd_args,smooth=0.5*u.arcmin,threshold=0.05,fontsize=22):

	#Set up plot
	fig,ax = plt.subplots(1,2,figsize=(16,8))

	#Load map
	conv = ConvergenceMap.load(os.path.join(fiducial["c0"].getMapSet("kappa").home,"WLconv_z2.00_0001r.fits"))
	conv.smooth(smooth,kind="gaussianFFT",inplace=True)

	#Build excursion set
	exc_data = np.zeros_like(conv.data)
	exc_data[conv.data>threshold] = 1.
	exc = ConvergenceMap(exc_data,angle=conv.side_angle)

	#Define binary colorbar
	cmap = plt.get_cmap("binary")
	cmaplist = [ cmap(i) for i in range(cmap.N) ]
	cmap = cmap.from_list('binary map',cmaplist,cmap.N)
	bounds = np.array([0.0,0.5,1.0])
	norm = matplotlib.colors.BoundaryNorm(bounds,cmap.N)

	#Plot the two alongside
	conv.visualize(colorbar=True,fig=fig,ax=ax[0])
	exc.visualize(colorbar=True,cmap="binary",norm=norm,fig=fig,ax=ax[1])

	#Save
	fig.tight_layout()
	fig.savefig("{0}/excursion.{0}".format(cmd_args.type))