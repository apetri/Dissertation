import sys,os

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import seaborn as sns

from lenstools.image.convergence import ConvergenceMap

data_path = "/Users/andreapetri/Documents/Columbia/LSST/data"

################################################################################

def visualize(cmd_args,fontsize=22):

	#Set up plot
	fig,ax = plt.subplots(1,2,figsize=(16,8))

	#Load data: tree rings, pixel size variation
	tr = ConvergenceMap.load(os.path.join(data_path,"tree_rings.fits"))
	px = ConvergenceMap.load(os.path.join(data_path,"spurious_convergence.fits"))
	extent = np.array([0.,tr.side_angle.to(u.deg).value/15.,0.,tr.side_angle.to(u.deg).value/15.])*u.deg
	tr = tr.cutRegion(extent)
	px = px.cutRegion(extent)

	#Visualize
	tr.visualize(fig=fig,ax=ax[0],colorbar=True)
	px.visualize(fig=fig,ax=ax[1],colorbar=True)

	#Titles
	ax[0].set_title(r"${\rm Tree}$ ${\rm rings}$",fontsize=fontsize)
	ax[1].set_title(r"${\rm Pixel}$ ${\rm size}$ ${\rm variations}$",fontsize=fontsize)

	#Save
	fig.tight_layout()
	fig.savefig("{0}/sensors_visualize.{0}".format(cmd_args.type))