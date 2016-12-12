import os
from operator import add
from functools import reduce

import numpy as np
import astropy.units as u

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

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

def excursion(cmd_args,smooth=0.5*u.arcmin,threshold=0.03,fontsize=22):

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
	cmap = plt.get_cmap("RdBu")
	cmaplist = [ cmap(i) for i in range(cmap.N) ]
	cmap = cmap.from_list("binary map",cmaplist,cmap.N)
	bounds = np.array([0.0,0.5,1.0])
	norm = matplotlib.colors.BoundaryNorm(bounds,cmap.N)

	#Plot the two alongside
	conv.visualize(colorbar=True,cbar_label=r"$\kappa$",fig=fig,ax=ax[0])
	exc.visualize(colorbar=True,cmap="binary",norm=norm,fig=fig,ax=ax[1])

	#Format right colorbar
	cbar = exc.ax.get_images()[0].colorbar
	cbar.outline.set_linewidth(1)
	cbar.outline.set_edgecolor("black")
	cbar_ticks = cbar.set_ticks([0,0.25,0.5,0.75,1])
	cbar.set_ticklabels(["",r"$\kappa<\kappa_T$","",r"$\kappa>\kappa_T$",""])

	#Save
	fig.tight_layout()
	fig.savefig("{0}/excursion.{0}".format(cmd_args.type))

##########################################################################################################################

def powerResiduals(cmd_args,collection="c0",fontsize=22):

	#Initialize plot
	fig,ax = plt.subplots()

	#Load data
	ell = np.load(os.path.join(batch.home,"ell_nb100.npy"))
	pFull = np.load(os.path.join(fiducial[collection].getMapSet("kappa").home,"convergence_power_s0_nb100.npy"))
	pBorn = np.load(os.path.join(fiducial[collection].getMapSet("kappaBorn").home,"convergence_power_s0_nb100.npy"))
	pLL_cross = np.load(os.path.join(fiducial[collection].getMapSet("kappaBorn").home,"cross_powerLL_s0_nb100.npy"))
	pGP_cross = np.load(os.path.join(fiducial[collection].getMapSet("kappaBorn").home,"cross_powerGP_s0_nb100.npy"))

	#Plot
	ax.plot(ell,ell*(ell+1)*(np.abs(pFull.mean(0)-pBorn.mean(0)))/(2.0*np.pi),label=r"${\rm ray-born}$")
	ax.plot(ell,ell*(ell+1)*np.abs(pGP_cross.mean(0))/np.pi,label=r"$2{\rm born}\times{\rm geo}$")
	ax.plot(ell,ell*(ell+1)*np.abs(pLL_cross.mean(0))/np.pi,label=r"$2{\rm born}\times{\rm ll}$")

	#Labels
	ax.legend(loc="upper left",prop={"size":20})
	ax.set_xscale("log")
	ax.set_yscale("log")
	ax.set_xlabel(r"$\ell$",fontsize=fontsize)
	ax.set_ylabel(r"$\ell(\ell+1)\vert P_\ell\vert/2\pi}$")

	#Save
	fig.tight_layout()
	fig.savefig("{0}/powerResiduals.{0}".format(cmd_args.type))

##########################################################################################################################

def plotSmooth(cmd_args,lines,collection="c0",moment=2,smooth=(0.5,1.,2.,3.,5.,7.,10.),ylabel=None,fontsize=22):

	#Set up plot
	fig,ax = plt.subplots()

	#Load reference data
	reference = list()
	for s in smooth:
		reference.append(np.load(os.path.join(fiducial[collection].getMapSet("kappaBorn").home,"convergence_moments_s{0}_nb9.npy".format(int(s*100))))[:,moment].mean())
	reference = np.array(reference)

	#Plot each of the lines
	lk = lines.keys()
	lk.sort(key=lambda k:lines[k][-1])

	for l in lk:
		ms,feat,idx,subtract,color,linestyle,order = lines[l]
		data = list()

		for s in smooth:
			addends = [ np.load(os.path.join(fiducial[collection].getMapSet(ms).home,f.format(int(s*100))))[:,idx].mean() for f in feat ]
			data.append(reduce(add,addends))

		data = np.array(data)
		if subtract:
			data-=reference

		ax.plot(smooth,data/reference,color=sns.xkcd_rgb[color],linestyle=linestyle,label=l)


	#Labels
	ax.set_xlabel(r"$\theta_G({\rm arcmin})$",fontsize=fontsize)
	ax.set_ylabel(ylabel,fontsize=fontsize)
	ax.legend(bbox_to_anchor=(0., 1.02, 1., .102),loc=3,ncol=2,mode="expand", borderaxespad=0.)

	#Save
	fig.savefig("{0}/delta_m{1}.{0}".format(cmd_args.type,moment))

def plotSmoothSkew(cmd_args,collection="c0",smooth=(0.5,1.,2.,3.,5.,7.,10.),fontsize=22):

	moment = 2 

	#Lines to plot
	lines = {

	r"$\kappa^3_{\rm ray}-\kappa^3_{\rm born}$" : ("kappa",("convergence_moments_s{0}_nb9.npy",),moment,True,"denim blue","-",0),
	r"$\kappa^3_{\rm born+geo}-\kappa^3_{\rm born}$" : ("kappaB+GP",("convergence_moments_s{0}_nb9.npy",),moment,True,"medium green","-",1),
	r"$\kappa^3_{\rm born+ll}-\kappa^3_{\rm born}$" : ("kappaB+LL",("convergence_moments_s{0}_nb9.npy",),moment,True,"pale red","-",2),
	r"$3\kappa^2_{\rm born}\kappa_{\rm geo}$" : ("kappaBorn",("cross_skewGP_s{0}_nb1.npy",),0,False,"medium green","--",4),
	r"$3\kappa^2_{\rm born}\kappa_{\rm ll}$" : ("kappaBorn",("cross_skewLL_s{0}_nb1.npy",),0,False,"pale red","--",5),
	r"$3\kappa^2_{\rm born}\kappa_{\rm ll}+3\kappa^2_{\rm born}\kappa_{\rm geo}$" : ("kappaBorn",("cross_skewLL_s{0}_nb1.npy","cross_skewGP_s{0}_nb1.npy"),0,False,"denim blue","--",3),

	}

	plotSmooth(cmd_args,lines,collection=collection,moment=moment,smooth=smooth,ylabel=r"$\langle\delta\kappa^3\rangle/\langle\kappa^3\rangle$",fontsize=fontsize)

def plotSmoothKurt(cmd_args,collection="c0",smooth=(0.5,1.,2.,3.,5.,7.,10.),fontsize=22):

	moment = 5 

	#Lines to plot
	lines = {

	r"$\kappa^4_{\rm ray}-\kappa^4_{\rm born}$" : ("kappa",("convergence_moments_s{0}_nb9.npy",),moment,True,"denim blue","-",0),
	r"$\kappa^4_{\rm born+geo}-\kappa^4_{\rm born}$" : ("kappaB+GP",("convergence_moments_s{0}_nb9.npy",),moment,True,"medium green","-",1),
	r"$\kappa^4_{\rm born+ll}-\kappa^4_{\rm born}$" : ("kappaB+LL",("convergence_moments_s{0}_nb9.npy",),moment,True,"pale red","-",2),
	r"$4\kappa^3_{\rm born}\kappa_{\rm geo}$" : ("kappaBorn",("cross_kurtGP_s{0}_nb1.npy",),0,False,"medium green","--",4),
	r"$4\kappa^4_{\rm born}\kappa_{\rm ll}$" : ("kappaBorn",("cross_kurtLL_s{0}_nb1.npy",),0,False,"pale red","--",5),
	r"$4\kappa^3_{\rm born}\kappa_{\rm ll}+4\kappa^3_{\rm born}\kappa_{\rm geo}$" : ("kappaBorn",("cross_kurtLL_s{0}_nb1.npy","cross_kurtGP_s{0}_nb1.npy"),0,False,"denim blue","--",3),

	}

	plotSmooth(cmd_args,lines,collection=collection,moment=moment,smooth=smooth,ylabel=r"$\langle\delta\kappa^4\rangle_c/\langle\kappa^4\rangle_c$",fontsize=fontsize)

##########################################################################################################################