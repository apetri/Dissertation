import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lenstools.pipeline.simulation import SimulationBatch
from lenstools.statistics.ensemble import Ensemble
from lenstools.statistics.database import Database

import algorithms

#Simulation batch handler
batch = SimulationBatch.current("/Users/andreapetri/Documents/Columbia/Simulations/CovarianceBatch/environment.ini")
data_home = os.path.join(batch.home,"data")
model_home = batch["m0"].getCollection("512b240").home

##############################
#Plot styles for each feature#
##############################

#Markers
markers = {
"power_logb_large" : "x",
"power_logb_small" : "s",
"power_logb_all" : "o",
"power_large" : "+",
"power_small" : "x",
"power_large+small" : "o",
"power_all" : "o",
"peaks_low" : "+",
"peaks_intermediate" : "*",
"peaks_high" : "d",
"peaks_low+intermediate" : "x",
"peaks_intermediate+high" : "s",
"peaks_all" : "s",
} 

#Colors
colors = {
"power_logb_large" : "dark grey",
"power_logb_small" : "dark grey",
"power_logb_all" : "pale red",
"power_large" : "pale red",
"power_small" : "pale red",
"power_large+small" : "medium green",
"power_all" : "denim blue",
"peaks_low" : "pale red",
"peaks_intermediate" : "pale red",
"peaks_high" : "pale red",
"peaks_low+intermediate" : "medium green",
"peaks_intermediate+high" : "medium green",
"peaks_all" : "dusty purple",
} 

#Labels
labels = {
"power_logb_large" : r"$\ell\in[100,800],N_d=8(\mathrm{log})$",
"power_logb_small" : r"$\ell\in[1000,6000],N_d=7(\mathrm{log})$",
"power_logb_all" : r"$\ell\in[100,6000],N_d=15(\mathrm{log})$",
"power_logb_lowest_ell" : r"$\ell\in[100,250],N_d=4(\mathrm{log})$",
"power_large" : r"$\ell\in[100,2000],N_d=15(\mathrm{lin})$",
"power_small" : r"$\ell\in[2500,4500],N_d=15(\mathrm{lin})$",
"power_large+small" : r"$\ell\in[100,4500],N_d=30(\mathrm{lin})$",
"power_all" : r"$\ell\in[2500,6000],N_d=39(\mathrm{lin})$",
"peaks_low" : r"$\kappa_0\in[-0.06,0.09],N_d=15$",
"peaks_intermediate" : r"$\kappa_0\in[0.1,0.27],N_d=15$",
"peaks_high" : r"$\kappa_0\in[0.28,0.45],N_d=15$",
"peaks_low+intermediate" : r"$\kappa_0\in[-0.06,0.27],N_d=30$",
"peaks_intermediate+high" : r"$\kappa_0\in[0.1,0.45],N_d=30$",
"peaks_all" : r"$\kappa_0\in[-0.06,0.45],N_d=45$",
"peaks_highest_kappa" : r"$\kappa_0\in[0.44,0.48],N_d=4$",
"peaks_highest_kappa_s1" : r"$\kappa_0(\theta_G=1^\prime)>0.15,N_d=20$",
} 

#Plot order
order = {
"power_logb_large" : 8,
"power_logb_small" : 7,
"power_logb_all" : 15,
"power_large" : 15,
"power_small" : 15,
"power_large+small" : 30,
"power_all" : 39,
"peaks_low" : 15,
"peaks_intermediate" : 15,
"peaks_high" : 15,
"peaks_low+intermediate" : 30,
"peaks_intermediate+high" : 30,
"peaks_all" : 45,
} 

#Offsets
offsets = {
"power_logb_large" : 0,
"power_logb_small" : 0,
"power_logb_all" : 0,
"power_large" : -0.5,
"power_small" : -1,
"power_large+small" : 0,
"power_all" : 0,
"peaks_low" : 0.5,
"peaks_intermediate" : 1,
"peaks_high" : 1.5,
"peaks_low+intermediate" : 0.5,
"peaks_intermediate+high" : 1,
"peaks_all" : 0,
}

#Curving effect of the variance versus 1/Nr fir different Nb
def curving_nb(cmd_args,db_filename="variance_scaling_nb_expected.sqlite",parameter="w",nsim=200,xlim=(0,1./65),ylim=(1,2.5),nr_top=[1000,500,300,200,150,100,90,70],fontsize=20,figname="curving_nb"):

	#Plot panel
	sns.set(font_scale=1)
	fig,ax = plt.subplots() 
	
	#################################################################################################################

	#Load the database and fit for the effective dimensionality of each feature space
	with Database(os.path.join(batch.home,"data",db_filename)) as db:

		features  = db.tables
		features.sort(key=order.get)
		
		for f in features:

			#Read the table corresponding to each feature
			v = db.read_table(f).query("nsim=={0}".format(nsim))
			v["1/nreal"] = v.eval("1.0/nreal")
			v = v.sort_values("1/nreal")

			#Find the variance in the limit of large Nr
			s0 = algorithms.fit_nbins(v,kind="linear",vfilter=lambda d:d.query("nreal>=500")).query("nsim=={0}".format(nsim))["s0"].mean()

			#Nb,Np
			Nb = v["bins"].mean()
			Np = 3

			#Plot the variance versus 1/nreal
			ax.scatter(v["1/nreal"],v["w"]/s0,color=sns.xkcd_rgb[colors[f]],marker=markers[f],label=labels[f],s=10+(100-10)*(Nb-1)/(200-1))

			#Plot the theory ppale redictions
			x = 1./np.linspace(1000,65,100)
			ax.plot(x,1+x*(Nb-Np),linestyle="--",color=sns.xkcd_rgb[colors[f]])
			ax.plot(x,1+(Nb-Np)*x+(Nb-Np)*(Nb-Np+2)*(x**2),linestyle="-",color=sns.xkcd_rgb[colors[f]])
			ax.plot(x,(1-2*x)/(1-(Nb-Np+2)*x),linestyle="-",linewidth=3,color=sns.xkcd_rgb[colors[f]])


	#Axis bounds
	ax.set_xlim(*xlim)
	ax.set_ylim(*ylim)

	#Axis labels and legends
	ax.set_xlabel(r"$1/N_r$",fontsize=fontsize)
	ax.set_ylabel(r"$\langle\hat{\mathbf{\Sigma}}_{w_0w_0}\rangle/\mathbf{\Sigma}_{w_0w_0,\infty}$",fontsize=fontsize)
	ax.legend(loc="upper left",prop={"size":10})

	#Mirror x axis to show Nr on top
	ax1 = ax.twiny()
	ax1.set_xlim(*xlim)
	ax1.set_xticks([1./n for n in nr_top])
	ax1.set_xticklabels([str(n) for n in nr_top])
	ax1.set_xlabel(r"$N_r$",fontsize=fontsize)

	#Save the figure
	fig.savefig("{1}/{0}.{1}".format(figname,cmd_args.type))
	sns.set(font_scale=2)

#####################################################################################################################################################

#Power spectrum pdf
def ps_pdf(cmd_args,nell=[0,4,9,14],nsim=[1,2,5,50,100],colors=["dark grey","denim blue","medium green","pale red","dusty purple"],fontsize=22):

	assert len(colors)>=len(nsim)
	assert len(nell)==4

	#Multipoles and number of modes
	ell = np.load(os.path.join(model_home,"ell.npy"))

	#Plot
	fig,ax = plt.subplots(2,2,figsize=(16,12))
	for nc,ns in enumerate(nsim):

		#Load the relevant ensemble,compute mean and variance
		ens = Ensemble(np.load(os.path.join(model_home,"Maps{0}".format(ns),"power_spectrum_s0.npy"))).head(1000)
		
		#Fill each sub plot
		for na,subax in enumerate(ax.reshape(4)):
			subax.hist(ens[nell[na]].values,histtype="step",bins=50,normed=True,label=r"$N_s={0}, N_r=1000$".format(ns),color=sns.xkcd_rgb[colors[nc]],lw=2)

	#Plot the result for the BIG ensemble generated with 1 simulation and 128000 realizations
	ens = Ensemble(np.load(os.path.join(model_home,"MillionMapsPower","power_spectrum_s0.npy")))

	#Fill each sub plot
	for na,subax in enumerate(ax.reshape(4)):
		subax.hist(ens[nell[na]].values,histtype="step",bins=50,normed=True,color=sns.xkcd_rgb[colors[0]],linestyle="--",label=r"$N_s=1,N_r=128000$",lw=2)

	#Labels
	for na,subax in enumerate(ax.reshape(4)):
		subax.set_xlabel(r"$P_{\kappa\kappa}(\ell)$",fontsize=fontsize)
		subax.set_ylabel(r"$\mathcal{L}(P_{\kappa\kappa}(\ell))$",fontsize=fontsize)
		subax.set_title(r"$\ell={0}$".format(int(ell[nell[na]])),fontsize=fontsize)
		if na==0:
			subax.legend()

	#Save
	fig.tight_layout()
	fig.savefig("{0}/ps_pdf.{0}".format(cmd_args.type))

#####################################################################################################################################################

#Scaling of the variance with Nr
def scaling_nr(cmd_args,db_filename="variance_scaling_nb_expected.sqlite",features=["power_logb_all","power_logb_lowest_ell","peaks_highest_kappa","peaks_highest_kappa_s1"],colors=["dark grey","denim blue","medium green","pale red","dusty purple"],nrmax=100000,parameter="w",fontsize=20):

	#Plot panel
	fig,ax = plt.subplots()

	##############################################################################################################################
	###############Consider the BIG simulation set################################################################################
	##############################################################################################################################

	#Open the database and look for different nsim
	with Database(os.path.join(data_home,"variance_scaling_largeNr.sqlite")) as db:

		for n,feature in enumerate(features):
			v = db.query("SELECT nsim,nreal,bins,{0} FROM {1}".format(parameter,feature))

			#Number of bins
			nb = v["bins"].iloc[0]

			#Group by nsim
			nsim_group = v.groupby("nsim")

			for ns in [1]:
				vgroup = nsim_group.get_group(ns)

				#Estimate the intercept using the variance at a high number of realizations
				s0 = vgroup.query("nreal=={0}".format(nrmax))[parameter].iloc[0]
				ax.plot(vgroup["nreal"],vgroup[parameter]-s0,linestyle="-",lw=2,color=sns.xkcd_rgb[colors[n]],label=labels[feature])
				ax.plot(vgroup["nreal"].values,s0*(nb-3)/vgroup["nreal"].values,linestyle="--",color=sns.xkcd_rgb[colors[n]],label=None)


	####################################################################################################################################

	#Axes scale
	ax.set_xscale("log")
	ax.set_yscale("log")

	#Axes limits
	ax.set_xlim(100,110000)
	ax.set_ylim(1.0e-7,1.0e3)

	#Labels
	ax.set_xlabel(r"$N_r$",fontsize=fontsize)
	ax.set_ylabel(r"$\langle\hat{\Sigma}_{w_0w_0}\rangle - \Sigma_{w_0w_0,\infty}$",fontsize=fontsize)
	ax.legend(loc="lower left",prop={"size":13})

	#Save
	fig.tight_layout()
	fig.savefig("{0}/scaling_nr.{0}".format(cmd_args.type))

#####################################################################################################################################################

#Scaling of the variance with Ns
def scaling_ns(cmd_args,db_filename="variance_scaling_nb_expected.sqlite",features=["power_logb_all","power_all","peaks_all"],fit_kind="quadratic",nreal_min=500,colors=["dark grey","pale red","medium green"],parameter="w",fontsize=20):

	assert len(colors)==len(features)

	#Plot panel
	fig,ax = plt.subplots()

	#Labels
	labels = {
	"power_logb_large" : None,
	"power_logb_small" : None,
	"power_logb_all" : "Power spectrum log binning",
	"power_large" : None,
	"power_small" : None,
	"power_large+small" : None,
	"power_all" : "Power spectrum linear binning",
	"peaks_low" : None,
	"peaks_intermediate" : None,
	"peaks_high" : None,
	"peaks_low+intermediate" : None,
	"peaks_intermediate+high" : None,
	"peaks_all" : "Peak counts",
	} 

	#Load the database and fit for the effective dimensionality of each feature space
	with Database(os.path.join(data_home,db_filename)) as db:
		nb_fit = algorithms.fit_nbins_all(db,parameter=parameter,kind=fit_kind,nreal_min=nreal_min)

	#Plot the variance coefficient for each feature
	for nc,f in enumerate(features):
		nb_fit_feature = nb_fit.query("feature=='{0}'".format(f)).sort_values("nsim")
		nb_fit_feature["relative"] = nb_fit_feature["s0"] / nb_fit_feature["s0"].mean() 
		nb_fit_feature.plot(x="nsim",y="relative",ax=ax,color=sns.xkcd_rgb[colors[nc]],label=labels[f],legend=False)

	#Labels
	ax.set_xlim(-10,210)
	ax.set_xlabel(r"$N_s$",fontsize=fontsize)
	ax.set_ylabel(r"$\Sigma_{\infty}(N_s)/\Sigma_{\infty,mean}$",fontsize=fontsize)
	ax.legend()

	#Save
	fig.tight_layout()
	fig.savefig("{0}/scaling_ns.{0}".format(cmd_args.type))

#####################################################################################################################################################

#Mean feature as a function of Nsim
def means_nsim(cmd_args,fontsize=22,figname="means_ns"):

	#Number of simulations
	nsim = [1,2,5,10,20,30,40,50,60,70,80,90,100,150,200]

	#Multipoles and peak thresholds
	ell = np.load(os.path.join(model_home,"ell.npy"))
	vpk = np.load(os.path.join(model_home,"th_peaks.npy"))

	#Multipoles to select
	ell_select = [0,8,-1]

	#Peak thresholds to select
	vpk_select = [10,20,30]

	#Create the labels
	labels = [r"$\ell={0}$".format(int(ell[n])) for n in ell_select] + [r"$\kappa_0={0:.2f}$".format(vpk[n]) for n in vpk_select]

	#Load all the features
	power_spectrum = np.empty((len(ell_select),len(nsim)))
	for n,ns in enumerate(nsim):
		ensemble_mean = np.load(os.path.join(model_home,"Maps{0}".format(ns),"power_spectrum_s0.npy")).mean(0)
		power_spectrum[:,n] = ensemble_mean[ell_select]

	power_spectrum_std = np.load(os.path.join(model_home,"Maps200","power_spectrum_s0.npy")).std(0)[ell_select]

	peaks = np.empty((len(vpk_select),len(nsim)))
	for n,ns in enumerate(nsim):
		ensemble_mean = np.load(os.path.join(model_home,"Maps{0}".format(ns),"peaks_s0.npy")).mean(0)
		peaks[:,n] = ensemble_mean[vpk_select]

	peaks_std = np.load(os.path.join(model_home,"Maps200","peaks_s0.npy")).std(0)[vpk_select]

	all_features = np.vstack((power_spectrum,peaks))
	all_features_std = np.hstack((power_spectrum_std,peaks_std))

	#Plot the ensemble means as a function of nsim
	fig,ax = plt.subplots()
	for n,f in enumerate(all_features):
		ax.plot(nsim,(f-f[-1])/all_features_std[n],label=labels[n])

	#Plot the 10% accuracy line for reference
	ax.plot(np.linspace(0,210,3),np.ones(3)*0.1,linestyle="--",linewidth=2,color="black")

	#Labels
	ax.set_xlim(-10,210)
	ax.set_ylim(-0.8,0.65)
	ax.set_xlabel(r"$N_s$",fontsize=fontsize)
	ax.set_ylabel(r"$[d_i(N_s)-d_i(200)]/\sqrt{C_{ii}(200)}$",fontsize=fontsize)
	ax.legend(loc="lower right",prop={"size":15})

	#Save
	fig.tight_layout()
	fig.savefig("{0}/{1}.{0}".format(cmd_args.type,figname))
