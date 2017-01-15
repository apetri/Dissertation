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
"power_logb_large" : "black",
"power_logb_small" : "black",
"power_logb_all" : "red",
"power_large" : "red",
"power_small" : "red",
"power_large+small" : "green",
"power_all" : "blue",
"peaks_low" : "red",
"peaks_intermediate" : "red",
"peaks_high" : "red",
"peaks_low+intermediate" : "green",
"peaks_intermediate+high" : "green",
"peaks_all" : "magenta",
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
def curving_nb(cmd_args,db_filename="variance_scaling_nb_expected.sqlite",parameter="w",nsim=200,xlim=(0,1./65),ylim=(1,2.5),nr_top=[1000,500,300,200,150,100,90,70],fontsize=22,figname="curving_nb"):

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
			ax.scatter(v["1/nreal"],v["w"]/s0,color=colors[f],marker=markers[f],label=labels[f],s=10+(100-10)*(Nb-1)/(200-1))

			#Plot the theory predictions
			x = 1./np.linspace(1000,65,100)
			ax.plot(x,1+x*(Nb-Np),linestyle="--",color=colors[f])
			ax.plot(x,1+(Nb-Np)*x+(Nb-Np)*(Nb-Np+2)*(x**2),linestyle="-",color=colors[f])
			ax.plot(x,(1-2*x)/(1-(Nb-Np+2)*x),linestyle="-",linewidth=3,color=colors[f])


	#Axis bounds
	ax.set_xlim(*xlim)
	ax.set_ylim(*ylim)

	#Axis labels and legends
	ax.set_xlabel(r"$1/N_r$",fontsize=fontsize)
	ax.set_ylabel(r"$\langle\hat{\mathbf{\Sigma}}_{ww}\rangle/\mathbf{\Sigma}_{ww,\infty}$",fontsize=fontsize)
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