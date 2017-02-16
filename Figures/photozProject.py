import os
from operator import add
from functools import reduce

import numpy as np
import astropy.units as u

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from lenstools.pipeline.simulation import SimulationBatch
from lenstools import ConvergenceMap,GaussianNoiseGenerator,Ensemble
from lenstools.statistics.constraints import FisherAnalysis
from lenstools.statistics.ensemble import SquareMatrix
from lenstools.simulations.design import Design
from lenstools.catalog.shear import Catalog

from featureDB import FisherDatabase,LSSTSimulationBatch

#Simulation batch handler
batch = LSSTSimulationBatch.current("/Users/andreapetri/Documents/Columbia/Simulations/LSST100Parameters/environment.ini")
data_home = os.path.join(batch.home,"data")

#Plot labels
par2label = {

"Om" : r"$\Omega_m$" ,
"w" : r"$w_0$" ,
"sigma8" : r"$\sigma_8$"

}

#Fiducial value
par2value = {

"Om" : 0.26 ,
"w" : -1. ,
"sigma8" : 0.8

}

#Plotting properties
feature_properties = {

"ps" : {"name":"power_spectrum_pca","table_name" : "pcov_noise", "pca_components" : 30, "color" : "pale red", "label" : r"$P_{\kappa\kappa}(N_c=30)$","linestyle" : "--", "marker" : "x"},
"ps70" : {"name":"power_spectrum_pca","table_name" : "pcov_noise", "pca_components" : 70, "color" : "pale red", "label" : r"$P_{\kappa\kappa}(N_c=70)$","linestyle" : "-", "marker" : "+"},

"mu" : {"name":"moments_pca","table_name" : "pcov_noise", "pca_components" : 30, "color" : "denim blue", "label" : r"${\rm Moments}(N_c=30)$","linestyle" : "--", "marker" : "x"},
"mu40" : {"name":"moments_pca","table_name" : "pcov_noise", "pca_components" : 40, "color" : "denim blue", "label" : r"${\rm Moments}(N_c=40)$","linestyle" : "-", "marker" : "+"},

"pk" : {"name":"peaks_pca", "table_name": "pcov_noise", "pca_components" : 40, "color" : "medium green", "label" : r"${\rm Peaks}(N_c=40)$","linestyle" : "--", "marker" : "x"},
"pk70" : {"name":"peaks_pca", "table_name": "pcov_noise", "pca_components" : 70, "color" : "medium green", "label" : r"${\rm Peaks}(N_c=70)$","linestyle" : "-", "marker" : "+"},

"ps+pk" : {"name" : "power_spectrum+peaks" , "table_name" : "pcov_noise_combine", "pca_components" : 30+40, "color" : "pumpkin", "label" : r"$P_{\kappa\kappa}(N_c=30)+{\rm Peaks}(N_c=40)$","linestyle" : "-", "marker" : "x"},
"ps+mu" : {"name" : "power_spectrum+moments" , "table_name" : "pcov_noise_combine", "pca_components" : 30+30, "color" : "dusty purple", "label" : r"$P_{\kappa\kappa}(N_c=30)+{\rm Moments}(N_c=30)$","linestyle" : "-", "marker" : "x"},
"ps+pk+mu" : {"name" : "power_spectrum+peaks+moments" , "table_name" : "pcov_noise_combine", "pca_components" : 30+30+40, "color" : "dark grey", "label" : r"$P_{\kappa\kappa}(N_c=30)+{\rm Peaks}(N_c=40)+{\rm Moments}(N_c=30)$","linestyle" : "-", "marker" : "x"}

}

###################################################################################################
###################################################################################################

def galdistr(cmd_args,fontsize=22):

	#Set up plot
	fig,ax = plt.subplots()

	#Read in galaxy redshifts
	colors = ["denim blue","medium green","pale red","dusty purple","pumpkin"]
	position_files = [os.path.join(batch.home,"data","positions_bin{0}.fits".format(n)) for n in range(1,6)]
	for n,f in enumerate(position_files):
		z = Catalog.read(f)["z"]

		#Make the histogram
		ng,zb = np.histogram(z,bins=np.arange(z.min(),z.max(),0.02))
		ax.plot(0.5*(zb[1:]+zb[:-1]),ng,color=sns.xkcd_rgb[colors[n]],label=r"$z_s\in[{0:.2f},{1:.2f}]$".format(z.min(),z.max()))
		ax.fill_between(0.5*(zb[1:]+zb[:-1]),np.zeros_like(ng),ng,color=sns.xkcd_rgb[colors[n]],alpha=0.3)

	#Labels
	ax.set_xlabel(r"$z_s$",fontsize=fontsize)
	ax.set_ylabel(r"$N_g(z_s)$",fontsize=fontsize)
	ax.legend()

	#Ticks
	ax.set_ylim(0,2.0e4)
	ax.tick_params(axis="both",which="major",labelsize=fontsize)

	#Save figure
	fig.savefig("{0}/lsst_galdistr.{0}".format(cmd_args.type))

###################################################################################################
###################################################################################################

def pca_components(cmd_args,db_name="constraints_combine.sqlite",feature_label="power_spectrum_pca",parameter="w",fontsize=20):

	#Query database
	with FisherDatabase(os.path.join(data_home,db_name)) as db:

		fig,ax = plt.subplots()

		#Get values without PCA
		var_db = db.query('SELECT "{0}-{0}",feature_label FROM pcov_noise_no_pca'.format(parameter))
		var_feature = var_db[var_db["feature_label"].str.contains(feature_label.strip("_pca"))]["{0}-{0}".format(parameter)].values
		
		#Plot single redshift
		for n in range(5):
			b,p = db.query_parameter_simple(feature_label+"_z{0}".format(n),table_name="pcov_noise",parameter="w")
			ax.plot(b,np.sqrt(p/var_feature[n]),label=r"$z\in[{0:.2f},{1:.2f}]$".format(*db.z_bins[n]))

	#Plot tomography
	b,p = db.query_parameter_simple(feature_label,table_name="pcov_noise",parameter="w")
	ax.plot(b,np.sqrt(p/var_feature[-1]),lw=3.5,label=r"$z$"+ r" ${\rm tomography}$")

	#Labels
	ax.set_xlabel(r"$N_c$",fontsize=fontsize)
	ax.set_ylabel(r"$\sqrt{\Sigma_{"+"{0}{0}".format(par2label[parameter].replace("$",""))+r"}}({\rm PCA/Full})$",fontsize=fontsize)
	ax.legend(ncol=2,prop={"size":15})
	ax.set_title(r"${\rm " + feature_label.replace("_pca","").replace("_","\,\,") + r"}$",fontsize=fontsize)

	#Ticks
	ax.set_ylim(0.9,3)
	ax.tick_params(axis="both",which="major",labelsize=fontsize)

	#Save figure
	fig.tight_layout()
	fig.savefig("{0}/{1}_{2}.{0}".format(cmd_args.type,parameter,feature_label))

def pca_components_power_spectrum(cmd_args):
	pca_components(cmd_args,feature_label="power_spectrum_pca")

def pca_components_peaks(cmd_args):
	pca_components(cmd_args,feature_label="peaks_pca")

def pca_components_moments(cmd_args):
	pca_components(cmd_args,feature_label="moments_pca")

##################################################################################################################
##################################################################################################################
##################################################################################################################

def photoz_bias(cmd_args,dbn="data/fisher/constraints_photoz.sqlite",parameters=["Om","w"],features_to_show=["ps","ps70","pk","pk70","mu","mu40"],fontsize=22):
	
	#Init figure
	fig,ax = plt.subplots()

	#Database
	db_name = os.path.join(batch.home,dbn)

	#Ellipses and labels
	ellipses = list()
	labels = list()

	#Cycle over features
	for f in features_to_show:

		#Feature properties
		feature_label = feature_properties[f]["name"]
		nbins = feature_properties[f]["pca_components"]
		color = sns.xkcd_rgb[feature_properties[f]["color"]]
		plot_label = feature_properties[f]["label"]
		marker = feature_properties[f]["marker"]
		linestyle = feature_properties[f]["linestyle"]

		############
		#No photo-z#
		############

		with FisherDatabase(db_name) as db:
			pfit = db.query_parameter_fit(feature_label,table_name="mocks_without_photoz",parameters=parameters).query("bins=={0}".format(nbins))
			p1f,p2f = [ pfit[parameters[n]+"_fit"].values for n in [0,1] ]

		############################
		#With photo-z: requirements#
		############################

		with FisherDatabase(db_name) as db:
			pfit = db.query_parameter_fit(feature_label,table_name="mocks_photoz_requirement",parameters=parameters).query("bins=={0}".format(nbins))
			p1,p2 = [ pfit[parameters[n]+"_fit"].values for n in [0,1] ]
			ax.scatter(p1-p1f,p2-p2f,color=color,marker=marker,s=30)
			ax.scatter((p1-p1f).mean(),(p2-p2f).mean(),color=color,marker="s")

			#Draw an error ellipse around the mean bias
			center = ((p1-p1f).mean(),(p2-p2f).mean())
			pcov = np.cov([p1-p1f,p2-p2f]) 
			ellipses.append(FisherAnalysis.ellipse(center,pcov,p_value=0.677,fill=False,edgecolor=color,linestyle=linestyle,lw=1))
			ax.add_artist(ellipses[-1])
			labels.append(plot_label)

	#Get axes bounds
	xlim = np.abs(np.array(ax.get_xlim())).max()
	ylim = np.abs(np.array(ax.get_ylim())).max()

	#Show the fiducial value
	ax.plot(np.zeros(100),np.linspace(-ylim,ylim,100),linestyle="--",color="black")
	ax.plot(np.linspace(-xlim,xlim,100),np.zeros(100),linestyle="--",color="black")

	#Set the axes bounds
	ax.set_xlim(-xlim,xlim)
	ax.set_ylim(-ylim,ylim)

	#Legends
	ax.set_xlabel(par2label[parameters[0]].rstrip("$") + r"^{\rm ph}$" + r"$-$" + par2label[parameters[0]],fontsize=fontsize)
	ax.set_ylabel(par2label[parameters[1]].rstrip("$") + r"^{\rm ph}$" + r"$-$" + par2label[parameters[1]],fontsize=fontsize)
	ax.legend(ellipses,labels,prop={"size" : 17},bbox_to_anchor=(0.,1.02,1.,.102),loc=3,ncol=1,mode="expand",borderaxespad=0.)
	plt.setp(ax.get_xticklabels(),rotation=30)

	#Save figure
	fig.savefig("{0}/photoz_bias_{1}.{0}".format(cmd_args.type,"-".join(parameters)))

##################################################################################################################
##################################################################################################################

def parameter_constraints(cmd_args,dbn="data/fisher/constraints_combine.sqlite",features_to_show=["ps","ps70","pk","pk70","mu","mu40","ps+pk","ps+mu","ps+pk+mu"],parameters=["Om","w"],all_parameters=["Om","w","sigma8"],xlim=(0.252,0.267),ylim=(-1.04,-0.96),cmb_prior_fisher=None,suffix="lensing",fontsize=22):

	#Database
	db_name = os.path.join(batch.home,dbn)

	#Init figure
	fig,ax = plt.subplots()
	ellipses = list()
	labels = list()

	#CMB prior
	if cmb_prior_fisher is not None:
		fisher_cmb = SquareMatrix.read(os.path.join(batch.home,cmb_prior_fisher))[all_parameters]

	#Plot the features 
	with FisherDatabase(db_name) as db:
		for f in features_to_show:

			#Query parameter covariance
			pcov = db.query_parameter_covariance(feature_properties[f]["name"],nbins=feature_properties[f]["pca_components"],table_name=feature_properties[f]["table_name"],parameters=parameters)

			#Show the ellipse
			center = (par2value[parameters[0]],par2value[parameters[1]])
			ellipse = FisherAnalysis.ellipse(center=center,covariance=pcov.values,p_value=0.677,fill=False,edgecolor=sns.xkcd_rgb[feature_properties[f]["color"]],linestyle=feature_properties[f]["linestyle"],lw=1)
			ax.add_artist(ellipse)

			#Labels
			ellipses.append(ellipse)
			labels.append(feature_properties[f]["label"])

			#If there is a CMB prior to include, add the fisher matrices
			if cmb_prior_fisher is not None:

				#Fisher matrix addition in quadrature
				pcov_lensing = db.query_parameter_covariance(feature_properties[f]["name"],nbins=feature_properties[f]["pca_components"],table_name=feature_properties[f]["table_name"],parameters=all_parameters)
				pcov_lensing_cmb = (pcov_lensing.invert() + fisher_cmb).invert()[parameters]

				#Add additional ellipses for the constraints with prior
				ax.add_artist(FisherAnalysis.ellipse(center=center,covariance=pcov_lensing_cmb.values,p_value=0.677,fill=False,edgecolor=sns.xkcd_rgb[feature_properties[f]["color"]],linestyle=feature_properties[f]["linestyle"],lw=4))

	#Axes bounds
	ax.set_xlim(*xlim)
	ax.set_ylim(*ylim)

	#Axes labels and legend
	ax.set_xlabel(par2label[parameters[0]],fontsize=fontsize)
	ax.set_ylabel(par2label[parameters[1]],fontsize=fontsize)
	ax.legend(ellipses,labels,prop={"size" : 17},bbox_to_anchor=(0.,1.02,1.,.102),loc=3,ncol=1,mode="expand",borderaxespad=0.)
	plt.setp(ax.get_xticklabels(),rotation=30)

	#Save figure
	fig.savefig("{0}/photoz_constraints_{1}_{2}.{0}".format(cmd_args.type,"-".join(parameters),suffix))


def parameter_constraints_with_cmb(cmd_args):
	parameter_constraints(cmd_args,features_to_show=["ps70","pk70","mu40","ps+pk","ps+mu","ps+pk+mu"],cmb_prior_fisher="data/planck/planck_base_w_TT_lowTEB.pkl",suffix="lensing_cmb")