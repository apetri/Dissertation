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
from lenstools.simulations.design import Design

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