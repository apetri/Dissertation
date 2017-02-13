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

#Simulation batch handler
batch = SimulationBatch.current("/Users/andreapetri/Documents/Columbia/Simulations/DEBatch/environment.ini")
batchCov = SimulationBatch.current("/Users/andreapetri/Documents/Columbia/Simulations/CovarianceBatch/environment.ini")

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

def powerSample(cmd_args,smooth=0.5*u.arcmin,ngal=(15,30,45),z=2.0,fontsize=22):

	#Set up plot
	fig,ax = plt.subplots()

	#Load data
	ell = np.load(os.path.join(batch.home,"ell_nb98.npy"))
	smth = np.exp(-(ell*smooth.to(u.rad).value)**2)
	pkappa = np.load(os.path.join(fiducial["c0"].getMapSet("kappa").home,"power_s0_nb98.npy"))
	pomega = np.load(os.path.join(fiducial["c0"].getMapSet("omega").home,"power_s0_nb98.npy"))

	#Plot kappa,omega
	ax.plot(ell,ell*(ell+1)*smth*pkappa.mean(0)/(2*np.pi),label=r"$\kappa\kappa$")
	ax.plot(ell,ell*(ell+1)*smth*pomega.mean(0)/(2*np.pi),label=r"$\omega\omega$")

	#Make up shape noise
	linestyles = ["-","--","-."]
	for n,ng in enumerate(ngal):
		level = (0.15+0.035*z)**2 / (ng*(u.arcmin**-2)).to(u.rad**-2).value
		ax.plot(ell,ell*(ell+1)*smth*level/(2*np.pi),color="black",linestyle=linestyles[n],label=r"${\rm Shape},\,\,"+r"n_g={0}".format(ng)+r"{\rm arcmin}^{-2}"+r"$")

	#Labels, scales
	ax.set_xscale("log")
	ax.set_yscale("log")
	ax.set_xlabel(r"$\ell$",fontsize=fontsize)
	ax.set_ylabel(r"$\ell(\ell+1)P(\ell)e^{-\ell^2\theta_G^2}/2\pi$",fontsize=fontsize)
	ax.legend(bbox_to_anchor=(0., 1.02, 1., .102),loc=3,ncol=2,mode="expand",borderaxespad=0.,prop={"size":15})

	#Save
	fig.savefig("{0}/powerSample.{0}".format(cmd_args.type))

def powerCov(cmd_args,fontsize=22):

	#Set up plot
	fig,ax = plt.subplots()

	#Load data (linear binning)
	ell = np.load(os.path.join(batch.home,"ell_nb100.npy"))
	nell = np.load(os.path.join(fiducial["c0"].getMapSet("kappa").home,"num_ell_nb100.npy"))
	pell0 = np.load(os.path.join(fiducial["c0"].getMapSet("kappa").home,"convergence_power_s0_nb100.npy"))
	pellN = np.load(os.path.join(fiducial["c0"].getMapSet("kappa").home,"convergence_powerSN_s0_nb100.npy"))

	#Load data (log binning)
	ellL = np.load(os.path.join(batchCov["m0c0"].home,"ell.npy"))
	nellL = np.load(os.path.join(batchCov["m0c0"].home,"num_ell.npy"))
	pellL = np.load(os.path.join(batchCov["m0c0"].getMapSet("Maps1").home,"power_spectrum_s0.npy"))

	#Mean, covariance
	p0 = pell0.mean(0)
	c0 = np.cov(pell0.T).diagonal()
	pN = pellN.mean(0)
	cN = np.cov(pellN.T).diagonal()
	pL = pellL.mean(0)
	cL = np.cov(pellL.T).diagonal()

	#Plot and compare to Gaussian prediction
	ax.plot(ell,c0*nell/(p0**2),label=r"${\rm Noiseless}$ ${\rm linear}$ ${\rm binning}$")
	ax.plot(ellL,cL*nellL/(pL**2),label=r"${\rm Noiseless}$ ${\rm log}$ ${\rm binning}$")
	ax.plot(ell,cN*nell/(pN**2),label=r"${\rm Shape, }n_g=15\,{\rm arcmin}^{-2}$")

	#Labels
	ax.set_xscale("log")
	ax.set_xlabel(r"$\ell$",fontsize=fontsize)
	ax.set_ylabel(r"${\rm cov}[\hat{P}(\ell)\hat{P}(\ell)]\times N_\ell/P^2_\ell$",fontsize=fontsize)
	ax.legend(loc="upper left",prop={"size":15})

	#Save
	fig.savefig("{0}/powerCov.{0}".format(cmd_args.type))


##########################################################################################################################

def excursion(cmd_args,smooth=0.5*u.arcmin,threshold=0.02,fontsize=22):

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

	#Overlay boundary on the image
	mask = conv.mask(exc_data.astype(np.int8))
	i,j = np.where(mask.boundary>0)
	scale = conv.resolution.to(u.deg).value
	ax[0].scatter(j*scale,i*scale,color="red",marker=".",s=0.5)
	ax[0].set_xlim(0,conv.side_angle.to(u.deg).value)
	ax[0].set_ylim(0,conv.side_angle.to(u.deg).value)

	#Format right colorbar
	cbar = exc.ax.get_images()[0].colorbar
	cbar.outline.set_linewidth(1)
	cbar.outline.set_edgecolor("black")
	cbar_ticks = cbar.set_ticks([0,0.25,0.5,0.75,1])
	cbar.ax.set_yticklabels(["",r"$\kappa<\kappa_0$","",r"$\kappa>\kappa_0$",""],rotation=90)

	#Save
	fig.tight_layout()
	fig.savefig("{0}/excursion.{0}".format(cmd_args.type))

##########################################################################################################################

def convergencePeaks(cmd_args,fontsize=22):

	#Plot setup
	fig,ax = plt.subplots(1,2,figsize=(16,8))

	#Load the convergence map and smooth on 0.5 arcmin
	conv = ConvergenceMap.load(os.path.join(fiducial["c0"].getMapSet("kappa").home,"WLconv_z2.00_0001r.fits"))
	conv.smooth(0.5*u.arcmin,kind="gaussianFFT",inplace=True)

	#Find the peak locations and height
	sigma = np.linspace(-2.,13.,101)
	height,positions = conv.locatePeaks(sigma,norm=True)

	#Show the convergence with the peak locations
	conv.visualize(fig=fig,ax=ax[0],colorbar=True,cbar_label=r"$\kappa$")
	ax[0].scatter(*positions[height>2.].to(u.deg).value.T,color="red",marker="o")
	ax[0].set_xlim(0,conv.side_angle.to(u.deg).value)
	ax[0].set_ylim(0,conv.side_angle.to(u.deg).value)

	#Build a gaussianized version of the map
	gen = GaussianNoiseGenerator.forMap(conv)
	ell = np.linspace(conv.lmin,conv.lmax,100)
	ell,Pell = conv.powerSpectrum(ell)
	convGauss = gen.fromConvPower(np.array([ell,Pell]),bounds_error=False,fill_value=0.)

	#Show the peak histogram (measured + gaussian)
	conv.peakHistogram(sigma,norm=True,fig=fig,ax=ax[1],label=r"${\rm Measured}$")
	convGauss.peakHistogram(sigma,norm=True,fig=fig,ax=ax[1],label=r"${\rm Gaussianized}$")
	conv.gaussianPeakHistogram(sigma,norm=True,fig=fig,ax=ax[1],label=r"${\rm Prediction}:(dN_{\rm pk}/d\nu)_G$")

	#Limits
	ax[1].set_ylim(1,1.0e3)

	#Labels
	ax[1].set_xlabel(r"$\kappa/\sigma_0$",fontsize=fontsize)
	ax[1].set_ylabel(r"$dN_{\rm pk}(\kappa)$")
	ax[1].legend()

	#Save
	fig.tight_layout()
	fig.savefig("{0}/convergencePeaks.{0}".format(cmd_args.type))

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
	ax.plot(ell,ell*(ell+1)*(np.abs(pFull.mean(0)-pBorn.mean(0)))/(2.0*np.pi),label=r"$P-P^{1,1}$")
	ax.plot(ell,ell*(ell+1)*np.abs(pGP_cross.mean(0))/np.pi,label=r"$2P^{1,2-{\rm gp}}$")
	ax.plot(ell,ell*(ell+1)*np.abs(pLL_cross.mean(0))/np.pi,label=r"$2P^{1,2-{\rm ll}}$")

	#Labels
	ax.legend(loc="upper left",prop={"size":20})
	ax.set_xscale("log")
	ax.set_yscale("log")
	ax.set_xlabel(r"$\ell$",fontsize=fontsize)
	ax.set_ylabel(r"$\ell(\ell+1)\vert P_{\kappa\kappa}(\ell)\vert/2\pi}$")

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
	ax.legend(bbox_to_anchor=(0., 1.02, 1., .102),loc=3,ncol=2,mode="expand", borderaxespad=0.,prop={"size":15})

	#Save
	fig.savefig("{0}/delta_m{1}.{0}".format(cmd_args.type,moment))

def plotSmoothSkew(cmd_args,collection="c0",smooth=(0.5,1.,2.,3.,5.,7.,10.),fontsize=22):

	moment = 2 

	#Lines to plot
	lines = {

	r"$\kappa^3-O(\Phi^3)$" : ("kappa",("convergence_moments_s{0}_nb9.npy",),moment,True,"denim blue","-",0),
	r"$(\kappa^{(1+2-{\rm gp})})^3-(\kappa^{(1)})^3$" : ("kappaB+GP",("convergence_moments_s{0}_nb9.npy",),moment,True,"medium green","-",1),
	r"$\delta(\kappa^{(1+2-{\rm ll})})^3-(\kappa^{(1)})^3$" : ("kappaB+LL",("convergence_moments_s{0}_nb9.npy",),moment,True,"pale red","-",2),
	r"$O(\Phi^4)_{\rm gp}$" : ("kappaBorn",("cross_skewGP_s{0}_nb1.npy",),0,False,"medium green","--",4),
	r"$O(\Phi^4)_{\rm ll}$" : ("kappaBorn",("cross_skewLL_s{0}_nb1.npy",),0,False,"pale red","--",5),
	r"$O(\Phi^4)_{\rm gp}+O(\Phi^4)_{\rm ll}$" : ("kappaBorn",("cross_skewLL_s{0}_nb1.npy","cross_skewGP_s{0}_nb1.npy"),0,False,"denim blue","--",3),

	}

	plotSmooth(cmd_args,lines,collection=collection,moment=moment,smooth=smooth,ylabel=r"$\langle\delta\kappa^3\rangle/\langle\kappa^3\rangle$",fontsize=fontsize)

def plotSmoothKurt(cmd_args,collection="c0",smooth=(0.5,1.,2.,3.,5.,7.,10.),fontsize=22):

	moment = 5 

	#Lines to plot
	lines = {

	r"$\kappa^4-O(\Phi^4)$" : ("kappa",("convergence_moments_s{0}_nb9.npy",),moment,True,"denim blue","-",0),
	r"$(\kappa^{(1+2-{\rm gp})})^4-(\kappa^{(1)})^4$" : ("kappaB+GP",("convergence_moments_s{0}_nb9.npy",),moment,True,"medium green","-",1),
	r"$(\kappa^{(1+2-{\rm ll})})^4-(\kappa^{(1)})^3$" : ("kappaB+LL",("convergence_moments_s{0}_nb9.npy",),moment,True,"pale red","-",2),
	r"$O(\Phi^5)_{\rm gp}$" : ("kappaBorn",("cross_kurtGP_s{0}_nb1.npy",),0,False,"medium green","--",4),
	r"$O(\Phi^5)_{\rm ll}$" : ("kappaBorn",("cross_kurtLL_s{0}_nb1.npy",),0,False,"pale red","--",5),
	r"$O(\Phi^5)_{\rm gp} + O(\Phi^5)_{\rm ll}$" : ("kappaBorn",("cross_kurtLL_s{0}_nb1.npy","cross_kurtGP_s{0}_nb1.npy"),0,False,"denim blue","--",3),

	}

	plotSmooth(cmd_args,lines,collection=collection,moment=moment,smooth=smooth,ylabel=r"$\langle\delta\kappa^4\rangle_c/\langle\kappa^4\rangle_c$",fontsize=fontsize)

##########################################################################################################################

def pbBias(cmd_args,feature_name="convergence_power_s0_nb100",title="Power spectrum",kappa_models=("Born",),callback=None,variation_idx=(0,),bootstrap_size=1000,resample=1000,return_results=False,fontsize=22):
	
	#Initialize plot
	fig,ax = plt.subplots(len(variation_idx),3,figsize=(24,8*len(variation_idx)))
	ax = np.atleast_2d(ax)

	##################
	#Load in the data#
	##################

	#Observation
	bootstrap_mean = lambda e: e.values.mean(0)
	feature_ray = Ensemble.read(os.path.join(fiducial["c0"].getMapSet("kappa").home,feature_name+".npy"),callback_loader=callback).bootstrap(bootstrap_mean,bootstrap_size=bootstrap_size,resample=resample,seed=0)

	#Containers for cosmological model
	modelFeatures = dict()
	for mf in kappa_models:
		modelFeatures[mf] = dict()

	parameters = dict()

	for model in models:
		parameters[model.cosmo_id] = np.array([model.cosmology.Om0,model.cosmology.w0,model.cosmology.sigma8])
		for mf in kappa_models:

			try:
				modelFeatures[mf][model.cosmo_id] = Ensemble.read(os.path.join(model["c0"].getMapSet("kappa"+mf).home,feature_name+".npy"),callback_loader=callback)
			except IOError:
				pass

	#Fit each model
	for mf in kappa_models:

		#Select correct 
		features = modelFeatures[mf]

		###############################
		#Compute the covariance matrix#
		###############################

		features_covariance = features[fiducial.cosmo_id].cov()

		################################################
		#Load in the feature to fit, bootstrap the mean#
		################################################
	
		feature_born = features[fiducial.cosmo_id].bootstrap(bootstrap_mean,bootstrap_size=bootstrap_size,resample=resample,seed=0)

		for nv,v in enumerate(variation_idx):

			###############################
			#Initialize the FisherAnalysis#
			###############################

			ftr = np.array([features[m.cosmo_id].values.mean(0) for m in [fiducial] + variations[v]])
			par = np.array([parameters[m.cosmo_id] for m in [fiducial] + variations[v]])
			fisher = FisherAnalysis.from_features(ftr,par,parameter_index=["Om","w0","si8"])

			#############
			####Fit######
			#############

			fitted_parameters_born = fisher.fit(feature_born,features_covariance)
			fitted_parameters_ray = fisher.fit(feature_ray,features_covariance)

			if return_results:
				assert len(kappa_models)==1
				assert len(variation_idx)==1

				return fitted_parameters_born,fitted_parameters_ray

			##########
			#Plotting#
			##########

			for n,p in enumerate(fisher.parameter_names):
				fitted_parameters_born[p].plot.hist(bins=50,ax=ax[nv,n],edgecolor="none",label=r"${\rm Control}$")
				fitted_parameters_ray[p].plot.hist(bins=50,ax=ax[nv,n],edgecolor="none",label=r"${\rm Observation}$")
				
				ax[nv,n].set_xlabel(plab[p],fontsize=fontsize)
				ax[nv,n].set_title(title)
				ax[nv,n].legend(loc="upper right",mode="expand",ncol=2,prop={"size":20})

	#Labels
	for a in ax.flatten():
		plt.setp(a.get_xticklabels(),rotation=30)
	
	#Save
	fig.tight_layout()
	fig.savefig("{0}/bornBias_{1}.{0}".format(cmd_args.type,feature_name))

def pbBiasPowerSN30(cmd_args,feature_name="convergence_powerSN30_s0_nb100"):
	pbBias(cmd_args,feature_name=feature_name,title=r"$P_{\kappa\kappa}(n_g=30{\rm galaxies/arcmin}^2)$")

def pbBiasMomentsSN15(cmd_args,feature_name="convergence_momentsSN15_s50_nb9"):
	pbBias(cmd_args,feature_name=feature_name,kappa_models=("Born",),title=r"${\rm Moments}(n_g=15{\rm galaxies/arcmin}^2)$")

def pbBiasMomentsSN30(cmd_args,feature_name="convergence_momentsSN30_s50_nb9"):
	pbBias(cmd_args,feature_name=feature_name,title=r"${\rm Moments}(n_g=30{\rm galaxies/arcmin}^2)$")

def pbBiasNgal(cmd_args,feature_names="convergence_powerSN{0}_s0_nb100",ngal=(10,15,20,30,40,50,60),kappa_model="Born",callback=None,variation_idx=0,bootstrap_size=1000,resample=1000,fontsize=22):
	
	#Set up plot
	fig,ax = plt.subplots()

	#Parameter placeholders
	lines = dict()

	#Cycle over ngal
	for ng in ngal:

		#Fit parameters with Born, ray
		pb,pr = pbBias(cmd_args,feature_name=feature_names.format(ng),
			kappa_models=(kappa_model,),callback=callback,variation_idx=(variation_idx,),bootstrap_size=bootstrap_size,
			resample=resample,return_results=True,fontsize=fontsize)

		#Add parameter
		for par in pb:
			if par not in lines:
				lines[par] = list()

		#Compute (pB-pR)/sigmaR
		for par in pb:
			bias = (pb[par].mean() - pr[par].mean())/pr[par].std()
			lines[par].append(bias)

	#Plot
	for par in lines:
		ax.plot(ngal,np.array(lines[par]),label=plab[par])

	#Legend
	ax.set_xlabel(r"$n_g({\rm arcmin}^{-2})$",fontsize=fontsize)
	ax.set_ylabel(r"$\langle p_{\rm born} - p_{\rm ray}\rangle/\sigma_{\rm ray}$")
	ax.legend(bbox_to_anchor=(0., 1.02, 1., .102),loc=3,ncol=3,mode="expand", borderaxespad=0.)

	#Save
	fig.tight_layout()
	fig.savefig("{0}/bornBias_ngal_{1}.{0}".format(cmd_args.type,feature_names.replace("{0}","")))