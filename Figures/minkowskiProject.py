import sys,os

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

from lenstools.pipeline.simulation import LensToolsCosmology

####################
####Book keeping####
####################

data_path = "/Users/andreapetri/Documents/Columbia/Minkowski"
cosmo_parameters = ["Om","Ol","w","ns","si"]
cosmo_legend = {"Om":"Om0","Ol":"Ode0","w":"w0","ns":"ns","si":"sigma8"}

fiducial = LensToolsCosmology(sigma8=0.798)
variations_plus = [ LensToolsCosmology(Om0=0.29,Ode0=0.71,sigma8=0.798), LensToolsCosmology(w0=-0.8,sigma8=0.798), LensToolsCosmology(sigma8=0.850) ]
variations_minus = [ LensToolsCosmology(Om0=0.23,Ode0=0.77,sigma8=0.798), LensToolsCosmology(w0=-1.2,sigma8=0.798), LensToolsCosmology(sigma8=0.750) ]

#########################
########Plotting#########
#########################

def minkPerturbation(cmd_args,nreal=1000,fontsize=26):
	
	#Set up plot
	fig,ax = plt.subplots(1,3,figsize=(24,8))

	#Load Minkowski functionals
	fname = os.path.join(data_path,"Output_final","{0}_"+"{0}_200z_5smth.txt".format(fiducial.cosmo_id(cosmo_parameters,cosmo_legend)))
	
	#Load statistics
	kappa,mink = load_mink(fname.format("mink"))
	mean,var,third,fourth = load_stats(fname.format("stat"))
	sigma,skew,kurt = compute_cumulants(var,third,fourth)

	#Compute normalized thresholds
	x = (kappa-mean) / sigma[0]

	#Compute Gaussian predictions
	gauss = mink_gauss(x,sigma)
	skew_corr = skew_correction(x,sigma,skew)
	kurt_corr = kurt_correction(x,sigma,skew,kurt)

	#Label add on 
	lab_add = (r"/\theta_{\rm FOV}^2",r"/4\theta_{\rm FOV}^2",r"/2\pi\theta_{\rm FOV}^2")

	#Plot
	for n in range(3):
		
		ax[n].plot(x,gauss[n],label=r"${\rm Gaussian}$")
		ax[n].plot(x,gauss[n]+skew_corr[n],label=r"${\rm Gaussian}+O(\sigma_0)$")
		ax[n].plot(x,gauss[n]+skew_corr[n]+kurt_corr[n],label=r"${\rm Gaussian}+O(\sigma_0)+O(\sigma_0^2)$")
		ax[n].errorbar(x,mink[n][0],yerr=mink[n][1]/np.sqrt(nreal),linestyle="none",marker=".",label=r"${\rm Measured}$")

		ax[n].get_yaxis().get_major_formatter().set_powerlimits((-2,0))
		ax[n].set_xlabel(r"$\kappa/\sigma_0$",fontsize=fontsize)
		ax[n].set_ylabel(r"$V_{0}(\kappa)".format(n)+lab_add[n]+r"$",fontsize=fontsize)

		if n==0:
			ax[n].legend(loc="upper right",prop={"size":20})

	#Save
	fig.tight_layout()
	fig.savefig("{0}/minkPerturbation.{0}".format(cmd_args.type))


def seriesConvergence():
	pass

def contours():
	pass

#############################################
########Analytical formula utilities#########
#############################################

h2 = lambda x: x**2-1
h3 = lambda x: x**3-3*x
h4 = lambda x: x**4 - 6*(x**2) +3
h5 = lambda x: x**5-10*(x**3)+15*x
h6 = lambda x: x**6 - 15*(x**4) + 45*(x**2) -15
h7 = lambda x: x**7 - 21*(x**5) + 105*(x**3) -105*x
h8 = lambda x: x**8 - 28*(x**6) + 210*(x**4) - 420*(x**2) +105

def mink_gauss(x,sigma):
	
	V0=0.5*sp.erfc(x/np.sqrt(2))
	V1=sigma[1]/(8*np.sqrt(2)*sigma[0])*np.exp(-0.5*(x**2))
	V2=(x/(2*(2*np.pi)**1.5))*((sigma[1]/sigma[0])**2)*np.exp(-0.5*(x**2))
 
 	return V0,V1,V2

def skew_correction(x,sigma,skew):
	
	c02=skew[0]/6.0
	c13=skew[0]/6.0
	c11=-skew[1]/4.0
	c24=skew[0]/6.0
	c22=-0.5*skew[1]
	c20=-0.5*skew[2]
 
	dV0=sigma[0]*(1.0/np.sqrt(2*np.pi))*np.exp(-0.5*(x**2))*(c02*h2(x))
	dV1=sigma[0]*(sigma[1]/(8*np.sqrt(2)*sigma[0]))*np.exp(-0.5*(x**2))*(c13*h3(x) + c11*x)
	dV2=sigma[0]*(((sigma[1]/sigma[0])**2)/(2*(2*np.pi)**1.5))*np.exp(-0.5*(x**2))*(c24*h4(x) + c22*h2(x) + c20)
 
	return dV0,dV1,dV2

def kurt_correction(x,sigma,skew,kurt):
 
	c05=(skew[0]**2)/72.0
	c03=kurt[0]/24.0
	c16=(skew[0]**2)/72.0
	c14=(kurt[0]-skew[0]*skew[1])/24.0
	c12=(1.0/12.0)*(kurt[1]+(3.0/8.0)*(skew[1]**2))
	c10=kurt[3]/8.0
	c27=(skew[0]**2)/72.0
	c25=(kurt[0]-2*skew[0]*skew[1])/24.0
	c23=(1.0/6.0)*(kurt[1]+0.5*skew[0]*skew[2])
	c21=0.5*(kurt[2]+0.5*skew[1]*skew[2])
 
	dV0=(sigma[0]**2)*(1.0/np.sqrt(2*np.pi))*np.exp(-0.5*(x**2))*(c05*h5(x) + c03*h3(x))
	dV1=(sigma[0]**2)*(sigma[1]/(8*np.sqrt(2)*sigma[0]))*np.exp(-0.5*(x**2))*(c16*h6(x) + c14*h4(x) - c12*h2(x) - c10)
	dV2=(sigma[0]**2)*(((sigma[1]/sigma[0])**2)/(2*(2*np.pi)**1.5))*np.exp(-0.5*(x**2))*(c27*h7(x) + c25*h5(x) - c23*h3(x) - c21*x)
 
	return dV0,dV1,dV2

def compute_cumulants(sigma,third,fourth):
 
	skew=np.zeros(3)
	kurt=np.zeros(4)
 
	skew[0] = (third[0]/(sigma[0]**4))
	skew[1] = third[1]/((sigma[1]**2)*(sigma[0]**2))
	skew[2] = 2*third[2]/(sigma[1]**4)
 
	kurt[0] = (fourth[0] - 3.0*(sigma[0]**4))/(sigma[0]**6)
	kurt[1] = (1.0/((sigma[1]**2)*(sigma[0]**4)))*(fourth[1] + 3.0*(sigma[0]**2)*(sigma[1]**2))
	kurt[2] = (1.0/((sigma[1]**4)*(sigma[0]**2)))*(2*fourth[2]+fourth[3])
	kurt[3] = (0.5/((sigma[1]**4)*(sigma[0]**2)))*(fourth[3] - 2*(sigma[1]**4))
 
	return sigma,skew,kurt

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

#######################
###Loading utilities###
#######################

def load_mink(fname,multiplicator=1.0):
 
	m = np.loadtxt(fname)
	kappa = m[:,0]
 
	V0 = [m[:,1]/multiplicator,m[:,2]/multiplicator]
	V1 = [m[:,3]/multiplicator,m[:,4]/multiplicator]
	V2 = [m[:,5]/multiplicator,m[:,6]/multiplicator]
 
	return kappa,(V0,V1,V2)

def load_stats(fname):
 
	stat = np.loadtxt(fname)
 
	mean = stat[0,0]
	var = np.sqrt(stat[1:3,0])
	third = stat[3:6,0]
	fourth = stat[6:10,0]
 
	return mean,var,third,fourth