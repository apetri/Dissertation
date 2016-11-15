import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

import lenstools as lt

def distortion(cmd_args,fontsize=22):

	#Set up plot
	fig,ax = plt.subplots(2,2,figsize=(16,16))

	#Distortion calculation
	transf = lambda theta,off,J:((theta.T-off).dot(J)+off).T

	#Original image
	b = np.arange(256.)
	g = np.array(np.meshgrid(b,b,indexing="xy"))
	beta = lt.ConvergenceMap(np.exp(-((g[0]-128)**2+(g[1]-128)**2)/(50.**2)),angle=5*u.arcsec)
	theta = g*beta.resolution.to(u.arcsec)

	#Original
	beta.visualize(fig=fig,ax=ax[0,0])
	ax[0,0].set_title("Original",fontsize=fontsize)

	#Distortion2: gamma1
	J = np.array([[0.75,0],[0,1.25]])   
	dist = lt.ConvergenceMap(beta.getValues(*transf(theta,beta.side_angle/2,J)),angle=beta.side_angle)
	dist.visualize(fig=fig,ax=ax[0,1])
	ax[0,1].set_title(r"$\gamma^1=0.25$",fontsize=fontsize)

	#Distortion3: gamma2
	J = np.array([[1.0,-0.25],[-0.25,1.0]])   
	dist = lt.ConvergenceMap(beta.getValues(*transf(theta,beta.side_angle/2,J)),angle=beta.side_angle)
	dist.visualize(fig=fig,ax=ax[1,0])
	ax[1,0].set_title(r"$\gamma^2=0.25$",fontsize=fontsize)

	#Distortion4: gamma1, omega
	J = np.array([[0.75,-0.25],[0.25,1.25]])   
	dist = lt.ConvergenceMap(beta.getValues(*transf(theta,beta.side_angle/2,J)),angle=beta.side_angle)
	dist.visualize(fig=fig,ax=ax[1,1])
	ax[1,1].set_title(r"$\gamma^1=0.25,\omega=0.25$",fontsize=fontsize)

	#Save 
	fig.savefig("{0}/distortion.{0}".format(cmd_args.type))

def EMode(cmd_args,fontsize=22):

	#Set up plot
	fig,ax = plt.subplots()

	#Set up image
	b = np.arange(512.)
	g = np.array(np.meshgrid(b,b,indexing="xy"))
	kappa = lt.ConvergenceMap(0.1*np.exp(-((g[0]-128)**2+(g[1]-256)**2)/(50.**2))-0.1*np.exp(-((g[0]-384)**2+(g[1]-256)**2)/(50.**2)),angle=50*u.arcsec)

	#KS inversion for shear
	gamma = lt.ShearMap.fromConvergence(kappa)

	#Visualize
	kappa.visualize(colorbar=True,fig=fig,ax=ax,cbar_label=r"$\kappa$")
	gamma.sticks(fig=fig,ax=ax,pixel_step=15)

	#Save
	fig.savefig("{0}/emode.{0}".format(cmd_args.type))