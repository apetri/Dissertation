import sys,os

import numpy as np
import matplotlib.pyplot as plt

from lenstools.pipeline.simulation import LensToolsCosmology

####################
####Book keeping####
####################

data_path = "/Users/andreapetri/Documents/Columbia/spurious_shear"
cosmo_parameters = ["Om","Ol","w","ns","si"]
cosmo_legend = {"Om":"Om0","Ol":"Ode0","w":"w0","ns":"ns","si":"sigma8"}

fiducial = LensToolsCosmology(sigma8=0.798)
variations = [ LensToolsCosmology(Om0=0.29,Ode0=0.71,sigma8=0.798), LensToolsCosmology(w0=-0.8,sigma8=0.798), LensToolsCosmology(sigma8=0.850) ]