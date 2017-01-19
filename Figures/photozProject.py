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
batch = SimulationBatch.current("/Users/andreapetri/Documents/Columbia/Simulations/LSST100Parameters/environment.ini")