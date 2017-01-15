#!/usr/bin/env python
import sys,argparse
sys.modules["mpi4py"] = None

#Pretty plotting
import seaborn as sns
sns.set(palette="muted",font_scale=2)

#Projects I worked on
import miscellaneous
import covarianceProject
import bornProject
import ltProject
import minkowskiProject
import spuriousProject
import cfhtProject

#Options
parser = argparse.ArgumentParser()
parser.add_argument("-t","--type",dest="type",default="png",help="format of the figure to save")
parser.add_argument("fig",nargs="*")

########################################################################

#Method dictionary
method = dict()

#########################################################################

method["1-distred"] = miscellaneous.distred
method["1-growth"] = miscellaneous.growth

#########################################################################

method["2-dist"] = miscellaneous.distortion
method["2-emode"] = miscellaneous.EMode

#########################################################################

method["3-plane"] = ltProject.planeVisualize
method["3-csample"] = bornProject.convergenceVisualize
method["3-ltflow"] = ltProject.flow
method["3-ltmemory"] = ltProject.memory_usage

#########################################################################

method["4-excursion"] = bornProject.excursion
method["4-minkpert01"] = minkowskiProject.minkPerturbation01
method["4-minkpert2"] = minkowskiProject.minkPerturbation2
method["4-minkseries"] = minkowskiProject.seriesConvergence
method["4-peaks"] = bornProject.convergencePeaks
method["4-powerSample"] = bornProject.powerSample
method["4-powerCov"] = bornProject.powerCov
method["4-powerRes"] = bornProject.powerResiduals
method["4-skewRes"] = bornProject.plotSmoothSkew
method["4-kurtRes"] = bornProject.plotSmoothKurt

#########################################################################

method["5-curvingnb"] = covarianceProject.curving_nb

#########################################################################

method["7-eb2d"] = spuriousProject.ebPlot
method["7-spfit"] = spuriousProject.ebFit

#Main
def main():
	cmd_args = parser.parse_args()

	for fig in cmd_args.fig:
		method[fig](cmd_args)

	if not(len(cmd_args.fig)):
		for l in sorted(method):
			print("{0} ---> {1}".format(l,method[l].__name__))

if __name__=="__main__":
	main()