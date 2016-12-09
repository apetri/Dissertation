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

method["3-csample"] = bornProject.convergenceVisualize
method["3-ltflow"] = ltProject.flow
method["3-ltmemory"] = ltProject.memory_usage

#########################################################################

method["4-excursion"] = bornProject.excursion
method["4-minkpert"] = minkowskiProject.minkPerturbation

#########################################################################

method["7-ebplot"] = spuriousProject.ebPlot

#Main
def main():
	cmd_args = parser.parse_args()
	for fig in cmd_args.fig:
		method[fig](cmd_args)

if __name__=="__main__":
	main()