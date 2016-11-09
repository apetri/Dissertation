#!/usr/bin/env python
import sys,argparse
sys.modules["mpi4py"] = None

#Pretty plotting
import seaborn as sns
sns.set(palette="muted",font_scale=2)

#Projects I worked on
import miscellaneous
import covarianceProject

#Options
parser = argparse.ArgumentParser()
parser.add_argument("-t","--type",dest="type",default="png",help="format of the figure to save")
parser.add_argument("fig",nargs="*")

########################################################################

#Method dictionary
method = dict()
method["2-dist"] = miscellaneous.distortion

#Main
def main():
	cmd_args = parser.parse_args()
	for fig in cmd_args.fig:
		method[fig](cmd_args)

if __name__=="__main__":
	main()