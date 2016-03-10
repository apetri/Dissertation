#!/usr/bin/env python

import sys,argparse
sys.modules["mpi4py"] = None

import numpy as np
import matplotlib.pyplot as plt

#Options
parser = argparse.ArgumentParser()
parser.add_argument("-t","--type",dest="type",default="png",help="format of the figure to save")
parser.add_argument("fig",nargs="*")


########################################################################

#Method dictionary
method = dict()

#Main
def main():
	cmd_args = parser.parse_args()
	for fig in cmd_args.fig:
		method[fig](cmd_args)

if __name__=="__main__":
	main()