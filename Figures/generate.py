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
import photozProject

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
method["5-pspdf"] = covarianceProject.ps_pdf
method["5-scalingnr"] = covarianceProject.scaling_nr
method["5-scalingns"] = covarianceProject.scaling_ns
method["5-meansns"] = covarianceProject.means_nsim
method["5-pca1"] = photozProject.pca_components_power_spectrum
method["5-pca2"] = photozProject.pca_components_peaks
method["5-pca3"] = photozProject.pca_components_moments
method["5-biasng"] = bornProject.pbBiasNgal

#########################################################################

method["6-design"] = cfhtProject.design
method["6-emulator"] = cfhtProject.emulatorAccuracy
method["6-pca"] = cfhtProject.pca
method["6-pcarobustness"] = cfhtProject.robustness
method["6-csingle"] = cfhtProject.contours_single
method["6-cmoments"] = cfhtProject.contour_moments

#########################################################################

method["7-eb2d"] = spuriousProject.ebPlot
method["7-spfit"] = spuriousProject.ebFit

#Main
def main():
	cmd_args = parser.parse_args()

	for fig in cmd_args.fig:
		
		try:
			int(fig)
			for f in filter(lambda s:s.startswith(fig),method):
				print(f)
				method[f](cmd_args)
		except:
			method[fig](cmd_args)

	if not(len(cmd_args.fig)):
		for l in sorted(method):
			print("{0} ---> {1}".format(l,method[l].__name__))

if __name__=="__main__":
	main()