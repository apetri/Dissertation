import re

import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt

from matplotlib import rc
import daft

#Pipeline flow
def flow(cmd_args):

	rc("font", family="serif", size=12)
	rc("text", usetex=False)

	r_color = {"ec" : "red"}
	g_color = {"ec" : "green"}

	#Instantiate PGM
	pgm = daft.PGM([16,7],origin=[0,0])

	#Nodes
	pgm.add_node(daft.Node("parameters","Parameters",1,2.5,aspect=3.,plot_params=r_color))
	pgm.add_node(daft.Node("geometry","geometry+seeds",4,2.5,aspect=4.))
	
	#ICS
	pgm.add_node(daft.Node("ic1","IC seed 1",6.5,3.25,aspect=2.5))
	pgm.add_node(daft.Node("ic2","IC seed 2",6.5,2.5,aspect=2.5))
	pgm.add_node(daft.Node("icN",r"IC seed $N$",6.5,1.75,aspect=2.5))

	#Evolution
	pgm.add_node(daft.Node("e1",r"$\delta^1(\mathbf{x},z)$",8.5,3.25,aspect=3.))
	pgm.add_node(daft.Node("e2",r"$\delta^2(\mathbf{x},z)$",8.5,2.5,aspect=3.))
	pgm.add_node(daft.Node("eN",r"$\delta^N(\mathbf{x},z)$",8.5,1.75,aspect=3.))

	#Lens Planes
	pgm.add_node(daft.Node("p1",r"Lens: $\Phi^1(\mathbf{x},z)$",10.5,3.25,aspect=3.5))
	pgm.add_node(daft.Node("p2",r"Lens: $\Phi^2(\mathbf{x},z)$",10.5,2.5,aspect=3.5))
	pgm.add_node(daft.Node("pN",r"Lens: $\Phi^N(\mathbf{x},z)$",10.5,1.75,aspect=3.5))

	#Mix planes
	pgm.add_plate(daft.Plate([9.4,1.0,2.1,3.0],label="Mix seeds"))

	#Lensing maps
	pgm.add_node(daft.Node("lens","Lensing maps " + r"$(\kappa,\gamma)$",13.0,2.5,aspect=4.5,plot_params=g_color))

	#Executables
	pgm.add_node(daft.Node("camb","CAMB+NGen-IC",4,0.5,aspect=4.5,observed=True))
	pgm.add_node(daft.Node("gadget","Gadget2",6.5,0.5,aspect=2.,observed=True))
	pgm.add_node(daft.Node("planes","lenstools.planes",8.5,0.5,aspect=4.,observed=True))
	pgm.add_node(daft.Node("ray","lenstools.raytracing",10.5,4.5,aspect=5.,observed=True))

	#Edges
	pgm.add_edge("parameters","geometry")
	pgm.add_edge("geometry","ic1")
	pgm.add_edge("geometry","ic2")
	pgm.add_edge("geometry","icN")
	pgm.add_edge("ic1","e1")
	pgm.add_edge("ic2","e2")
	pgm.add_edge("icN","eN")
	pgm.add_edge("e1","p1")
	pgm.add_edge("e2","p2")
	pgm.add_edge("eN","pN")
	pgm.add_edge("p2",'lens')
	pgm.add_edge("camb","geometry")
	pgm.add_edge("gadget","icN")
	pgm.add_edge("planes","eN")
	pgm.add_edge("ray","p1")

	#Render and save
	pgm.render()
	pgm.figure.savefig("{0}/lt_flow.{0}".format(cmd_args.type))

#Plot vertical line
def _bline(ax,x,top,**kwargs):
	down,up = ax.get_ylim()
	xp = np.ones(100)*x
	yp = np.linspace(down,top,100)
	ax.plot(xp,yp,**kwargs)

def _tline(ax,x,bottom,**kwargs):
	down,up = ax.get_ylim()
	xp = np.ones(100)*x
	yp = np.linspace(bottom,up,100)
	ax.plot(xp,yp,**kwargs)

#Parse logs into a pandas DataFrame
def parse_log(fp):

	#Regex to parse lines
	linefmt = re.compile(r"([0-9\-\:\.\s]+)\:lenstools\.stderr\:(INFO|DEBUG)\:(.+)\:[a-zA-Z\s]*([0-9\.]+) Gbyte \(task\)")
	
	#Keep track of this information
	timestamp = list()
	log_level = list()
	step_type = list()
	peak_memory = list()

	#Cycle through the lines
	for line in fp.readlines():
		match = linefmt.search(line)
		if match:
			match_results = match.groups()
			timestamp.append(pd.to_datetime(_fill(match_results[0]),format="%m-%d %H:%M:%S.%f"))
			log_level.append(match_results[1])
			step_type.append(match_results[2])
			peak_memory.append(float(match_results[3]))

	#Construct the DataFrame
	df = pd.DataFrame.from_dict({"timestamp":timestamp,"level":log_level,"step":step_type,"peak_memory(GB)":peak_memory})
	df["delta_timestamp"] = df.timestamp.diff()
	dt_s = df.delta_timestamp.values.astype(np.float) / 1.0e9
	dt_s[0] = 0
	df["delta_timestamp_s"] = dt_s
	df["timestamp_s"] = dt_s.cumsum()

	#Return to user
	return df

#Fill milliseconds
def _fill(s):
	last = s.split('.')[-1]
	nzeros = 3-len(last)
	if nzeros:
		return s.replace('.'+last,'.'+'0'*nzeros+last)
	else:
		return s

#Memory usage
def memory_usage(cmd_args):

	#Setup plot
	fig,ax = plt.subplots()

	#Plot memory usage for planes operations
	with open("planes.err","r") as fp:
		df_planes = parse_log(fp)

	ax.plot(df_planes["timestamp_s"].values,df_planes["peak_memory(GB)"].values,label="Lens Planes",color="black")
	ax.set_ylim(0,2.)

	#Plot a black line after each plane is completed
	planes_completed = df_planes.timestamp_s[df_planes.step.str.contains("Plane")].values
	planes_completed_memory = df_planes["peak_memory(GB)"][df_planes.step.str.contains("Plane")].values
	for n,t in enumerate(planes_completed):
		_bline(ax,t,planes_completed_memory[n],color="black",linestyle="--")

	#Plot memory usage for raytracing operations
	with open("ray.err","r") as fp:
		df_ray = parse_log(fp)

	ax_top = ax.twiny()
	ax_top.plot(df_ray["timestamp_s"].values,df_ray["peak_memory(GB)"].values,label="Ray--tracing",color="red")
	ax_top.set_xscale("log")
	ax_top.spines["top"].set_color("red")
	ax_top.tick_params(axis="x",colors="red")

	#Plot a red line after each lens crossing
	lens_crossed = df_ray.timestamp_s[df_ray.step.str.contains("Lens")].values
	lens_crossed_memory = df_ray["peak_memory(GB)"][df_ray.step.str.contains("Lens")].values
	for n,t in enumerate(lens_crossed):
		_tline(ax_top,t,lens_crossed_memory[n],color="red",linestyle="--")

	#Labels
	ax.set_xlabel(r"${\rm Runtime(s)}$",fontsize=22)
	ax.set_ylabel(r"${\rm Peak\,\, memory(GB)}$",fontsize=22)

	#Save
	fig.savefig("{0}/lt_memory_usage.{0}".format(cmd_args.type))