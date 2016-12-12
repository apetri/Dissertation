import numpy as np
import numpy.fft as ft
from astropy.io import fits
cimport numpy as np
cimport cython
from libc.math cimport abs,sqrt,exp

DTYPE=np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)

cdef gauss_kernel(float t1,float t2,float p1, float p2,float filter_size_arcmin):
	return exp(-0.5*((t1-p1)**2+(t2-p2)**2)/(filter_size_arcmin**2))

def gauss_pixelized_shear_map(map_name, float num_lsst_pixels, float left_bound, float right_bound, float lsst_pixel_arcmin, float filter_size_arcmin):

	cdef long map_size = <long> ((right_bound - left_bound)/num_lsst_pixels) + 1
	cdef long n,px,py
	cdef float ker
	
	data = np.loadtxt(map_name,skiprows=1)
	cdef long Nsamples = len(data[:,0])

	cdef np.ndarray[DTYPE_t,ndim=1] x=np.zeros(Nsamples,dtype=DTYPE),y=np.zeros(Nsamples,dtype=DTYPE),g1=np.zeros(Nsamples,dtype=DTYPE),g2=np.zeros(Nsamples,dtype=DTYPE)
	cdef np.ndarray[DTYPE_t,ndim=2] shear1=np.zeros([map_size,map_size],dtype=DTYPE), shear2=np.zeros([map_size,map_size],dtype=DTYPE) 
	cdef np.ndarray[DTYPE_t,ndim=2] hits=np.zeros([map_size,map_size],dtype=DTYPE)
	
	x,y,g1,g2 = data[:,2],data[:,3],data[:,10],data[:,11]
	x *= lsst_pixel_arcmin
	y *= lsst_pixel_arcmin 
	left_bound *= lsst_pixel_arcmin
	right_bound *= lsst_pixel_arcmin

	for px in range(map_size):
		for py in range(map_size):
			for n in range(Nsamples):
				if(x[n]<=right_bound and x[n]>=left_bound and y[n]<=right_bound and y[n]>=left_bound):
					ker = gauss_kernel(x[n],y[n],px*num_lsst_pixels*lsst_pixel_arcmin,py*num_lsst_pixels*lsst_pixel_arcmin,filter_size_arcmin)
					shear1[px,py] += g1[n]*ker
					shear2[px,py] += g2[n]*ker
					hits[px,py] += ker

	return shear1/hits,shear2/hits



def pixelized_shear_map(map_name, float num_pixels, float left_bound,float right_bound):
	
	cdef long map_size = <long> ((right_bound - left_bound)/num_pixels) + 1
	cdef long n,px,py
	
	data = np.loadtxt(map_name,skiprows=1)
	cdef long Nsamples = len(data[:,0])

	cdef np.ndarray[np.int_t,ndim=2] hits=np.zeros([map_size,map_size],dtype=np.int) 
	cdef np.ndarray[DTYPE_t,ndim=1] x=np.zeros(Nsamples,dtype=DTYPE),y=np.zeros(Nsamples,dtype=DTYPE),g1=np.zeros(Nsamples,dtype=DTYPE),g2=np.zeros(Nsamples,dtype=DTYPE)
	cdef np.ndarray[DTYPE_t,ndim=2] shear1=np.zeros([map_size,map_size],dtype=DTYPE), shear2=np.zeros([map_size,map_size],dtype=DTYPE) 
	
	x,y,g1,g2 = data[:,2],data[:,3],data[:,10],data[:,11]
	
	for n in range(Nsamples):
		if(x[n]<=right_bound and x[n]>=left_bound and y[n]<=right_bound and y[n]>=left_bound):
			
			px = <long>((x[n]-left_bound)/num_pixels)
			py = <long>((y[n]-left_bound)/num_pixels)
			hits[px,py] += 1
			shear1[px,py] += g1[n]
			shear2[px,py] += g2[n]
	
	for px in range(map_size):
		for py in range(map_size):
			
			if(hits[px,py] != 0):
				shear1[px,py] = shear1[px,py]/hits[px,py]
				shear2[px,py] = shear2[px,py]/hits[px,py]
	
	return shear1,shear2 

def eb_decompose(shear1,shear2,pixel_area):

	cdef long i,j
	cdef float kx,ky
	cdef long map_size=shear1.shape[0]
	cdef long freq_size = map_size/2 + 1
	cdef np.ndarray[DTYPE_t,ndim=1] freq=ft.fftfreq(map_size)
	
	cdef np.ndarray[np.complex_t,ndim=2] ft_shear1=np.zeros([map_size,freq_size],dtype=np.complex),ft_shear2=np.zeros([map_size,freq_size],dtype=np.complex)
	cdef np.ndarray[np.complex_t,ndim=2] e_mode=np.zeros([map_size,freq_size],dtype=np.complex),b_mode=np.zeros([map_size,freq_size],dtype=np.complex)
	
	ft_shear1 = ft.rfft2(shear1) * pixel_area
	ft_shear2 = ft.rfft2(shear2) * pixel_area
	
	for i in range(map_size):
		for j in range(freq_size):
			
			if(i!=0 or j!=0):
				
				kx = freq[i]
				ky = freq[j]
				
				e_mode[i,j].real = ((kx*kx-ky*ky)/(kx*kx+ky*ky))*ft_shear1[i,j].real + (2*(kx*ky)/(kx*kx+ky*ky))*ft_shear2[i,j].real
				e_mode[i,j].imag = ((kx*kx-ky*ky)/(kx*kx+ky*ky))*ft_shear1[i,j].imag + (2*(kx*ky)/(kx*kx+ky*ky))*ft_shear2[i,j].imag
				b_mode[i,j].real = ((kx*kx-ky*ky)/(kx*kx+ky*ky))*ft_shear2[i,j].real - (2*(kx*ky)/(kx*kx+ky*ky))*ft_shear1[i,j].real
				b_mode[i,j].imag = ((kx*kx-ky*ky)/(kx*kx+ky*ky))*ft_shear2[i,j].imag - (2*(kx*ky)/(kx*kx+ky*ky))*ft_shear1[i,j].imag
				
			else:
			
				e_mode[i,j].real = 0.0
				b_mode[i,j].real = 0.0
				e_mode[i,j].imag = 0.0
				b_mode[i,j].imag = 0.0
				
	
	return freq,e_mode,b_mode

def power_spectrum(spectrum_file):

	cdef long i,j,num_freq
	cdef float lmin

	hdulist = fits.open(spectrum_file)
	num_freq = hdulist[0].header['NAXIS1']
	map_size = hdulist[0].header['NAXIS2']
	lmin = hdulist[0].header['LMIN']

	cdef np.ndarray[DTYPE_t,ndim=3] eb_modes = hdulist[0].data.astype(DTYPE), two_d_spectrum = np.zeros([map_size,num_freq,3],dtype=DTYPE)
	cdef np.ndarray[DTYPE_t,ndim=1] ell = lmin * map_size * ft.fftfreq(map_size)
	cdef np.ndarray[DTYPE_t,ndim=2] one_d_spectrum = np.zeros([map_size*num_freq,4],dtype=DTYPE)

	hdulist.close()

	for i in range(map_size):
		for j in range(num_freq):
			one_d_spectrum[num_freq*i + j,0] = sqrt(ell[i]*ell[i] + ell[j]*ell[j])
			one_d_spectrum[num_freq*i + j,1] = eb_modes[0,i,j] * eb_modes[0,i,j] + eb_modes[1,i,j] * eb_modes[1,i,j]
			one_d_spectrum[num_freq*i + j,2] = eb_modes[2,i,j] * eb_modes[2,i,j] + eb_modes[3,i,j] * eb_modes[3,i,j]
			one_d_spectrum[num_freq*i + j,3] = eb_modes[0,i,j] * eb_modes[2,i,j] + eb_modes[1,i,j] * eb_modes[3,i,j]
			
			two_d_spectrum[i,j,0] = one_d_spectrum[num_freq*i + j,1]
			two_d_spectrum[i,j,1] = one_d_spectrum[num_freq*i + j,2]
			two_d_spectrum[i,j,2] = one_d_spectrum[num_freq*i + j,3]
	
	one_d_spectrum[:,1:] = one_d_spectrum[:,1:] * (lmin/(2*np.pi))**2
	two_d_spectrum = two_d_spectrum * (lmin/(2*np.pi))**2

	return one_d_spectrum,two_d_spectrum 

def bin_power_spectrum(np.ndarray[DTYPE_t,ndim=2] power, np.ndarray[DTYPE_t,ndim=1] bins):

	cdef long i,j
	cdef long Nbins = len(bins), Npoints = len(power[:,0])
	cdef np.ndarray[np.int_t,ndim=1] hits = np.zeros(Nbins-1,dtype=np.int)
	cdef np.ndarray[DTYPE_t,ndim=1] ell = np.zeros(Nbins-1,dtype=DTYPE)
	cdef np.ndarray[DTYPE_t,ndim=2] binned_power = np.zeros([Nbins-1,3],dtype=DTYPE)
	
	for k in range(Nbins-1):
		ell[k] = (bins[k+1] + bins[k])/2.0

	for i in range(Npoints):
		for j in range(0,Nbins-1):

			if(power[i,0]<=bins[j+1] and power[i,0]>bins[j]):
				hits[j] += 1 
				binned_power[j,0] += power[i,1]
				binned_power[j,1] += power[i,2]
				binned_power[j,2] += power[i,3]
	
	binned_power[:,0][hits>0] = binned_power[:,0][hits>0]/hits[hits>0]
	binned_power[:,1][hits>0] = binned_power[:,1][hits>0]/hits[hits>0]
	binned_power[:,2][hits>0] = binned_power[:,2][hits>0]/hits[hits>0]

	return ell,binned_power

				



	
				
	
	