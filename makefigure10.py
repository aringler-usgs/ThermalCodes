###This code creates a colormap based on a theoretical model of temperature attenuation
###with depth (Turcotte and Schubert, 2002). This corresponds to Figure 10.



#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import matplotlib as mpl
from matplotlib import cm
import matplotlib.colors as colors


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
    
    
#Set font parameters using matplotlib
mpl.rc('font',family='serif')
mpl.rc('font',serif='Times') 
mpl.rc('text', usetex=True)
mpl.rc('font',size=24)

#Setting our temperature and distance steps that dictate the resolution of our colormap.
dT, dy = 0.1, 0.1

#Set the initial temperature, frequency (in our case, one day), and the thermal diffusivities of the two 
#materials we have chosen (thermal diffusivities from Robertson, 1988).
T0 = 0.
w = 24.*60.*60.
k1 = 0.0000003 #Sandy Soil
k2 = 0.0000016 #Granite

#Creates a mesh of temperature and depth that have the dimensions of our plot, and go in steps
#of dT and dy that we defined above.
depth, temp = np.mgrid[slice(10., 100.+ dy, dy),
                slice(0., 40. + dT, dT)]


#Calculates the relative temperature at every depth for every surface temperature
#variation for each type of material. 
T1 = T0 + temp*(np.exp(-depth*np.sqrt(w/2*k1)))
T2 = T0 + temp*(np.exp(-depth*np.sqrt(w/2*k2)))

#Removes the last row and column from T1
T1 = T1[:-1, :-1]

#Turns temperature into an RMS so that it is time-independent.
T1 = T1/(np.sqrt(2))

#Sets minimum and maximum temperatures.
T1_min, T1_max = -np.abs(T1).max(), np.abs(T1).max()

#Same calculations as above, but for T2.
T2 = T2[:-1, :-1]
T2 = T2/(np.sqrt(2))
T2_min, T2_max = -np.abs(T2).max(), np.abs(T2).max()

cmap = plt.get_cmap('viridis')
plsc = truncate_colormap(cmap, 0.0, 0.9)


fig = plt.figure(1,figsize=(16,16))
fig.text(0.5, 0.93,'Surface Temperature Variation $({}^{\circ}C RMS)$', ha='center', va='center')

#We normalize our color scale so that when we pick the color limits later, part of the colormap that
#we chose is not cut off.
norm = mpl.colors.Normalize(vmin= 0.0, vmax = 0.01, clip = False)

ax1 = fig.add_subplot(121)

#Plots temperature, depth, and relative temperature that still needs to be attenuated.
att = ax1.pcolormesh(temp, depth, T1, cmap=plsc, norm=norm, vmin=T1_min, vmax=T1_max)

#Plots a white contour line where the relative temperature left to attenuate is 0.002degC.
CS = plt.contour(temp[:-1,:-1], depth[:-1,:-1], T1, [0.002], colors = ('w'), linewidth = (3))

#Sets axes limits.
plt.axis([temp.min(), temp.max(), depth.max(),depth.min()])
ax1.xaxis.set_ticks_position('top')
#Plots labels and y axis label for T1.
plt.ylabel('Depth Below Surface $(m)$')
plt.text(1.,12., 'A)', color = 'k')
plt.text(0.5,0.05,'Sandy Soil', horizontalalignment = 'center', verticalalignment = 'center', transform = ax1.transAxes, color = 'w')

#Same process as above, but for T2.
ax1 = fig.add_subplot(122)

att = ax1.pcolormesh(temp, depth, T2, cmap=plsc, norm = norm, vmin=T2_min, vmax=T2_max)

CS = plt.contour(temp[:-1,:-1], depth[:-1,:-1], T2, [0.002], colors = ('w'), linewidth = (3))

plt.axis([temp.min(), temp.max(), depth.max(),depth.min()])
ax1.xaxis.set_ticks_position('top')
plt.text(1.,12., 'B)', color = 'k')
plt.text(0.5,0.05,'Granite', horizontalalignment = 'center', verticalalignment = 'center', transform = ax1.transAxes, color = 'w')

#Adjusts subplots so that they don't squished by the colorbar.
fig.subplots_adjust(right = 0.8)

#Adding in a colorbar and setting its label and limits. 
cbar_ax = fig.add_axes([0.85, 0.15,0.05, 0.7])
cb = fig.colorbar(att, cax = cbar_ax)
cb.set_label('Relative Temperature $({}^{\circ}C RMS)$')
color_limits = (0, 0.01)
att.set_clim((0.,0.01))
cb.set_clim((0.,0.01))


plt.savefig('Tatten.pdf',format='PDF',dpi=400)
plt.savefig('Tatten.jpg',format='JPEG', dpi=400)
plt.clf()
plt.close()
