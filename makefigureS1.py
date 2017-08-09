###This is the code for the supplemental figure that calculates the peak
###to peak temperature change in the ASL cross-tunnel.


#!/usr/bin/env python

from obspy.core import UTCDateTime, read
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

#Import font parameters from matplotlib
mpl.rc('font',family='serif')
mpl.rc('font',serif='Times') 
mpl.rc('text', usetex=True)
mpl.rc('font',size=20)

#Set our day range for pulling data
days = range(034, 054)

#Reads in the data from one temperature sensor in the vault for the 20 day period chosen.
for day in days:
    try:
        string = '/tr1/telemetry_days/'+ 'XX' + '_' + 'TST6' + '/' + '2017' \
                + '/' + '2017' +'_' + str(day).zfill(3) + '/' + 'WV' + '_' + 'LK2' + '*'
        sttemp = read(string)
        sttemp.merge()
        sttemp.detrend('constant')
    except:
        #If there is a data gap on any of the days, that day gets printed.
        print(str(day).zfill(3) + 'bad')
        
#Creates an array of times in hours.
t = (np.arange(0,len(sttemp[0].data))/sttemp[0].stats.sampling_rate)/(60*60)

#Does a 1000-point moving average on the data to remove transient pulsing.
N= 1000
sttemp =np.convolve(sttemp[0].data, np.ones((N,))/N, mode='same')

figure = plt.figure(figsize=(12,12))

#Plots and saves the figure. The 335544. counts/deg C is the sensitivity
#of the temperature sensors.
plt.plot(t, (sttemp/335544.)) 
plt.xlabel('Time (Days)')
plt.ylabel('Temperature $({}^{\circ} C)$')
plt.xlim([0,24])
plt.savefig('vault.jpg', format= 'JPEG', dpi = 400)
