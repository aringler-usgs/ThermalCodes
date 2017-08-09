#!/usr/bin/env python


###This code pulls earth tide data from IRIS/USGS network station TUC (Tucson,
###Arizona) and separates the semi-diurnal and diurnal measurements of three 
###seismometers at TUC (one in a foam box, one in water bricks, and one in a 1.7m
###post-hole. A power spectra is also plotted to compare the difference in amplitude
###between each sensor for both tides.

from obspy.core import read, UTCDateTime, Stream
from obspy.io.xseed import Parser
import matplotlib.pyplot as plt
import numpy as np
from obspy.signal.invsim import evalresp
from matplotlib.mlab import csd
from obspy.signal.spectral_estimation import get_nlnm, get_nhnm

#Font parameters from matplotlib
import matplotlib as mpl
mpl.rc('font',family='serif')
mpl.rc('font',serif='Times') 
mpl.rc('text', usetex=True)
mpl.rc('font',size=18)

#Function for calculating PSD measurements
def cp(tr1,tr2,lenfft,lenol,delta):
    sr = 1/delta
    cpval,fre = csd(tr1.data,tr2.data,NFFT=lenfft,Fs=sr,noverlap=lenol,scale_by_freq=True)
    fre = fre[1:]
    cpval = cpval[1:]
    return cpval, fre  

#Set the length of our Fast Fourier Transform to be the same length as our data.
lenfft = 51*24*60*60
#Set overlap to 0 to get power spectra, not PSD
lenol = 0.
delta = 1.    

#Read in data for all three sensors and their temperature sensors in 
#the day range we specify.
st = Stream()
for days in range(100,151):
    st += read('/msd/IU_TUC/2017/' + str(days).zfill(3) + '/*LHZ*')
st.merge()


fig = plt.figure(figsize=(12,16)) 

#Set minimum and maximum frequencies for bandpass filtering for the semi-
#diurnal tide.
fM=1./(10.*60.*60)
fm =1./(13*60*60)

#Set minimum and maximum frequencies for bandpass filtering for the
#diurnal tide.
fm1 = 1./(25*60*60)
fM1 = 1./(22*60*60)

stc = st.copy()

#Create an array of time in days for plotting time series data.
t = (np.arange(0,len(stc[0].data))/stc[0].stats.sampling_rate)/(24*60*60)

#Create first subplot for time series of semi-diurnal tides.
ax = fig.add_subplot(3, 1, 1)

for tr in stc.select(channel='LH*'):
        #Demean data
        tr.detrend('constant')
        #Bandpass filter around the semi-diurnal tide
        tr.filter('bandpass',freqmin=fm, freqmax=fM, corners=1)
        #Normalize the data between all three sensors
        tr.normalize()
        #Plot
        ax.plot(t,tr.data)
        plt.xlim([0,51])
        plt.xlabel('Time (Days)')
        plt.ylabel('Counts (Normalized)')
        plt.text(0.5,0.9, 'A)')

#Create second subplot for time series of diurnal tides
ax = fig.add_subplot(3, 1, 2)

std = st.copy()
        
for tr in std.select(channel='LH*'):
        #Demean data
        tr.detrend('constant')
        #Bandpass filter around diurnal tide
        tr.filter('bandpass',freqmin=fm1, freqmax=fM1, corners=1)
        #Normalize data between all three sensors
        tr.normalize()
        #Plot
        ax.plot(t,tr.data)
        plt.xlim([0,51])
        plt.xlabel('Time (Days)')
        plt.ylabel('Counts (Normalized)')
        plt.text(0.5,0.9, 'B)')

#Create third subplot for power spectra        
ax = fig.add_subplot(3, 1, 3) 
    
for tr in st.select(channel='LH*'):
    #Retrieve the response data for each sensor in units of acceleration
    seedresp ={'filename' : '/APPS/metadata/RESPS/RESP.' + tr.id, 'date': tr.stats.starttime, 'units': 'ACC'}
    print(seedresp)
    print(tr)
    #Calculate the power spectra
    p,f = cp(tr,tr,lenfft, lenol, delta)
    #Evaluate the response data
    resp = evalresp(1.0,lenfft,seedresp['filename'], seedresp['date'], units = 'ACC')
    #Remove the first value of the response since we want units of acceleration.
    resp = resp[1:]
    #Remove the response of the instruments from the power spectra
    p = p/np.abs(resp**2)
    #Convert Hz to mHz
    f = f*1000.
    #Setting labels for each sensor based on their trace id.
    if tr.id == 'IU.TUC.10.LHZ':
        label = 'STS-2 (Foam Box)'
    elif tr.id == 'IU.TUC.60.LHZ':
        label = 'STS-6 (Post-hole)'
    else:
        label = 'STS-1 (Water Bricks)' 
    #Plot
    ax.semilogy(f,p, label = label)
    plt.xlim([0.005, 0.03])
    plt.ylim([10**-18, 10**-4])
    plt.xlabel('Frequency (mHz)')
    plt.ylabel('Power ($(m/s^2)^2/Hz$)')
    plt.text(0.0052, 10**(-5), 'C)')
    
#Import NLNM and NHNM
model_periods, high_noise = get_nhnm()
model_periods, low_noise = get_nlnm()
#Turn period into frequency, and then Hz into mHz
model_freq = 1000/model_periods
#Plots the NLNM and NLHM on the third plot.
plt.semilogy(model_freq, 10**(high_noise/10), color = 'k', linewidth = 2)
plt.semilogy(model_freq, 10**(low_noise/10), color = 'k', linewidth = 2)     
#Plot the legend centerd outside of the plot, with 3 columns.
plt.legend(loc = 'center', bbox_to_anchor=(0.5, -0.31), ncol = 3, fontsize = 20)

plt.savefig('tidecomp.jpg', format= 'JPEG', dpi = 400)
plt.clf()
plt.close()
    
    

    








