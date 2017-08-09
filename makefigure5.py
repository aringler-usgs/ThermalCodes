### This code calculates the PSD using a modified Welch method (Welch, 1967; Oppenheim and Schafer, 1975)
### and the incoherent self-noise using the three-sensor method (Sleeman, 2006). This code corresponds to
### Figure 5.



#!/usr/bin/env python

from obspy.core import read, Stream, UTCDateTime
import glob
from obspy.signal.invsim import paz_to_freq_resp, evalresp
import numpy as np
from matplotlib.mlab import csd
import math
import matplotlib.pyplot as plt
from obspy.signal.spectral_estimation import get_nlnm, get_nhnm
import sys
from obspy.io.xseed import Parser
from matplotlib import cm


#Setting font parameters from matplotlib
import matplotlib as mpl
mpl.rc('font',family='serif')
mpl.rc('font',serif='Times') 
mpl.rc('text', usetex=True)
mpl.rc('font',size=30)



#Set window length, overlap, and delta value.
lenfft = 2**14
lenol = 2**11
delta = 1

#List of lettering to use later when plotting.
letters = ['A) STS-2.5', 'B) Trillium 120', 'C) Trillium Compact','D)', 'E)','F)']



#Defining the poles and zeros for each sensor to be used when calculating the response. Poles and 
#zeros taken from IRIS.
def returnresp(sta, loc):
    if loc == '10':
        #paz data for STS-2.5
        paz = {'zeros': [0., 0., -15.708, -15.708, -630.2, -556.-936.19j, -556.+936.19j,-973.8],
            'poles': [-0.03702 + 0.03702j, -0.03702 - 0.03701j, -16.041, -16.041, -327.3 -74.14j,
                    -327.3+74.14j, -97], 'gain': 0.000157286, 'sensitivity': 1500.*(2**26)/40.}
    elif sta == 'TST5':
        #paz data for STS-2.5
        paz = {'zeros': [0., 0., -15.708, -15.708, -630.2, -556.-936.19j, -556.+936.19j,-973.8],
                'poles': [-0.03702 + 0.03702j, -0.03702 - 0.03701j, -16.041, -16.041, -327.3 -74.14j,
                        -327.3+74.14j, -97], 'gain': 0.000157286, 'sensitivity': 1500.*(2**26)/40.}
    elif sta == 'TST6':
        #paz data for Trillium Compact
        paz = {'zeros': [0., 0., -392.0, -1960.0, -1490.0 + 1740.0j, -1490.0 - 1740.0j],
                'poles': [-0.03691 + 0.03702j, -0.03691 - 0.03702j, -343.0, -370.0 + 467.0j, -370.0 - 467.0j, -836.0 + 1522.0j,
                        -836.0 - 1522.0j, -4900.0 + 4700.0j, -4900.0 - 4700.0j, -6900.0, -15000.0], 'gain': 4.344928*(10**17), 
                        'sensitivity': 754.3*(2**26)/40.}
    elif sta == 'TST4':
        #paz data for Trillium 120
        paz = {'gain': 8.318710*10**17, 'zeros': [0 + 0j, 0 + 0j, -31.63 + 0j, 
                -160.0 + 0j, -350.0 + 0j, -3177.0 + 0j], 'poles':[-0.036614 + 0.037059j,  
                -0.036614 - 0.037059j, -32.55 + 0j, -142.0 + 0j, -364.0  + 404.0j, 
                -364.0 - 404.0j, -1260.0 + 0j, -4900.0 + 5204.0j, -4900.0 - 5204.0j, 
                -7100.0 + 1700.0j, -7100.0 - 1700.0j], 'sensitivity': 1200.*(2**26)/40.}
                
    return paz

#Function for the PSD calculation
def cp(tr1,tr2,lenfft,lenol,delta):
    sr = 1/delta
    cpval,fre = csd(tr1.data,tr2.data,NFFT=lenfft,Fs=sr,noverlap=lenol,scale_by_freq=True)
    fre = fre[1:]
    cpval = cpval[1:]
    return cpval, fre  

#Function for the incoherent self-noise calculation
def selfnoise(st, pazs):
    
    #Calculate the response of each instrument.
    resp0 = computeresp(pazs[0], delta, lenfft)
    resp1 = computeresp(pazs[1], delta, lenfft)
    resp2 = computeresp(pazs[2], delta, lenfft)
    
    #Calculating the PSD for each sensor
    (p11, f) = cp(st[0],st[0],lenfft,lenol,delta)
    (p22, f) = cp(st[1],st[1],lenfft,lenol,delta)
    (p33, f) = cp(st[2],st[2],lenfft,lenol,delta)
    
    #Calculate the PSD of two sensors combined
    (p21, f) = cp(st[1],st[0],lenfft,lenol,delta)
    (p13, f) = cp(st[0],st[2],lenfft,lenol,delta)
    (p23, f) = cp(st[1],st[2],lenfft,lenol,delta)
    
    #Calculate the noise based on the PSDs and responses
    n11 = ((2.*math.pi*f)**2)*(p11 - p21*p13/p23)/resp0
    n22 = ((2.*math.pi*f)**2)*(p22 - np.conjugate(p23)*p21/np.conjugate(p13))/resp1
    n33 = ((2.*math.pi*f)**2)*(p33 - p23*np.conjugate(p13)/p21)/resp2
    
    #Convert noise to dB
    n11 = 10.*np.log10(np.abs(n11))
    n22 = 10.*np.log10(np.abs(n22))
    n33 = 10.*np.log10(np.abs(n33))
    
    #Convert PSDs to dB
    p11= 10.*np.log10(np.abs(((2.*math.pi*f)**2)*p11/resp0))
    p22= 10.*np.log10(np.abs(((2.*math.pi*f)**2)*p22/resp1))
    p33= 10.*np.log10(np.abs(((2.*math.pi*f)**2)*p33/resp2))
    
    #Setting an index for each noise and PSD to be set later
    n11 = [int(idx) for idx in n11]
    n22 = [int(idx) for idx in n22]
    n33 = [int(idx) for idx in n33]
    p11 = [int(idx) for idx in p11]
    p22 = [int(idx) for idx in p22]
    p33 = [int(idx) for idx in p33]
    
    #5-point moving average calculation to smooth peaks in data.
    N= 5
    n11 =np.convolve(n11, np.ones((N,))/N, mode='same')
    n22 =np.convolve(n22, np.ones((N,))/N, mode='same')
    n33 =np.convolve(n33, np.ones((N,))/N, mode='same')
    p11 =np.convolve(p11, np.ones((N,))/N, mode='same')
    p22 =np.convolve(p22, np.ones((N,))/N, mode='same')
    p33 =np.convolve(p33, np.ones((N,))/N, mode='same')
    f =np.convolve(f, np.ones((N,))/N, mode='same')
    
    #Puts noise and PSDs into a list so that they can be called by index later
    #depending on which sensor we're looping through.
    n = [n11, n22, n33]
    p=[p11, p22, p33]
    return n, p, f

#Function to compute the response of each seismometer    
def computeresp(resp,delta,lenfft):
    fnorm = 0.1
    respval, freq = paz_to_freq_resp(resp['poles'],resp['zeros'],resp['sensitivity'],t_samp = delta, 
    nfft=lenfft,freq = True)
    index = np.argmin(np.abs(freq-fnorm))
    respval = np.absolute(respval*np.conjugate(respval))
    respval = respval/respval[index]
    respval = respval[1:]
    respval = respval*resp['sensitivity']**2
    return respval



#List of times corresponding to the beginning of the 24-hour period over which analyzed data was chosen. The order
#goes from lowest voltage (0V) to highest voltage (15.6V).
times = [UTCDateTime('2017-203T12:00:00.0'),UTCDateTime('2017-199T12:00:00.0'), UTCDateTime('2017-196T12:00:00.0'),
        UTCDateTime('2017-216T17:00:00.0'), UTCDateTime('2017-194T12:00:00.0'), UTCDateTime('2017-214T20:00:00.0'), 
        UTCDateTime('2017-190T12:00:00.0'), UTCDateTime('2017-210T00:00:00.0'), UTCDateTime('2017-207T19:30:00.0')]

#SNCLs for our sensors.
stations = ['XX_TST5_00','XX_TST4_00','XX_TST6_00']
chan = 'LH0'

#Importing the NLNM and the NHNM from Obspy
model_periods, high_noise = get_nhnm()
model_periods, low_noise = get_nlnm()



#Starting to define our colormap. This sets the number of colors we want, and the subsection of the map we'll use.
start = 0.0
stop = 1.0
number_of_lines= len(times)
cm_subsection = np.linspace(start, stop, number_of_lines) 



#Run calculation at each station for each time window
for sidx, sta in enumerate(stations):
    net,sta,loc = sta.split('_')
    fig = plt.figure(1, figsize=(28,18))
    plt.subplots_adjust(hspace=0.001, wspace=0.001)
    
    #Indexing over times so that we can plot all times against each other for each sensor.
    for cidx, time in enumerate(times):
        
        #Setting the color that will correspond to each time section
        color = cm.jet(cm_subsection[cidx])
        
        #Import data
        st = Stream()
        string = '/tr1/telemetry_days/'+ net + '_' + sta + '/' + str(time.year) \
                    + '/' + str(time.year) +'_' + str(time.julday).zfill(3) + '/' + loc + '_' + chan + '*'
        st += read(string)
        string = '/tr1/telemetry_days/'+ net + '_' + sta + '/' + str(time.year) \
                + '/' + str(time.year) +'_' + str((time+24.*60.*60.).julday).zfill(3) + '/' + loc + '_' + chan + '*'
        st += read(string)
        st.merge()
        st.trim(time, time+(24.*60.*60.))
        
        string = '/tr1/telemetry_days/'+ net + '_' + 'TST4' + '/' + str(time.year) \
                    + '/' + str(time.year) +'_' + str(time.julday).zfill(3) + '/' + '10' + '_' + chan + '*'
        st += read(string)
        string = '/tr1/telemetry_days/'+ net + '_' + 'TST4' + '/' + str(time.year) \
                + '/' + str(time.year) +'_' + str((time+24.*60.*60.).julday).zfill(3) + '/' + '10' + '_' + chan + '*'
        st += read(string)
        st.merge()
        st.trim(time, time+(24.*60.*60.))
        string = '/tr1/telemetry_days/'+ net + '_' + 'TST5' + '/' + str(time.year) \
                    + '/' + str(time.year) +'_' + str(time.julday).zfill(3) + '/' + '10' + '_' + chan + '*'
        st += read(string)
        string = '/tr1/telemetry_days/'+ net + '_' + 'TST5' + '/' + str(time.year) \
                + '/' + str(time.year) +'_' + str((time+24.*60.*60.).julday).zfill(3) + '/' + '10' + '_' + chan + '*'
        st += read(string)
        st.merge()
        
        #Trim the data to the desired 24-hour time period.
        st.trim(time, time+(24.*60.*60.))
        
        #Set the labels so that the peak-to-peak temperature variation at each time window is listed in the legend.
        if time == UTCDateTime('2017-207T19:30:00.0'):
            label = '$\Delta$T = 0.195 ${}^{\circ} C$'
        elif time == UTCDateTime('2017-210T00:00:00.0'):
            label = '$\Delta$T = 0.114 ${}^{\circ} C$'
        elif time == UTCDateTime('2017-190T12:00:00.0'):
            label = '$\Delta$T = 0.061 ${}^{\circ} C$'
        elif time == UTCDateTime('2017-214T20:00:00.0'):
            label = '$\Delta$T = 0.029 ${}^{\circ} C$'
        elif time == UTCDateTime('2017-194T12:00:00.0'):
            label = '$\Delta$T = 0.020 ${}^{\circ} C$'
        elif time == UTCDateTime('2017-196T12:00:00.0'):
            label = '$\Delta$T = 0.0013 ${}^{\circ} C$'
        elif time == UTCDateTime('2017-216T17:00:00.0'):
            label = '$\Delta$T = 0.0016 ${}^{\circ} C$'
        elif time == UTCDateTime('2017-199T12:00:00.0'):
            label = '$\Delta$T = 0.0007 ${}^{\circ} C$'
        elif time == UTCDateTime('2017-203T12:00:00.0'):
            label = 'Reference'
        pazs = []
        
        # Set the correct poles and zeros values for the type of sensor we are looping over.
        for idx, tr in enumerate(st):
            net2, sta2, loc2, chan2 = (tr.id).split('.')
            paz = returnresp(sta2, loc2)
            pazs.append(paz)
            if loc2 == '00':
                goodidx = idx
        
        #Calculate self-noise and frequency.
        n, p, f = selfnoise(st, pazs)
        
        #PSD Figures. sidx indexing makes it so that each sensor's PSD and noise match up vertically.
        ax = fig.add_subplot(2, 3, sidx+1)
        ax.semilogx(model_periods, high_noise, '0.4', linewidth=2)
        ax.semilogx(model_periods, low_noise, '0.4', linewidth=2)
        ax.semilogx(1./f, p[goodidx], c = color)
        ax.text(4.5,-90., letters[sidx])
        plt.xticks([])
        
        #Plot y ticks only on the furthest left plot.
        if sidx == 0:
            plt.yticks([-180., -160., -140., -120., -100.])
        else:
            plt.yticks([])
        plt.xlim([4.,10000.])
        plt.ylim(-200,-80)
        
        #Self-noise Figures
        ax = fig.add_subplot(2, 3, sidx+4)
        ax.semilogx(model_periods, high_noise, '0.4', linewidth=2)
        
        #if statement in label keyword makes sure that the legend entry for NLNM/NLHM only comes up once.
        ax.semilogx(model_periods, low_noise, '0.4', linewidth=2, label = 'NLNM/NHNM' if cidx ==0 else '_nolegend_')
        ax.semilogx(1./f, n[goodidx], c = color, label = label)
        ax.text(4.5,-90., letters[sidx+3])
        
        #Plot y ticks only on the furthest left plot.
        if sidx == 0:
            plt.yticks([-180., -160., -140., -120., -100.])
        else:
            plt.yticks([])
        plt.xticks([10**1,10**2,10**3])
        plt.xlim((4.,10000.))
        plt.ylim(-200,-80)

#Centering axes labels over entire figure       
fig.text(0.05, 0.5,'Power (dB rel. 1 $(m/s^2)^2/Hz$)', ha='center', va='center', rotation='vertical', fontsize = 30)
fig.text(0.5, 0.06,'Period (s)', ha='center', va='center', fontsize = 30)

#Creating and centering legend below the plots.
plt.subplot(235)
plt.legend(loc = 'center', bbox_to_anchor=(0.5, -0.22), ncol = 5, fontsize = 20)

plt.savefig('PSD_and_Self-Noise_' + sta + '.jpg', format= 'JPEG', dpi = 200)
plt.savefig('PSD_and_Self-Noise_' + sta + '.pdf', format= 'PDF', dpi = 200)
plt.clf()
plt.close()
    
        
