###This code calculates the PSD and self-noise values of the three instruments
###at IRIS/USGS network station TUC. This code also fits the 1/f noise seen ove
###50 days at TUC.



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

#Setting font parameters
import matplotlib as mpl
mpl.rc('font',family='serif')
mpl.rc('font',serif='Times') 
mpl.rc('text', usetex=True)
mpl.rc('font',size=18)

#Defining function for calculating PSD values.
def cp(tr1,tr2,lenfft,lenol,delta):
    sr = 1/delta
    cpval,fre = csd(tr1.data,tr2.data,NFFT=lenfft,Fs=sr,noverlap=lenol,scale_by_freq=True)
    fre = fre[1:]
    cpval = cpval[1:]
    return cpval, fre  

#Calculated PSD and self-noise
def selfnoise(st, resps):
    delta = st[0].stats.delta
    
    #Cacluates PSD for each sensor
    (p11, f) = cp(st[0],st[0],lenfft,lenol,delta)
    (p22, f) = cp(st[1],st[1],lenfft,lenol,delta)
    (p33, f) = cp(st[2],st[2],lenfft,lenol,delta)
    
    #Calculates PSD between different sensors.
    (p21, f) = cp(st[1],st[0],lenfft,lenol,delta)
    (p13, f) = cp(st[0],st[2],lenfft,lenol,delta)
    (p23, f) = cp(st[1],st[2],lenfft,lenol,delta)
    
    #Calculates the sensor self-noise using the three-sensor method.
    n11 = ((2.*math.pi*f)**2)*(p11 - p21*p13/p23)/resps['00']
    n22 = ((2.*math.pi*f)**2)*(p22 - np.conjugate(p23)*p21/np.conjugate(p13))/resps['10']
    n33 = ((2.*math.pi*f)**2)*(p33 - p23*np.conjugate(p13)/p21)/resps['60']
    #Converts noise and power to PSDs
    n11 = 10.*np.log10(np.abs(n11))
    n22 = 10.*np.log10(np.abs(n22))
    n33 = 10.*np.log10(np.abs(n33))
    p11= 10.*np.log10(np.abs(((2.*math.pi*f)**2)*p11/resps['00']))
    p22= 10.*np.log10(np.abs(((2.*math.pi*f)**2)*p22/resps['10']))
    p33= 10.*np.log10(np.abs(((2.*math.pi*f)**2)*p33/resps['60']))
    #Sets index for each value
    n11 = [int(idx) for idx in n11]
    n22 = [int(idx) for idx in n22]
    n33 = [int(idx) for idx in n33]
    p11 = [int(idx) for idx in p11]
    p22 = [int(idx) for idx in p22]
    p33 = [int(idx) for idx in p33]
    #Puts values in lists 
    n = [n11, n22, n33]
    p=[p11, p22, p33]
    
    return n, p, f

#Computes the response of each instrument    
def computeresp(resp,delta,lenfft):
    respval = paz_to_freq_resp(resp['poles'],resp['zeros'],resp['sensitivity']*resp['gain'],t_samp = delta, 
        nfft=lenfft,freq = False)
    respval = np.absolute(respval*np.conjugate(respval))
    respval = respval[1:]
    return respval   
    
    

#4096 second time windows with 512 seconds of overlap
lenol=512
lenfft=4096

#Sets response for each type of sensor based on its location on the digitizers
locs =['00','10','60']
resps ={}
for loc in locs:
    resp = evalresp(t_samp = 1., nfft = lenfft, filename= '/APPS/metadata/RESPS/RESP.IU.' + \
                            'TUC.' + loc + '.LHZ',  date = UTCDateTime('2017-001T00:00:00.0'), station = 'TUC',
                            channel = 'LHZ', network = 'IU', locid = loc, units = 'VEL') 
    resp = resp[1:]
    resp = np.absolute(resp*np.conjugate(resp))
    resps[loc] = np.abs(resp)





#Sets dB range, the size of the bins, and length of the bins for calculating a PDF.
db_range = (-220, -80, 1)
numbins = int((db_range[1] - db_range[0])/db_range[2]) 
db_bins = np.linspace(db_range[0], db_range[1], numbins+1, endpoint=True)


#Initialize lists and then convert them into arrays.
freqs =[]
ns0 =[]
ps0 = []
ns1 =[]
ps1 = []
ns2 =[]
ps2 = []
freqs = np.asarray(freqs)
ns0 = np.asarray(ns0)
ns1 = np.asarray(ns1)
ns2 = np.asarray(ns2)
ps0=np.asarray(ps0)
ps1=np.asarray(ps1)
ps2=np.asarray(ps2)
#Start and end dates
sday =1
eday=150
hourC=4

for day in range(sday,eday):
    #Allows you to monitor which day you're running through.
    print('On day: ' + str(day))
    #Reads in data and sorts it by sensor through its location code
    st = read('/msd/IU_TUC/2017/' + str(day).zfill(3) + '/*LHZ*')
    st.sort(['location'])
    
    #For every 4 hour period in each day,...
    for hour in range(0,24,hourC):
        try:
            #Copy in temperature data
            stTemp=st.copy()
            stTemp.trim(st[0].stats.starttime + hour*60*60., st[0].stats.starttime + (hour+ hourC)*60.*60.)
            #Calculate PSD and self-noise of each sensor.
            n,p,f = selfnoise(stTemp, resps)
            ns0 =np.append(ns0, n[0])
            ps0 = np.append(ps0,p[0])
            ns1 =np.append(ns1, n[1])
            ps1 = np.append(ps1,p[1])
            ns2 =np.append(ns2, n[2])
            ps2 = np.append(ps2,p[2])
            freqs = np.append(freqs,f)
        except:
            #This gets printed when the above is not possible
            print('Day ' + str(day) + ' bad')

#Puts all noise values from every day into an array.        
ns0 = np.asarray(ns0)      
ns1 = np.asarray(ns1) 
ns2 = np.asarray(ns2)      
#Creates a list of frequencies.
fedge=list(set(freqs))
fedge = np.sort(fedge)
fedge = np.asarray(fedge)
#Initializing lists of mean and standard deviation.
pows0=[]
pows1=[]
pows2=[]
pows0std=[]
pows1std=[]
pows2std=[]
npows0=[]
npows1=[]
npows2=[]
npows0std=[]
npows1std=[]
npows2std=[]
for idx, ele in enumerate(fedge):
    #Calculate mean noise value at every frequency.
    npows0.append(np.mean(ns0[freqs == ele]))
    npows1.append(np.mean(ns1[freqs == ele]))
    npows2.append(np.mean(ns2[freqs == ele]))
    npows0std.append(np.std(ns0[freqs == ele]))
    #Calculate noise STD value at every frequency.
    npows1std.append(np.std(ns1[freqs == ele]))
    npows2std.append(np.std(ns2[freqs == ele]))
    #Mean PSD value at every frequency
    pows0.append(np.mean(ps0[freqs == ele]))
    pows1.append(np.mean(ps1[freqs == ele]))
    pows2.append(np.mean(ps2[freqs == ele]))
    #PSD STD value at every frequency
    pows0std.append(np.std(ps0[freqs == ele]))
    pows1std.append(np.std(ps1[freqs == ele]))
    pows2std.append(np.std(ps2[freqs == ele]))

#Puts noise into an array, then trims the array in a certain frequency band (.001Hz to 0.01Hz)
npows0=np.asarray(npows0)
npows1=np.asarray(npows1)
npows2=np.asarray(npows2)
lppow0 = npows0[(.01 >= fedge) & (fedge >= .001)]
lppow1 = npows1[(.01 >= fedge) & (fedge >= .001)]
lppow2 = npows2[(.01 >= fedge) & (fedge >= .001)]
freqslp=fedge[(.01 >= fedge) & (fedge >= .001)]

#Define function to which we will fit our data to 1/f noise.
def fun(x, b,c):
    return (x)**b + c

from scipy.optimize import curve_fit

#Find optimized values for mean noise vs. period. 
popt0, pcov0 = curve_fit(fun, 1/freqslp, lppow0)
popt1, pcov1 = curve_fit(fun, 1/freqslp, lppow1)
popt2, pcov1 = curve_fit(fun, 1/freqslp, lppow2)
print(popt0)
print(popt1)
print(popt2)


fig = plt.figure(1,figsize=(12,12))
plt.subplots_adjust(hspace=0.001)

#Plot of PSD values
plt.subplot(211)
plt.semilogx(1/fedge,pows0, 'g', label='STS-1',linewidth=1.5)
plt.semilogx(1/fedge,pows1,'b',  label='STS-2.5',linewidth=1.5)
plt.semilogx(1/fedge,pows2,'r', label='STS-6',linewidth=1.5)
#Plot NLNM/NLHM for reference
model_periods, high_noise = get_nhnm()
plt.semilogx(model_periods, high_noise, '0.4', linewidth=2)
model_periods, low_noise = get_nlnm()
plt.semilogx(model_periods, low_noise, '0.4', linewidth=2,label='NLNM/NHNM')
plt.xlim((4.,1000))
plt.ylim(-200,-120)
plt.yticks([-180., -160., -140.])
plt.text(4.5, -125, 'A)')
#Remove x ticks from top subplot.
plt.xticks([])
plt.legend(loc='upper right')

#Plot of noise values
plt.subplot(212)
plt.semilogx(1/fedge,npows0,'g',linewidth=1.5, label='STS-1 Self-Noise')
plt.semilogx(1/fedge,npows1,'b',linewidth=1.5, label='STS-2.5 Self-Noise')
plt.semilogx(1/fedge,npows2, 'r',linewidth=1.5, label='STS-6 Self-Noise')


#Plot of fits to noise values
plt.subplot(212)
plt.semilogx(1/fedge, fun(1/fedge,*popt0),'g--', linewidth=2., label='STS-1 1/f Fit: ' + str(round(popt0[0],4)))
plt.semilogx(1/fedge, fun(1/fedge,*popt1),'b--', linewidth=2., label='STS-2.5 1/f Fit: ' + str(round(popt1[0],4)))
plt.semilogx(1/fedge, fun(1/fedge,*popt2),'r--', linewidth=2., label='STS-6 1/f Fit: ' + str(round(popt2[0],4)))
#NLNM/NHNM for reference
model_periods, high_noise = get_nhnm()
plt.semilogx(model_periods, high_noise, '0.4', linewidth=2)
model_periods, low_noise = get_nlnm()
plt.semilogx(model_periods, low_noise, '0.4', linewidth=2,label='NLNM/NHNM')
plt.text(4.5, -155, 'B)')
plt.yticks([-180., -160., -140.])
plt.xlim((4.,1000))
plt.ylim(-200,-150)
plt.xlabel('Period (s)')

plt.legend(loc='upper right', ncol =2)
#Center y axis label between the two plots
fig.text(0.05, 0.5,'Power (dB rel. 1 $(m/s^2)^2/Hz$)', ha='center', va='center', rotation='vertical')

plt.savefig('TUCPower.jpg',format='JPEG', dpi=400)
plt.savefig('TUCPower.pdf', format='PDF', dpi=400)
plt.clf()
plt.close()

