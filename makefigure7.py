###This code calculates the acceleration output and temperature output for
###the three sensors at IRIS/USGS network station TUC. 



#!/usr/bin/env python


from obspy.core import read, UTCDateTime, Stream
from obspy.io.xseed import Parser
import matplotlib.pyplot as plt
import numpy as np
from obspy.signal.invsim import simulate_seismometer

#Font parameters from matplotlib
import matplotlib as mpl
mpl.rc('font',family='serif')
mpl.rc('font',serif='Times') 
mpl.rc('text', usetex=True)
mpl.rc('font',size=18)

#Read in data for all three sensors and their temperature sensors in 
#the day range we specify.
st = Stream()
for days in range(101,151):
    st += read('/msd/IU_TUC/2017/' + str(days).zfill(3) + '/*LHZ*')
# Temp of STS-2.5
    st += read('/msd/IU_TUC/2017/' + str(days).zfill(3) + '/30_LK8*')
    st += read('/msd/IU_TUC/2017/' + str(days).zfill(3) + '/10_LK5*')
    st += read('/msd/IU_TUC/2017/' + str(days).zfill(3) + '/00_LK2*')

st.merge()

#Find sensor response values, demean, and then remove response. 
for tr in st.select(channel='LH*'):
    seedresp ={'filename' : '/APPS/metadata/RESPS/RESP.' + tr.id, 'date': tr.stats.starttime, 'units': 'ACC'}
    tr.detrend('constant')
    tr.simulate(seedresp=seedresp )

#Demean the temperature data and remove the bitweight of the digitizer    
for tr in st.select(channel='LK*'):
    tr.detrend('linear')
    if tr.stats.location == '10' or tr.stats.location == '00':
        tr.data *= (1./(1.26*10**5))
    else:
        tr.data *= (1./208.7)
# Decimate data to 1 sample per every 10 seconds to smooth curves.        
st.decimate(2)
st.decimate(5)
#1000s lowpass filter
st.filter('lowpass',freq=0.001)
#Sort the stream and print it along with the ratios between temperature and acceleration
#for each sensor and its corresponding temperature sensor.
st.sort()
print(st)
print('Ratio STS-2.5:' + str(st[2].std()/st[4].std()))
print('Ratio STS-1:' + str(st[0].std()/st[3].std()))
print('Ratio STS-6:' + str(st[5].std()/st[1].std()))

#Create a vector of time in seconds.
t = (np.arange(0,len(st[0].data))/st[0].stats.sampling_rate)/(24*60*60) 

rat25 = st[2].std()/st[4].std()
rat1 = st[0].std()/st[3].std()
rat6 = st[5].std()/st[1].std()

#To be used in a 20-second moving average later in the code.
N=20

fig = plt.figure(1, figsize=(12,12))
plt.subplots_adjust(hspace=0.001)
#Plot for the STS-1 in water bricks with a foam covering
ax1 = plt.subplot(311)
#Plot the sensor acceleration output in blue
for tr in st.select(id="IU.TUC.00.LHZ"):
    ax1.plot(t, tr.data/(10**-6) ,label='STS-1', color = 'b')
    ax1.set_ylim((-200.,200))
    ax1.set_yticks([-150., 0., 150.])
    ax1.tick_params(axis='y', colors='blue')
    plt.text(0.5, 148., 'A) STS-1 Vertical in Water Bricks (r = -0.89)')
#Plot the temperature output in red on the same plot.
ax2= ax1.twinx()
for tr in st.select(id="IU.TUC.10.LK5"):
    td=np.convolve(tr.data, np.ones((N,))/N, mode='same')
    ax2.plot(t, td ,label=tr.id, color='r')
    ax2.set_ylim((-.7,.7))
    ax2.set_yticks([-.5,0., .5])
    ax2.tick_params(axis='y', colors='red')
plt.xlim((min(t),max(t)))
#Remove x ticks
plt.xticks([])

#Plot for the STS-2.5 in a foam box.
ax1=plt.subplot(312)
#Acceleration output in blue
for tr in st.select(id="IU.TUC.10.LHZ"):
    ax1.plot(t, tr.data/(10**-6),label='STS-2.5',color='b')

    ax1.tick_params(axis='y', colors='blue')

    ax1.set_ylim((-200.,200))
    ax1.set_yticks([-150., 0., 150.])
    plt.text(0.5, 148., 'B) STS-2.5 in Foam Box (r = -0.63)')
#Temperature output in red
ax2= ax1.twinx()
for tr in st.select(id="IU.TUC.30.LK8"):
    td=np.convolve(tr.data, np.ones((N,))/N, mode='same')
    ax2.plot(t, td ,label=tr.id,color='r')
    ax2.set_ylim((-.7,.7))
    ax2.set_yticks([-.5,0., .5])
    ax2.tick_params(axis='y', colors='red')
plt.xlim((min(t),max(t)))
#Remove x ticks.
plt.xticks([])

#Plot for STS-6 in 1.7m post-hole.
ax1=plt.subplot(313)
#Acceleration output in blue
for tr in st.select(id="IU.TUC.60.LHZ"):
    ax1.plot(t,tr.data/(10**-6),label='STS-6',color='b')
    ax1.set_ylim((-200.,200))
    ax1.set_yticks([-150., 0., 150.])
    ax1.tick_params(axis='y', colors='blue')
    plt.text(0.5, 148., 'C) Post-Hole STS-6 (r = 0.33)')
#Temperature output in red
ax2= ax1.twinx()
for tr in st.select(id="IU.TUC.00.LK2"):
    td=np.convolve(tr.data, np.ones((N,))/N, mode='same')
    ax2.plot(t, td ,label=tr.id,color='r')
    ax2.set_ylim((-.7,.7))
    ax2.set_yticks([-.5,0., .5])
    ax2.tick_params(axis='y', colors='red')
plt.xlim((min(t), max(t)))
ax1.set_xlabel('Time (days)')

#Set y labels on either side to be centered with all three subplots.
fig.text(0.05, 0.5,'Acceleration $(\mu m/s^2)$', ha='center', va='center', rotation='vertical', color='b')
fig.text(0.97, 0.5,'Relative Temperature $({}^{\circ}C)$', ha='center', va='center', rotation='vertical', color='r')

plt.savefig('TEMPTUC.pdf',format='PDF',dpi=400)
plt.savefig('TEMPTUC.jpg',format='JPEG', dpi=400)
plt.clf()
plt.close()
