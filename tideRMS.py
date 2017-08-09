#!/usr/bin/env python


from obspy.core import read, UTCDateTime, Stream
from obspy.io.xseed import Parser
import matplotlib.pyplot as plt
import numpy as np
from obspy.signal.invsim import simulate_seismometer
from obspy.signal.cross_correlation import xcorr

import matplotlib as mpl
mpl.rc('font',family='serif')
mpl.rc('font',serif='Times') 
mpl.rc('text', usetex=True)
mpl.rc('font',size=22)

#sp = Parser('/APPS/metadata/SEED/IU.dataless')
st = Stream()
for days in range(101,151):
    st += read('/msd/IU_TUC/2017/' + str(days).zfill(3) + '/*LHZ*')
# Temp of STS-2.5
    st += read('/msd/IU_TUC/2017/' + str(days).zfill(3) + '/30_LK8*')
    st += read('/msd/IU_TUC/2017/' + str(days).zfill(3) + '/10_LK5*')
    st += read('/msd/IU_TUC/2017/' + str(days).zfill(3) + '/00_LK2*')

st.merge()

 
for tr in st.select(channel='LH*'):
    seedresp ={'filename' : '/APPS/metadata/RESPS/RESP.' + tr.id, 'date': tr.stats.starttime, 'units': 'ACC'}
    tr.detrend('constant')
    tr.simulate(seedresp=seedresp )
    #tr.filter("lowpass", freq=.001)
    #p=np.poly1d(np.polyfit(t,tr.data,3))
    #tr.data = tr.data -p(t)
    
for tr in st.select(channel='LK*'):
    tr.detrend('linear')
    #tr.filter("lowpass", freq=.001)
    if tr.stats.location == '10' or tr.stats.location == '00':
        tr.data *= (1./(1.26*10**5))
    else:
        tr.data *= (1./208.7)
        
st.decimate(2)
st.decimate(5)
st.filter('lowpass',freq=0.001)
st.sort()
print(st)
print('Ratio STS-2.5:' + str(st[2].std()/st[4].std()))
print('Ratio STS-1:' + str(st[0].std()/st[3].std()))
print('Ratio STS-6:' + str(st[5].std()/st[1].std()))

t = (np.arange(0,len(st[0].data))/st[0].stats.sampling_rate)/(24*60*60)


sts1 =[]
sts25=[]
sts6=[]
sts1T=[]
sts25T=[]
sts6T=[]
for idx, stT in enumerate(st.slide(window_length=24.*60.60, step=24.*60.*60.)):
    print(str(idx))
    sts25.append(stT[2].std())
    sts25T.append(stT[4].std())
    sts1.append(stT[0].std())
    sts1T.append(stT[3].std())
    sts6.append(stT[5].std())
    sts6T.append(stT[1].std())
    
    
sts1= np.asarray(sts1)
sts1T= np.asarray(sts1T)
sts25= np.asarray(sts25)
sts25T= np.asarray(sts25T)
sts6= np.asarray(sts6)
sts6T= np.asarray(sts6T)
t2 = np.arange(0,len(sts25))
fig =plt.figure(2, figsize=(12,12))
plt.subplots_adjust(hspace=0.001)
plt.subplot(311)
plt.plot(t2,sts1/sts1T, color='k', linewidth=2)
plt.ylim((0.,0.0007))
plt.text(2,0.0006,'STS-1')
plt.xlim((1., max(t2)))
plt.subplot(312)
plt.plot(t2,sts25/sts25T, color='k', linewidth=2)
plt.ylabel('Daily RMS Ratio ($m/s^2/{}^{\circ}C$)')
plt.ylim((0.,0.0007))
plt.xlim((1., max(t2)))
plt.text(2,0.0006,'STS-2.5')
plt.subplot(313)
plt.plot(t2,sts6/sts6T, color='k',linewidth=2)
plt.ylim((0.,0.0007))
plt.text(2,0.0006,'STS-6')
plt.xlim((1., max(t2)))
plt.xlabel('Time(Days)')
plt.tight_layout()
#plt.show()
plt.savefig('TUCRMS.pdf',format='PDF',dpi=400)
plt.savefig('TUCRMS.jpg',format='JPEG', dpi=400)

import sys
sys.exit()

rat25 = st[2].std()/st[4].std()
rat1 = st[0].std()/st[3].std()
rat6 = st[5].std()/st[1].std()

N=5000

fig = plt.figure(1, figsize=(12,12))
plt.subplots_adjust(hspace=0.001)
ax1 = plt.subplot(311)
for tr in st.select(id="IU.TUC.00.LHZ"):
    blah=tr
    ax1.plot(t, tr.data/(10**-6) ,label='STS-1', color = 'b',linewidth=2)
    ax1.set_ylim((-200.,200))
    ax1.set_yticks([-150., 0., 150.])
    ax1.tick_params(axis='y', colors='blue')
    plt.legend(loc='upper left')
ax2= ax1.twinx()
for tr in st.select(id="IU.TUC.10.LK5"):
    td=np.convolve(tr.data, np.ones((N,))/N, mode='same')
    ax2.plot(t, td ,label=tr.id, color='r',linewidth=2)
    ax2.set_ylim((-.7,.7))
    ax2.set_yticks([-.5,0., .5])
    ax2.tick_params(axis='y', colors='red')
    blah2=tr
plt.xlim((min(t),max(t)))
plt.xticks([])

print('Here is the corr 00LHZ: ' + str(xcorr(blah,blah2,10000)))


#ax1.set_ylabel('Acceleration $(m/s^2)$',color='b')
#ax2.set_ylabel('Temperature $({}^{\circ}C)$', color='r')
ax1=plt.subplot(312)
for tr in st.select(id="IU.TUC.10.LHZ"):
    ax1.plot(t, tr.data/(10**-6),label='STS-2.5',color='b',linewidth=2)
    blah=tr
    ax1.tick_params(axis='y', colors='blue')

    ax1.set_ylim((-200.,200))
    ax1.set_yticks([-150., 0., 150.])
    plt.legend(loc='upper left')
ax2= ax1.twinx()
for tr in st.select(id="IU.TUC.30.LK8"):
    td=np.convolve(tr.data, np.ones((N,))/N, mode='same')
    blah2=tr
    ax2.plot(t, td ,label=tr.id,color='r',linewidth=2)
    ax2.set_ylim((-.7,.7))
    ax2.set_yticks([-.5,0., .5])
    ax2.tick_params(axis='y', colors='red')
plt.xlim((min(t),max(t)))
#ax1.set_ylabel('Acceleration $(m/s^2)$',color='k')
#ax2.set_ylabel('Temperature $({}^{\circ}C)$',color='.5')
print('Here is the corr 10LHZ: ' + str(xcorr(blah,blah2,10000)))
plt.xticks([])
ax1=plt.subplot(313)
for tr in st.select(id="IU.TUC.60.LHZ"):
    ax1.plot(t,tr.data/(10**-6),label='STS-6',color='b',linewidth=2)
    blah=tr
    ax1.set_ylim((-200.,200))
    ax1.set_yticks([-150., 0., 150.])
    ax1.tick_params(axis='y', colors='blue')
    plt.legend(loc='upper left')
ax2= ax1.twinx()
for tr in st.select(id="IU.TUC.00.LK2"):
    td=np.convolve(tr.data, np.ones((N,))/N, mode='same')
    blah2=tr
    ax2.plot(t, td ,label=tr.id,color='r',linewidth=2)
    ax2.set_ylim((-.7,.7))
    ax2.set_yticks([-.5,0., .5])
    ax2.tick_params(axis='y', colors='red')
plt.xlim((min(t), max(t)))
ax1.set_xlabel('Time (days)')
print('Here is the corr 60LHZ: ' + str(xcorr(blah,blah2,10000)))
#ax2.set_ylabel('Temperature $({}^{\circ}C)$',color='r')
fig.text(0.05, 0.5,'Acceleration $(\mu m/s^2)$', ha='center', va='center', rotation='vertical', color='b')
fig.text(0.97, 0.5,'Relative Temperature $({}^{\circ}C)$', ha='center', va='center', rotation='vertical', color='r')

plt.savefig('TEMPTUC.pdf',format='PDF',dpi=400)
plt.savefig('TEMPTUC.jpg',format='JPEG', dpi=400)
