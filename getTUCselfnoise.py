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
from matplotlib.colors import LinearSegmentedColormap

import matplotlib as mpl
mpl.rc('font',family='serif')
mpl.rc('font',serif='Times') 
mpl.rc('text', usetex=True)
mpl.rc('font',size=18)

CDICT = {'red': ((0.0, 1.0, 1.0),
                 (0.05, 1.0, 1.0),
                 (0.2, 0.0, 0.0),
                 (0.4, 0.0, 0.0),
                 (0.6, 0.0, 0.0),
                 (0.8, 1.0, 1.0),
                 (1.0, 1.0, 1.0)),
         'green': ((0.0, 1.0, 1.0),
                   (0.05, 0.0, 0.0),
                   (0.2, 0.0, 0.0),
                   (0.4, 1.0, 1.0),
                   (0.6, 1.0, 1.0),
                   (0.8, 1.0, 1.0),
                   (1.0, 0.0, 0.0)),
         'blue': ((0.0, 1.0, 1.0),
                  (0.05, 1.0, 1.0),
                  (0.2, 1.0, 1.0),
                  (0.4, 1.0, 1.0),
                  (0.6, 0.0, 0.0),
                  (0.8, 0.0, 0.0),
                  (1.0, 0.0, 0.0))}

def cp(tr1,tr2,lenfft,lenol,delta):
    sr = 1/delta
    cpval,fre = csd(tr1.data,tr2.data,NFFT=lenfft,Fs=sr,noverlap=lenol,scale_by_freq=True)
    fre = fre[1:]
    cpval = cpval[1:]
    return cpval, fre  


def selfnoise(st, resps):
    delta = st[0].stats.delta

    (p11, f) = cp(st[0],st[0],lenfft,lenol,delta)
    (p22, f) = cp(st[1],st[1],lenfft,lenol,delta)
    (p33, f) = cp(st[2],st[2],lenfft,lenol,delta)

    (p21, f) = cp(st[1],st[0],lenfft,lenol,delta)
    (p13, f) = cp(st[0],st[2],lenfft,lenol,delta)
    (p23, f) = cp(st[1],st[2],lenfft,lenol,delta)

    n11 = ((2.*math.pi*f)**2)*(p11 - p21*p13/p23)/resps['00']
    n22 = ((2.*math.pi*f)**2)*(p22 - np.conjugate(p23)*p21/np.conjugate(p13))/resps['10']
    n33 = ((2.*math.pi*f)**2)*(p33 - p23*np.conjugate(p13)/p21)/resps['60']
    n11 = 10.*np.log10(np.abs(n11))
    n22 = 10.*np.log10(np.abs(n22))
    n33 = 10.*np.log10(np.abs(n33))
    p11= 10.*np.log10(np.abs(((2.*math.pi*f)**2)*p11/resps['00']))
    p22= 10.*np.log10(np.abs(((2.*math.pi*f)**2)*p22/resps['10']))
    p33= 10.*np.log10(np.abs(((2.*math.pi*f)**2)*p33/resps['60']))
    n11 = [int(idx) for idx in n11]
    n22 = [int(idx) for idx in n22]
    n33 = [int(idx) for idx in n33]
    p11 = [int(idx) for idx in p11]
    p22 = [int(idx) for idx in p22]
    p33 = [int(idx) for idx in p33]
    n = [n11, n22, n33]
    p=[p11, p22, p33]
    
    return n, p, f
    
def computeresp(resp,delta,lenfft):
    respval = paz_to_freq_resp(resp['poles'],resp['zeros'],resp['sensitivity']*resp['gain'],t_samp = delta, 
        nfft=lenfft,freq = False)
    respval = np.absolute(respval*np.conjugate(respval))
    respval = respval[1:]
    return respval   
    
    


lenol=512
lenfft=4096

#sp = Parser('/APPS/metadata/SEED/IU.dataless')



locs =['00','10','60']
resps ={}
for loc in locs:
    resp = evalresp(t_samp = 1., nfft = lenfft, filename= '/APPS/metadata/RESPS/RESP.IU.' + \
                            'TUC.' + loc + '.LHZ',  date = UTCDateTime('2017-001T00:00:00.0'), station = 'TUC',
                            channel = 'LHZ', network = 'IU', locid = loc, units = 'VEL') 
    resp = resp[1:]
    resp = np.absolute(resp*np.conjugate(resp))
    #paz= sp.get_paz('IU.TUC.' + loc + '.LHZ', UTCDateTime('2017-001T00:00:00.0'))
    #resp = computeresp(paz,1., lenfft)
    
    resps[loc] = np.abs(resp)






db_range = (-220, -80, 1)
numbins = int((db_range[1] - db_range[0])/db_range[2]) 
db_bins = np.linspace(db_range[0], db_range[1], numbins+1, endpoint=True)



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
sday =1
eday=150
hourC=4
for day in range(sday,eday):
    print('On day: ' + str(day))
    st = read('/msd/IU_TUC/2017/' + str(day).zfill(3) + '/*LHZ*')
    st.sort(['location'])
    
    for hour in range(0,24,hourC):
        try:
        #if True:
            stTemp=st.copy()
            stTemp.trim(st[0].stats.starttime + hour*60*60., st[0].stats.starttime + (hour+ hourC)*60.*60.)

            n,p,f = selfnoise(stTemp, resps)
            ns0 =np.append(ns0, n[0])
            ps0 = np.append(ps0,p[0])
            ns1 =np.append(ns1, n[1])
            ps1 = np.append(ps1,p[1])
            ns2 =np.append(ns2, n[2])
            ps2 = np.append(ps2,p[2])
            freqs = np.append(freqs,f)
        except:
            print('Day ' + str(day) + ' bad')
    #fh=open('NOISE' + str(day).zfill(3) , 'w')
    #for idx, fre in enumerate(f):
        #fh.write(str(f[idx]) + ', ' + str(p[0][idx]) + ', ' + str(p[1][idx]) + ', ' + str(p[2][idx]) + ', ' + str(n[0][idx]) + ', ' + str(n[1][idx]) + ', ' + str(n[2][idx]) + '\n') 
    #fh.close()

        
ns0 = np.asarray(ns0)      
ns1 = np.asarray(ns1) 
ns2 = np.asarray(ns2)      
fedge=list(set(freqs))
fedge = np.sort(fedge)
fedge = np.asarray(fedge)
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
    npows0.append(np.mean(ns0[freqs == ele]))
    npows1.append(np.mean(ns1[freqs == ele]))
    npows2.append(np.mean(ns2[freqs == ele]))
    npows0std.append(np.std(ns0[freqs == ele]))
    npows1std.append(np.std(ns1[freqs == ele]))
    npows2std.append(np.std(ns2[freqs == ele]))
    pows0.append(np.mean(ps0[freqs == ele]))
    pows1.append(np.mean(ps1[freqs == ele]))
    pows2.append(np.mean(ps2[freqs == ele]))
    pows0std.append(np.std(ps0[freqs == ele]))
    pows1std.append(np.std(ps1[freqs == ele]))
    pows2std.append(np.std(ps2[freqs == ele]))


npows0=np.asarray(npows0)
npows1=np.asarray(npows1)
npows2=np.asarray(npows2)
lppow0 = npows0[(.01 >= fedge) & (fedge >= .001)]
lppow1 = npows1[(.01 >= fedge) & (fedge >= .001)]
lppow2 = npows2[(.01 >= fedge) & (fedge >= .001)]
freqslp=fedge[(.01 >= fedge) & (fedge >= .001)]


def fun(x, b,c):
    return (x)**b + c

from scipy.optimize import curve_fit

popt0, pcov0 = curve_fit(fun, 1/freqslp, lppow0)
popt1, pcov1 = curve_fit(fun, 1/freqslp, lppow1)
popt2, pcov1 = curve_fit(fun, 1/freqslp, lppow2)
print(popt0)
print(popt1)
print(popt2)
#p=np.polyfit(1./freqslp, lppow, 2)
#print(p)
#p = np.poly1d(p)


fig = plt.figure(1,figsize=(12,12))
plt.subplots_adjust(hspace=0.001)
plt.subplot(211)
plt.semilogx(1/fedge,pows0, 'g', label='STS-1',linewidth=1.5)
plt.semilogx(1/fedge,pows1,'b',  label='STS-2.5',linewidth=1.5)
plt.semilogx(1/fedge,pows2,'r', label='STS-6',linewidth=1.5)
plt.text(5,-87,'A)', fontsize=24)
model_periods, high_noise = get_nhnm()
plt.semilogx(model_periods, high_noise, '0.4', linewidth=2)
model_periods, low_noise = get_nlnm()
plt.semilogx(model_periods, low_noise, '0.4', linewidth=2,label='NLNM/NHNM')
plt.xlim((4.,1000))
plt.ylim(-200,-80)
plt.yticks([-180., -160., -140., -120., -100.])
plt.xticks([])
plt.xlabel('Period (s)')

plt.legend(loc=1)
plt.subplot(212)
plt.semilogx(1/fedge,npows0,'g',linewidth=1.5, label='STS-1 Self-Noise')
plt.semilogx(1/fedge,npows1,'b',linewidth=1.5, label='STS-2.5 Self-Noise')
plt.semilogx(1/fedge,npows2, 'r',linewidth=1.5, label='STS-6 Self-Noise')
plt.text(5,-87,'B)', fontsize=24)
plt.subplot(212)
plt.semilogx(1/fedge, fun(1/fedge,*popt0),'g--', linewidth=2., label='STS-1 1/f Fit: ' + str(round(popt0[0],4)))
plt.semilogx(1/fedge, fun(1/fedge,*popt1),'b--', linewidth=2., label='STS-2.5 1/f Fit: ' + str(round(popt1[0],4)))
plt.semilogx(1/fedge, fun(1/fedge,*popt2),'r--', linewidth=2., label='STS-6 1/f Fit: ' + str(round(popt2[0],4)))
model_periods, high_noise = get_nhnm()
plt.semilogx(model_periods, high_noise, '0.4', linewidth=2)
model_periods, low_noise = get_nlnm()
plt.semilogx(model_periods, low_noise, '0.4', linewidth=2,label='NLNM/NHNM')
plt.yticks([-180., -160., -140., -120., -100.])
plt.xlim((4.,1000))
plt.ylim(-200,-80)
plt.xlabel('Period (s)')

plt.legend(loc=1)
fig.text(0.05, 0.5,'Power (dB rel. 1 $(m/s^2)^2/Hz$)', ha='center', va='center', rotation='vertical')
#plt.show()
plt.savefig('TUCPower.jpg',format='JPEG', dpi=400)
plt.savefig('TUCPower.pdf', format='PDF', dpi=400)
plt.clf()
plt.close()


#Hn0, fedges, powsedges = np.histogram2d(freqs, ns0, bins=(fedge,db_bins), normed=True)
#Hp0 , fedges, powsedges = np.histogram2d(freqs, ps0, bins=(fedge, db_bins), normed=True)
#Hn0 = Hn0 *100 
#Hp0 = Hp0*100
#Hn1, fedges, powsedges = np.histogram2d(freqs, ns1, bins=(fedge,db_bins), normed=True)
#Hp1 , fedges, powsedges = np.histogram2d(freqs, ps1, bins=(fedge, db_bins), normed=True)
#Hn1 = Hn1 *100 
#Hp1 = Hp1*100
#Hn2, fedges, powsedges = np.histogram2d(freqs, ns2, bins=(fedge,db_bins), normed=True)
#Hp2 , fedges, powsedges = np.histogram2d(freqs, ps2, bins=(fedge, db_bins), normed=True)
#Hn2 = Hn2 *100 
#Hp2 = Hp2*100

#print(np.max(Hn0.T))
#X, Y = np.meshgrid(fedges, powsedges)
#fig = plt.figure(2,figsize=(12,6))
#plt.subplots_adjust(wspace=0.001)
#ax1 = fig.add_subplot(121)
##ax1.set_title('PSD STS-1')
#ppsd2=ax1.pcolormesh(1./X, Y, Hp0.T, cmap= LinearSegmentedColormap('mcnamara', CDICT, 1024))
#plt.text(5,-85,'PSD STS-1 TUC')
##cb = plt.colorbar(ppsd2, ax=ax1)
##cb.set_label("(%)")
##color_limits = (0, 30.)
#ax1.set_xlim((1./max(fedge), 1./min(fedge)))
#plt.ylabel('Power (dB rel. 1 $(m/s^2)^2/Hz$)')
#ax1.semilogx()
#ax1.set_xlabel('Period (s)')
#ax1.set_ylim((-210., -70.0))
#ax1.plot(model_periods, high_noise, '0.4', linewidth=2)
#ax1.plot(model_periods, low_noise, '0.4', linewidth=2)
#ax2= fig.add_subplot(122)
##ax2.set_title('Self-Noise STS-1')

#ppsd2=ax2.pcolormesh(1./X, Y, Hn0.T, cmap= LinearSegmentedColormap('mcnamara', CDICT, 1024))
#cb = plt.colorbar(ppsd2, ax=ax2, ticks=[0.,10., 20., 30.])
#plt.text(5,-85,'Self-Noise STS-1 TUC')
##ppsd2.set_label('(%)')
#cb.set_label('(\%)')
##color_limits = (0, 30.)
#ppsd2.set_clim((0.,30.))
#cb.set_clim((0.,30.))
#ax2.plot(model_periods, high_noise, '0.4', linewidth=2)
#ax2.plot(model_periods, low_noise, '0.4', linewidth=2)
#ax2.semilogx()
#ax2.set_xlim((1./max(fedge), 1./min(fedge)))
#ax2.set_ylim((-210, -70))
#ax2.set_xlabel('Period (s)')
#plt.yticks([])

#plt.savefig('STS1Noise.jpg',format='JPEG',dpi=400)
#plt.savefig('STS1Noise.pdf', format='PDF', dpi=400)    
#plt.clf()
#plt.close()
##############################################################################################3
#X, Y = np.meshgrid(fedges, powsedges)
#fig = plt.figure(3,figsize=(12,6))
#plt.subplots_adjust(wspace=0.001)
#ax1 = fig.add_subplot(121)
##ax1.set_title('PSD STS-1')
#ppsd2=ax1.pcolormesh(1./X, Y, Hp1.T, cmap= LinearSegmentedColormap('mcnamara', CDICT, 1024))
#plt.text(5,-85,'PSD STS-2.5 TUC')
##cb = plt.colorbar(ppsd2, ax=ax1)
##cb.set_label("(%)")
##color_limits = (0, 30.)
#ax1.set_xlim((1./max(fedge), 1./min(fedge)))
#plt.ylabel('Power (dB rel. 1 $(m/s^2)^2/Hz$)')
#ax1.semilogx()
#ax1.set_xlabel('Period (s)')
#ax1.set_ylim((-210., -70.0))
#ax1.plot(model_periods, high_noise, '0.4', linewidth=2)
#ax1.plot(model_periods, low_noise, '0.4', linewidth=2)
#ax2= fig.add_subplot(122)
##ax2.set_title('Self-Noise STS-1')

#ppsd2=ax2.pcolormesh(1./X, Y, Hn1.T, cmap= LinearSegmentedColormap('mcnamara', CDICT, 1024))
#cb = plt.colorbar(ppsd2, ax=ax2, ticks=[0.,10., 20., 30.])
#plt.text(5,-85,'Self-Noise STS-2.5 TUC')
##ppsd2.set_label('(%)')
#cb.set_label('(\%)')
##color_limits = (0, 30.)
#ppsd2.set_clim((0.,30.))
#cb.set_clim((0.,30.))
#ax2.plot(model_periods, high_noise, '0.4', linewidth=2)
#ax2.plot(model_periods, low_noise, '0.4', linewidth=2)
#ax2.semilogx()
#ax2.set_xlim((1./max(fedge), 1./min(fedge)))
#ax2.set_ylim((-210, -70))
#ax2.set_xlabel('Period (s)')
#plt.yticks([])

#plt.savefig('STS25Noise.jpg',format='JPEG',dpi=400)
#plt.savefig('STS25Noise.pdf', format='PDF', dpi=400)    
#plt.clf()
#plt.close()

##############################################################################################3
#X, Y = np.meshgrid(fedges, powsedges)
#fig = plt.figure(4,figsize=(12,6))
#plt.subplots_adjust(wspace=0.001)
#ax1 = fig.add_subplot(121)
##ax1.set_title('PSD STS-1')
#ppsd2=ax1.pcolormesh(1./X, Y, Hp2.T, cmap= LinearSegmentedColormap('mcnamara', CDICT, 1024))
#plt.text(5,-85,'PSD STS-6 TUC')
##cb = plt.colorbar(ppsd2, ax=ax1)
##cb.set_label("(%)")
##color_limits = (0, 30.)
#ax1.set_xlim((1./max(fedge), 1./min(fedge)))
#plt.ylabel('Power (dB rel. 1 $(m/s^2)^2/Hz$)')
#ax1.semilogx()
#ax1.set_xlabel('Period (s)')
#ax1.set_ylim((-210., -70.0))
#ax1.plot(model_periods, high_noise, '0.4', linewidth=2)
#ax1.plot(model_periods, low_noise, '0.4', linewidth=2)
#ax2= fig.add_subplot(122)
##ax2.set_title('Self-Noise STS-1')

#ppsd2=ax2.pcolormesh(1./X, Y, Hn2.T, cmap= LinearSegmentedColormap('mcnamara', CDICT, 1024))
#cb = plt.colorbar(ppsd2, ax=ax2, ticks=[0.,10., 20., 30.])
#plt.text(5,-85,'Self-Noise STS-6 TUC')
##ppsd2.set_label('(%)')
#cb.set_label('(\%)')
##color_limits = (0, 30.)
#ppsd2.set_clim((0.,30.))
#cb.set_clim((0.,30.))
#ax2.plot(model_periods, high_noise, '0.4', linewidth=2)
#ax2.plot(model_periods, low_noise, '0.4', linewidth=2)
#ax2.semilogx()
#ax2.set_xlim((1./max(fedge), 1./min(fedge)))
#ax2.set_ylim((-210, -70))
#ax2.set_xlabel('Period (s)')
#plt.yticks([])

#plt.savefig('STS6Noise.jpg',format='JPEG',dpi=400)
#plt.savefig('STS6Noise.pdf', format='PDF', dpi=400)    
#plt.clf()
#plt.close()
