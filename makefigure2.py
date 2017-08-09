###This calculates the raw time series data for all of our sensors under temperature variability,
###and one sensor under water bricks that sees extremely little variability. The temperature
###variation is also plotted. This corresponds to Figure 2.



#!/usr/bin/env python

from obspy.core import read, UTCDateTime, Stream
import matplotlib.pyplot as plt
import numpy as np


#Set font parameters from matplotlib. 
import matplotlib as mpl
mpl.rc('font',family='serif')
mpl.rc('font',serif='Times') 
mpl.rc('text', usetex=True)
mpl.rc('font',size=20)



#SNCLs and the name of each sensor in lists sot that they can be called and labelled later.
sncls = ['XX_TST5_10_LH0', 'XX_TST5_00_LH0', 'XX_TST4_00_LH0','XX_TST6_00_LH0', 'XX_TST6_XT_LK1']
sensDes=['A) STS-2.5 in Water Bricks','B) STS-2.5', 'C) Trillium 120','D) Trillium Compact', 'E) Temperature Variation']

#Range of days over which we take our data.
days =range(188,200)

fig = plt.figure(1,figsize=(12,8))
plt.subplots_adjust(hspace=0.001)



#We pair together each sensor with its appropriate designation to make labelling easier.
for idx, pair in enumerate(zip(sncls,sensDes)):
    net, sta, loc, chan= pair[0].split('_')
    
    #Set the second item in our pair list to be the label for each sensor.
    verbiage=pair[1]
    
    #Import data
    st = Stream()
    for day in days:
        st += read('/msd/' + net + '_' + sta + '/2017/' + str(day).zfill(3) + '/' + loc + '_' + chan + '*')
    
    #Printing checks for any data errors or gaps that we might not know about
    print(st)
    st.merge()
    
    #Demean our data
    st.detrend('constant')
    
    #Create an array of times in seconds so that our x and y dimensions are the same for plotting purposes.
    t = (np.arange(0,len(st[0].data))/st[0].stats.sampling_rate)/(60*60) 
    
    #Sets the size of our subplot to be equal to the total nubmer of stations we have, 
    #and plots each station by index.
    plt.subplot(len(sncls),1,idx+1)
    
    #Plots our data and removes bitweight of the digitizer.
    plt.plot(t,st[0].data/(2**26/40.),color='k')
    plt.xlim((min(t),max(t)))
    
    #Sets it so that only the bottom subplot will have xticks 
    if idx+1 < len(sncls):
        plt.xticks([])
    
    #Sets different y limits for the temperature data, and sets position of label.
    if idx > 3:
        plt.ylim((-.01,.025))
        plt.yticks([0.00,0.02])
        plt.text(1, 0.015, verbiage)
    #Sets y limits and label location for all other subplots.
    else:
        plt.ylim((-.08,.08))
        plt.yticks([-0.05,0.00,0.05])
        plt.text(1, 0.055, verbiage)
     
     
        
#Sets axes labels        
fig.text(0.03, 0.6,'Sensor Output (V)', ha='center', va='center', rotation='vertical', fontsize= 20)
fig.text(0.03, 0.2,'Temperature $({}^{\circ} C)$', ha='center', va='center', rotation='vertical', fontsize= 20)
plt.xlabel('Time (hours)')



plt.savefig('TIMESERIESLONG.jpg',format='JPEG',dpi=400)
plt.savefig('TIMESERIESLONG.pdf',format='PDF',dpi=400)
plt.show()
