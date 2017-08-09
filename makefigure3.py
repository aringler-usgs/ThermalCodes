###This code plots the velocity and acceleration output and the temperature variations
###seen by the variable sensors and the water bricks sensors over one 24-hour period 
###(12 hours on, 12 hours off). This corresponds to Figure 3.


#!/usr/bin/env python

from obspy.core import UTCDateTime, read
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


#Import font parameters from matplotlib
mpl.rc('font',family='serif')
mpl.rc('font',serif='Times') 
mpl.rc('text', usetex=True)
mpl.rc('font',size=24)


#SNCLs from the stations we want.
stas =['XX_TST4_00', 'XX_TST5_00', 'XX_TST5_10','XX_TST6_00']
chans = ['LH0']

#Start time of 24 hours when the heat box was set to 7.8V (deltaT ~0.06)
times = [UTCDateTime('2017-203T12:00:00.0'), UTCDateTime('2017-203T12:00:00.0'), UTCDateTime('2017-203T12:00:00.0'), UTCDateTime('2017-203T12:00:00.0')]

#Sets end time at 24 hours after the start time.
etime = 24.*60.*60.

#List of letters that will later be used to identify the plots.
letters = ['A)', 'B)', 'C)']

#Function that gives the correct poles and zeros for velocity and acceleration depending on which sensor 
#we're looping over. It also sets the labels for the legend and the color of each line once it's plotted.
def returnresponse(sta,loc):
    if sta == 'TST4':
        #paz data for Trillium 120 P
        pazvel = {'zeros': [0., 0., -9., -160.7, -3108.],
                'poles': [-0.03852 + 0.03658j, -0.03852 - 0.03658j, -178., -135. +160.j, -135. - 160.j,
                        -671. + 1514.j, -671. - 1514.j], 'gain': 3.08*(10**5), 'sensitivity': 1201.*(2**26/40.)}
        pazaccel = {'zeros': [0., -9., -160.7, -3108.],
                'poles': [-0.03852 + 0.03658j, -0.03852 - 0.03658j, -178., -135. +160.j, -135. - 160.j,
                        -671. + 1514.j, -671. - 1514.j], 'gain': 3.08*(10**5), 'sensitivity': 20*1201.*(2**26/40.)}
        label1 = 'Trillium 120'
        label2 = 'Temperature for Trillium Sensors'
        color = 'purple'
    elif sta =='TST5': 
        #paz data for STS-2.5
        pazvel = {'zeros': [0., 0., -15.78, -630.2, -556.-936.19j, -556.+936.19j,-973.8],
                'poles': [-0.03702 + 0.03702j, -0.03702 - 0.03701j, -16.041, -16.041, -327.3 -74.14j,
                -327.3+74.14j, -97], 'gain': 0.000157286, 'sensitivity': 1500.*(2**26/40.)}
        pazaccel = {'zeros': [0., -15.78, -630.2, -556.-936.19j, -556.+936.19j,-973.8],
            'poles': [-0.03702 + 0.03702j, -0.03702 - 0.03701j, -16.041, -16.041, -327.3 -74.14j,
                -327.3+74.14j, -97], 'gain': 0.000157286, 'sensitivity': 1500.*(2**26/40.)}
        if loc == '00':
            label1 = 'STS-2.5'
            label2 = 'Temperature for STS-2.5'
            color = 'r'
        else:
            label1 = 'STS-2.5 in Water Bricks'
            label2 = 'Temperature for STS-2.5 in Water Bricks'
            color = 'b'
    elif sta == 'TST6':
        #paz data for Trillium Compact 120s
        pazvel = {'zeros': [0., 0., -392.0, -1960.0, -1490.0 + 1740.0j, -1490.0 - 1740.0j],
            'poles': [-0.03691 + 0.03702j, -0.03691 - 0.03702j, -343.0, -370.0 + 467.0j, -370.0 - 467.0j, -836.0 + 1522.0j,
                    -836.0 - 1522.0j, -4900.0 + 4700.0j, -4900.0 - 4700.0j, -6900.0, -15000.0], 'gain': 4.344928*(10**17), 
                    'sensitivity': 754.3*(2**26/40.)}
        pazaccel = {'zeros': [0., -392.0, -1960.0, -1490.0 + 1740.0j, -1490.0 - 1740.0j],
            'poles': [-0.03691 + 0.03702j, -0.03691 - 0.03702j, -343.0, -370.0 + 467.0j, -370.0 - 467.0j, -836.0 + 1522.0j,
                    -836.0 - 1522.0j, -4900.0 + 4700.0j, -4900.0 - 4700.0j, -6900.0, -15000.0], 'gain': 4.344928*(10**17), 
                    'sensitivity': 754.3*(2**26/40.)}
        label1 = 'Trillium Compact'
        label2 = 'Temperature for Trillium Sensors'
        color = 'g'

    return pazvel, pazaccel, label1, label2, color

                
figure= plt.figure(1, figsize=(14,14))
plt.subplots_adjust(hspace=0.001)

#Pairs together each station with the start time (all start times are the same)        
for pair in zip(stas, times):
    for chan in chans:
        
        #Defines station as the first item in the pair, and time as the second.
        sta =pair[0]
        time= pair[1]
        
        net,sta,loc = sta.split('_')
        #Import data 
        string = '/tr1/telemetry_days/'+ net + '_' + sta + '/' + str(time.year) \
                + '/' + str(time.year) +'_' + str(time.julday).zfill(3) + '/' + loc + '_' + chan + '*'
        st = read(string)
        try:
            string = '/tr1/telemetry_days/'+ net + '_' + sta + '/' + str(time.year) \
                    + '/' + str(time.year) +'_' + str((time+24.*60.*60.).julday).zfill(3) + '/' + loc + '_' + chan + '*'
            st += read(string)
        except:
            pass
        st.merge()
        #trims to the 24-hour period we want
        st.trim(time, time + etime)
        
        #Print to check for gaps or errors.
        print(st)

        #Loops through to read in the temperature sensor that corresponds to each sensor,
        #as they each have their own temperature sensor.
        if loc == '10':
            string = '/tr1/telemetry_days/'+ net + '_' + 'TST6' + '/' + str(time.year) \
                + '/' + str(time.year) +'_' + str(time.julday).zfill(3) + '/' + 'XT' + '_' + 'LK0' + '*'
            sttemp = read(string)
            try:
                string = '/tr1/telemetry_days/'+ net + '_' + 'TST6' + '/' + str(time.year) \
                    + '/' + str(time.year) +'_' + str((time+24.*60.*60.).julday).zfill(3) + '/' + 'XT' + '_' + 'LK0' + '*'
                sttemp += read(string)
            except:
                pass
            sttemp.merge()
            #Trims to the 24-hour period we want
            sttemp.trim(time, time + etime)
            #Demeans the data
            sttemp.detrend('constant')

        elif loc == '00' and sta == 'TST5':
            string = '/tr1/telemetry_days/'+ net + '_' + 'TST2' + '/' + str(time.year) \
                + '/' + str(time.year) +'_' + str(time.julday).zfill(3) + '/' + '00' + '_' + 'LH0' + '*'
            sttemp = read(string)
            try:
                string = '/tr1/telemetry_days/'+ net + '_' + 'TST2' + '/' + str(time.year) \
                    + '/' + str(time.year) +'_' + str((time+24.*60.*60.).julday).zfill(3) + '/' + '00' + '_' + 'LH0' + '*'
                sttemp += read(string)
            except:
                pass
            sttemp.merge()
            sttemp.trim(time, time + etime)
            sttemp.detrend('constant')
        else:
            string = '/tr1/telemetry_days/'+ net + '_' + 'TST6' + '/' + str(time.year) \
                + '/' + str(time.year) +'_' + str(time.julday).zfill(3) + '/' + 'XT' + '_' + 'LK1' + '*'
            sttemp = read(string)
            try:
                string = '/tr1/telemetry_days/'+ net + '_' + 'TST6' + '/' + str(time.year) \
                    + '/' + str(time.year) +'_' + str((time+24.*60.*60.).julday).zfill(3) + '/' + 'XT' + '_' + 'LK1' + '*'
                sttemp += read(string)
            except:
                pass
            sttemp.merge()
            sttemp.trim(time, time + etime)
            sttemp.detrend('constant')

        #60-point moving average is applied to the temperature data to smooth our pulses
        #due to grounding issues.
        N= 60
        sttemp =np.convolve(sttemp[0].data, np.ones((N,))/N, mode='same')
        
        #Makes a list of times in seconds so that x and y dimensions are the same for plotting.
        t = np.arange(0, st[0].stats.npts)/(60.*60.)
        
        #Gives the pazs, labels, and color fo the sensor we're looping over.
        pazvel, pazaccel, label1, label2, color = returnresponse(sta,loc)
        
        #Creates two new streams (one for velocity, one for acceleration) and removes the instrument response.
        stvel = st.copy()
        staccel = st.copy()
        stvel.simulate(paz_remove = pazvel)
        staccel.simulate(paz_remove = pazaccel)
        
        #first subplot plots time series of velocity in m/s
        plt.subplot(311)
        plt.plot(t, stvel[0].data, label=label1, color = color, linewidth = 2)
        plt.xlim((min(t), max(t)))
        #Plots letter label
        plt.text(0.5, 0.05, letters[0])
        
        #Removes x ticks and labels y-axis
        plt.xticks([])
        plt.ylabel('Velocity $(m/s)$', labelpad=23, fontsize = 24)
        
        #second subplot plots time series of acceleration in mm/s**2.
        plt.subplot(312)
        plt.plot(t, (staccel[0].data*1000.), label=label1, color = color, linewidth = 2)
        plt.xlim((min(t), max(t)))
        plt.xticks([])
        plt.ylabel('Acceleration $(mm/s^2)$', labelpad= 36, fontsize = 24)
        plt.text(0.5, 0.012, letters[1])
        #Plots legend for first two subplots.
        plt.legend(fontsize= 15, loc = 'lower right')
        
        #Third subplot for temperature time series
        plt.subplot(313)
        #Dividing by 335544 corrects for the sensitivity of the temperature sensors. If statement in label
        #section makes sure that the Trillium temperature data doesn't come up twice.
        plt.plot(t, (sttemp/335544.),label = label2 if not(sta == 'TST4') else '_nolegend_', color = color)
        plt.xlim((min(t), max(t)))
        #Sets legend for temperature sensors.
        plt.legend(fontsize = 15, loc = 'lower right')
        #Setting x and y labels
        plt.xlabel('Time (hours)', fontsize= 24)
        plt.ylabel('Rel. Temp. $({}^{\circ} C)$', fontsize = 24)
        plt.text(0.5, 0.0004, letters[2])

        
        
plt.savefig('timeseries_' + str(time.julday) + '.jpg', format= 'JPEG', dpi = 200)
plt.savefig('timeseries_' + str(time.julday) + '.pdf', format= 'PDF', dpi = 200)
plt.clf()
plt.close()
    

