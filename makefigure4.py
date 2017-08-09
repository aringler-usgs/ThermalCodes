###This code calculates the RMS values of temperature, velocity, and acceleration
###outputs for each temperature variable step we induced. A curve is then fit to the
###data to find the threshold of stability, which we set as the corner of the fit.




#!/usr/bin/env python

from obspy.core import UTCDateTime, read
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

#Set font parameters from matplotlib
mpl.rc('font',family='serif')
mpl.rc('font',serif='Times') 
mpl.rc('text', usetex=True)
mpl.rc('font',size=20)

#List of all stations (repeated for each time period).
stas = ['XX_TST4_00', 'XX_TST5_00', 'XX_TST5_10','XX_TST6_00',
        'XX_TST4_00', 'XX_TST5_00', 'XX_TST5_10','XX_TST6_00', 
        'XX_TST4_00', 'XX_TST5_00', 'XX_TST5_10','XX_TST6_00',
        'XX_TST4_00', 'XX_TST5_00', 'XX_TST5_10','XX_TST6_00',
        'XX_TST4_00', 'XX_TST5_00', 'XX_TST5_10','XX_TST6_00',
        'XX_TST4_00', 'XX_TST5_00', 'XX_TST5_10','XX_TST6_00',
        'XX_TST4_00', 'XX_TST5_00', 'XX_TST5_10','XX_TST6_00',
        'XX_TST4_00', 'XX_TST5_00', 'XX_TST5_10','XX_TST6_00',
        'XX_TST4_00', 'XX_TST5_00', 'XX_TST5_10','XX_TST6_00']
chans = ['LH0']

#List of temperature variable time periods (repeated for each station).
times = [UTCDateTime('2017-190T12:00:00.0'),UTCDateTime('2017-190T12:00:00.0'), UTCDateTime('2017-190T12:00:00.0'), UTCDateTime('2017-190T12:00:00.0'),
        UTCDateTime('2017-194T12:00:00.0'), UTCDateTime('2017-194T12:00:00.0'), UTCDateTime('2017-194T12:00:00.0'), UTCDateTime('2017-194T12:00:00.0'),
        UTCDateTime('2017-196T12:00:00.0'), UTCDateTime('2017-196T12:00:00.0'), UTCDateTime('2017-196T12:00:00.0'), UTCDateTime('2017-196T12:00:00.0'),
        UTCDateTime('2017-199T12:00:00.0'), UTCDateTime('2017-199T12:00:00.0'), UTCDateTime('2017-199T12:00:00.0'), UTCDateTime('2017-199T12:00:00.0'),  
        UTCDateTime('2017-203T12:00:00.0'), UTCDateTime('2017-203T12:00:00.0'), UTCDateTime('2017-203T12:00:00.0'), UTCDateTime('2017-203T12:00:00.0'),
        UTCDateTime('2017-207T19:30:00.0'), UTCDateTime('2017-207T19:30:00.0'), UTCDateTime('2017-207T19:30:00.0'), UTCDateTime('2017-207T19:30:00.0'),
        UTCDateTime('2017-210T00:00:00.0'), UTCDateTime('2017-210T00:00:00.0'), UTCDateTime('2017-210T00:00:00.0'), UTCDateTime('2017-210T00:00:00.0'),
        UTCDateTime('2017-214T20:00:00.0'), UTCDateTime('2017-214T20:00:00.0'), UTCDateTime('2017-214T20:00:00.0'), UTCDateTime('2017-214T20:00:00.0'),
        UTCDateTime('2017-219T12:00:00.0'), UTCDateTime('2017-219T12:00:00.0'), UTCDateTime('2017-219T12:00:00.0'), UTCDateTime('2017-219T12:00:00.0')]

#Setting end time to 24 hours
etime = 24.*60.*60.
#Initialize lists of data
vels = []
accels = []
temps = []
#List of letters to be used when plotting later.
letters = ['A)', 'B)']


#A function that assigns the response of each instrument based on its location. 
def returnresponse(sta,loc):
    if sta == 'TST4':
        #paz data for Trillium 120 P
        pazvel = {'zeros': [0., 0., -9., -160.7, -3108.],
                'poles': [-0.03852 + 0.03658j, -0.03852 - 0.03658j, -178., -135. +160.j, -135. - 160.j,
                        -671. + 1514.j, -671. - 1514.j], 'gain': 3.08*(10**5), 'sensitivity': 1201.*(2**26/40.)}
        pazaccel = {'zeros': [0., -9., -160.7, -3108.],
                'poles': [-0.03852 + 0.03658j, -0.03852 - 0.03658j, -178., -135. +160.j, -135. - 160.j,
                        -671. + 1514.j, -671. - 1514.j], 'gain': 3.08*(10**5), 'sensitivity': 20*1201.*(2**26/40.)}
        #Sets label and color to be used when graphing later.
        label = 'Trillium 120'
        color = 'purple'
    elif sta =='TST5': 
        #paz data for STS-2.5
        pazvel = {'zeros': [0., 0., -15.78, -630.2, -556.-936.19j, -556.+936.19j,-973.8],
                'poles': [-0.03702 + 0.03702j, -0.03702 - 0.03701j, -16.041, -16.041, -327.3 -74.14j,
                -327.3+74.14j, -97], 'gain': 0.000157286, 'sensitivity': 1500.*(2**26/40.)}
        pazaccel = {'zeros': [0., -15.78, -630.2, -556.-936.19j, -556.+936.19j,-973.8],
            'poles': [-0.03702 + 0.03702j, -0.03702 - 0.03701j, -16.041, -16.041, -327.3 -74.14j,
                -327.3+74.14j, -97], 'gain': 0.000157286, 'sensitivity': 1500.*(2**26/40.)}
        #Sets label and color to be used when graphing later.
        if loc == '00':
            label = 'STS-2.5'
            color = 'r'
        else:
            label = 'STS-2.5 in Water Bricks'
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
        #Sets label and color to be used when graphing later.
        label = 'Trillium Compact'
        color = 'g'

    return pazvel, pazaccel, label, color
                
fig = plt.figure(1, figsize=(14,14))
plt.subplots_adjust(hspace=0.001)

#Creates pairs out of each station and time in the lists above        
for pair in zip(stas, times):
    for chan in chans:
        #Sets the station parameter as the first item in the pair, and time
        #as the second
        sta =pair[0]
        time= pair[1]
        #Splits SNCLs into their components.
        net,sta,loc = sta.split('_') 
        #Read in data
        string = '/tr1/telemetry_days/'+ net + '_' + sta + '/' + str(time.year) \
                + '/' + str(time.year) +'_' + str(time.julday).zfill(3) + '/' + loc + '_' + chan + '*'
        st = read(string)
        try:
            string = '/tr1/telemetry_days/'+ net + '_' + sta + '/' + str(time.year) \
                    + '/' + str(time.year) +'_' + str((time+24.*60.*60.).julday).zfill(3) + '/' + loc + '_' + chan + '*'
            st += read(string)
        except:
            #Print if data cannot be read in.
            print(str(time.julday).zfill(3) + 'bad')
        st.merge()
        st.trim(time, time + etime)
        print(st)
        
        #Reads in temperature sensor data based off of which sensor is collocated with which sensor.
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
        
        #Creates a range of times in hours.
        t = np.arange(0, st[0].stats.npts)/(60.*60.)
        #Creates variables for the paz values of velocity and acceleration, as well as the
        #label and color for each sensor.
        pazvel, pazaccel, label, color = returnresponse(sta,loc)
        stvel = st.copy()
        staccel = st.copy()
        #remove instrument response
        stvel.simulate(paz_remove = pazvel)
        staccel.simulate(paz_remove = pazaccel)
        #1000s lowpass filter
        stvel.filter('lowpass', freq = 0.001)
        staccel.filter('lowpass', freq = 0.001)
        
        #Put lists into RMS values from peak-to-peak.
        vels.append(stvel[0].std())
        accels.append(staccel[0].std())
        temps.append(sttemp[0].std()/335544.)
        #Plotting velocity RMS vs. temperature RMS.
        plt.subplot(211)
        plt.scatter(sttemp[0].std()/335544., stvel[0].std(), s = 200., color = color)
        #Plotting black line at corner, which corresponds to 0.002 degC.
        plt.plot([0.001999,0.002001], [0.0001,302.], linewidth=2, color = 'k')
        plt.ylim([0.01,300.])
        #Set y scale as logarithmic.
        plt.yscale('log')
        plt.ylabel('RMS of Velocity $(m/s)$', labelpad = 11, fontsize = 20)
        #remove x ticks.
        plt.xticks([])
        
        #Plotting acceleration RMS vs. temperature RMS.
        plt.subplot(212)
        plt.scatter(sttemp[0].std()/335544., 1000.*staccel[0].std(), label = label if time.julday == 190 else '_nolegend_',  color = color, s = 200.)
        #Plotting black line at corner, which corresponds to 0.002 degC.
        plt.plot([0.001999,0.002001], [0.0001,100.], linewidth=2, color = 'k')
        plt.ylim([0.001,5])
        #Set y scale as logarithmic
        plt.yscale('log')
        plt.legend(fontsize='large', loc = 'lower right')
        plt.ylabel('RMS of Acceleration $(mm/s^2)$', labelpad = 11, fontsize = 20)
        plt.xlabel('RMS of Temperature $({}^{\circ} C)$', fontsize = 20)


#This is where we fit the scatter to a curve.


#Initialize lists of values.
temps1 = []
accels1 = []
vels1 = []
temps2 = []
accels2 = []
vels2 = []
temps3 = []
accels3 = []
vels3 = []

#For each station,...
for idx,sta in enumerate(stas):
    
    #Import values of temperature, acceleration, and velocity calculated above.
    if sta == 'XX_TST4_00':
        temps1.append(temps[idx])
        vels1.append(vels[idx])
        accels1.append(accels[idx])
    elif sta == 'XX_TST5_00':
        temps2.append(temps[idx])
        vels2.append(vels[idx])
        accels2.append(accels[idx])
    elif sta == 'XX_TST6_00':
        temps3.append(temps[idx])
        vels3.append(vels[idx])
        accels3.append(accels[idx])
#Turn lists into arrays.
temps1 = np.asarray(temps1)
temps2 = np.asarray(temps2)
temps3 = np.asarray(temps3)
vels1 = np.asarray(vels1)
vels2 = np.asarray(vels2)
vels3 = np.asarray(vels3)
#Turn lists into arrays and multipy by 1000 to convert to mm/s**2.
accels1 = 1000.*np.asarray(accels1)
accels2 = 1000.*np.asarray(accels2)
accels3 = 1000.*np.asarray(accels3)

#Define the function with which we will fit our data.
def fun(T, a):
    return a*T


from scipy.optimize import curve_fit
#Create vector of temperature steps.
Ts = np.linspace(-0.01,0.1,400)

#Calculate optimization for each set of sensors for velocity and acceleration.
popt1,pcov1 = curve_fit(fun, temps1, vels1, [1])
popt2,pcov2 = curve_fit(fun, temps2, vels2, [1])
popt3, pcov3 = curve_fit(fun, temps3, vels3, [1])
popt4,pcov4 = curve_fit(fun, temps1, accels1, [1])
popt5,pcov5 = curve_fit(fun, temps2, accels2, [1])
popt6, pcov6 = curve_fit(fun, temps3, accels3, [1])
print(popt1)
print(popt2)
print(popt3)
print(popt4)
print(popt5)
print(popt6)
print(vels2)
print(accels2)
print(temps2)

#Plot fit for velocity RMS vs. temeprature RMS.
plt.subplot(211)
plt.plot(Ts, fun(Ts,*popt1), color = 'purple', linewidth = 1.5)
plt.plot(Ts, fun(Ts,*popt2), color = 'r', linewidth = 1.5)
plt.plot(Ts, fun(Ts, *popt3), color = 'g', linewidth = 1.5)
plt.text(-0.0015, 50., letters[0], fontsize=25.)
plt.xlim([-0.002,0.08])
#Plot fits for acceleration RMS vs. temperature RMS.
plt.subplot(212)
plt.plot(Ts, fun(Ts,*popt4), color = 'purple', linewidth = 1.5)
plt.plot(Ts, fun(Ts,*popt5), color = 'r', linewidth = 1.5)
plt.plot(Ts, fun(Ts, *popt6), color = 'g', linewidth = 1.5)
plt.xlim([-0.002,0.08])
plt.text(-0.0015, 2.5, letters[1], fontsize = 25.)

plt.savefig('velaccel_' + str(time.julday) + '.jpg', format= 'JPEG', dpi = 400)
plt.savefig('velaccel_' + str(time.julday) + '.pdf', format= 'PDF', dpi = 400)
plt.show()
    

