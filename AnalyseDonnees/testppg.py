import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy import stats

def filter_signal(signal, th):
    f_s = fft_filter(signal, th)
    return np.real(np.fft.ifft(f_s))
def fft_filter(signal, perc):
    fft_signal = np.fft.fft(signal)
    fft_abs = np.abs(fft_signal)
    th=perc*(2*fft_abs[0:int(len(signal)/2.)]/len(signal)).max()
    fft_tof=fft_signal.copy()
    fft_tof_abs=np.abs(fft_tof)
    fft_tof_abs=2*fft_tof_abs/len(signal)
    fft_tof[fft_tof_abs<=th]=0
    return fft_tof
def fft_filter_amp(signal, th):
    fft = np.fft.fft(signal)
    fft_tof=fft.copy()
    fft_tof_abs=np.abs(fft_tof)
    fft_tof_abs=2*fft_tof_abs/len(signal)
    fft_tof_abs[fft_tof_abs<=th]=0
    return fft_tof_abs[0:int(len(fft_tof_abs)/2.)]

'''
for i in range(2):
    for j in range(7):
        df = pd.read_csv('C:\\Users\\Master\\Desktop\\Emteq\\DonneesTest\\Donnees\\Part1\\Results\\'+str(i)+'_'+str(j)+'_ppg.csv', sep=';')
        dfPPG = df.loc[:, 'Ppg']
        dfProximity = df.loc[:, 'Proximity']

        ax = dfPPG.plot()
        dfProximity.plot(ax=ax)
        ax.set_title('Participant '+str(i)+', Video '+str(j))
        plt.show()
'''
'''
for j in range(7):
    df = pd.read_csv('C:\\Users\\Master\\Desktop\\Emteq\\DonneesTest\\Records\\Part1\\Results\\0_'+str(j)+'_emg.csv', sep=';')
    df = df.loc[:, 'LeftZygomaticus']

    ax = df.plot()
    ax.set_title('Participant 0, Video '+str(j))
    plt.show()
'''
for j in range(7):
    df = pd.read_csv('C:\\Users\\Master\\Desktop\\Emteq\\DataExpe\\DataExpe\\Part1\\0_'+str(j)+'_ppg.csv', sep=';')
    #x = list(df.index.values)
    x = df.loc[:, 'timestamp'].to_list()
    xMin = x[0]
    xMax = x[-1]
    y = df.loc[:, 'Ppg'].to_list()
    y = np.array(y)
    peaks, _ = find_peaks(y, prominence=1000, distance=40)
    print(len(peaks)/(xMax - xMin)*60)

    plt.plot(y)
    plt.plot(peaks, y[peaks], 'x')
    plt.title('Participant 0, Video '+str(j))
    plt.show()


    th_list = np.linspace(0,0.02,1000)
    th_list = th_list[0:len(th_list)]
    p_values = []
    corr_values = []
    for t in th_list:
        filt_signal = filter_signal(y, t)
        res = stats.spearmanr(y,y-filt_signal)
        p_values.append(res.pvalue)
        corr_values.append(res.correlation)
    #plt.figure(figsize=(20,10))
    #plt.subplot(1,2,1)
    #plt.scatter(th_list,corr_values,s=2,color='navy')
    #plt.plot(th_list,p_values)
    #plt.ylabel('Correlation Value')
    #plt.xlabel('Threshold Value')
    #plt.subplot(1,2,2)
    #plt.plot(th_list,p_values,color='navy')
    #plt.plot(th_list,p_values)
    #plt.ylabel('P-Value')
    #plt.xlabel('Threshold Value')
    #plt.show()

    th_opt = th_list[np.array(corr_values).argmin()]
    opt_signal = filter_signal(y, th_opt)
    #plt.plot(x,y,color='navy',label='Original Signal')
    #plt.plot(x,opt_signal,color='firebrick',label='Optimal signal (Th=%.3f)'%(th_opt))
    #plt.plot(x,(y-opt_signal),color='darkorange',label='Difference')
    #plt.xlim(xMin, xMax)
    #plt.xlabel('Time')
    #plt.ylabel('Signal')
    #plt.legend()
    #plt.show()

    peaks, _ = find_peaks(opt_signal, prominence=1000, distance=40)
    print(len(peaks)/(xMax - xMin)*60)

    plt.plot(opt_signal)
    plt.plot(peaks, opt_signal[peaks], 'x')
    plt.title('Participant 0, Video '+str(j))
    plt.show()