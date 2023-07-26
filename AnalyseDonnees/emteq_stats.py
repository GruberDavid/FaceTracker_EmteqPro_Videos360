import pandas as pd
import numpy as np
import os
from scipy import stats
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
from io import StringIO

EMG_SENSOR_NAMES = ['RightOrbicularis', 'RightZygomaticus', 'RightFrontalis', 'CenterCorrugator',
                    'LeftFrontalis', 'LeftZygomaticus', 'LeftOrbicularis']
AXES = ['x', 'y', 'z']

def emg_columns(emg_sensor, numerical=False):
    return [f'{emg_sensor}[{x}]' for x in EMG_SENSOR_NAMES]

def imu_columns(imu_sensor):
    return [f'{imu_sensor}{x}' for x in AXES]
    
def importdataCSV2DF(filename_csv): 
    '''
    """ The function reads the .csv file from a specified path and 
    formats the data into the required structure for further processing.
     
    Parameters
    ----------
    filename_csv : str
        Path to where the .csv file is stored. 
     
    Returns
    -------
    df : pandas.DataFrame
        Formatted dataframe containing the sensors data.
    """
    '''
    
    _file = open(filename_csv, 'r')
    data = _file.read()
    _file.close()

    metadata = [line for line in data.split('\n') if '#' in line]
    for line in metadata:
        if line.find('Frame#') == -1:
            data=data.replace("{}".format(line),'', 1)
        if line.find('#Time/Seconds.referenceOffset') != -1:
            time_offset = float(line.split(',')[1])
        if line.find('#Emg/Properties.rawToVoltageDivisor') != -1:
            emg_divisor = float(line.split(',')[1])
        if line.find('#Emg/Properties.contactToImpedanceDivisor') != -1:
            impedance_divisor = float(line.split(',')[1])
        if line.find('#Imu/Properties.accelerationDivisor') != -1 or line.find('#Accelerometer/Properties.rawDivisor') != -1:
            acceleration_divisor = float(line.split(',')[1])
        if line.find('#Imu/Properties.magnetometerDivisor') != -1 or line.find('#Magnetometer/Properties.rawDivisor') != -1:
            magnetometer_divisor = float(line.split(',')[1])
        if line.find('#Imu/Properties.gyroscopeDivisor') != -1 or line.find('#Gyroscope/Properties.rawDivisor') != -1:
            gyroscope_divisor = float(line.split(',')[1])
        if line.find('#Emg/Properties.ids[]') != -1:
            emg_order = line.split(',')[1:]
    data = re.sub(r'\n\s*\n', '\n', data, re.MULTILINE)
    data = re.sub(r'\s*\n\s*Frame#','Frame#', data, re.MULTILINE)

    df = pd.read_csv(StringIO(data), skip_blank_lines=True, delimiter = ',', na_filter=False, low_memory=False)
    df = df.replace('', np.nan)
    df = df.dropna()

    #convert raw values to appropriate unit
    if "Emg/Contact[RightOrbicularis]" in df.columns:  # check the version if the EMG sensor names are with names or numbers 
        df[emg_columns('Emg/Amplitude')] = df[emg_columns('Emg/Amplitude')].astype('float') / emg_divisor
        df[emg_columns('Emg/Filtered')] = df[emg_columns('Emg/Filtered')].astype('float') / emg_divisor
        df[emg_columns('Emg/Raw')] = df[emg_columns('Emg/Raw')].astype('float') / emg_divisor
        df[emg_columns('Emg/Contact')] = df[emg_columns('Emg/Contact')].astype('float') / impedance_divisor
        df[imu_columns('Accelerometer/Raw.')] = df[imu_columns('Accelerometer/Raw.')].astype('float') / acceleration_divisor
        #df[imu_columns('Magnetometer/Raw.')] = df[imu_columns('Magnetometer/Raw.')].astype('float') / magnetometer_divisor
        df[imu_columns('Gyroscope/Raw.')] = df[imu_columns('Gyroscope/Raw.')].astype('float') / gyroscope_divisor
    else:
        df[emg_columns('Emg/Amplitude', True)] = df[emg_columns('Emg/Amplitude', True)].astype('float') / emg_divisor
        df[emg_columns('Emg/Filtered', True)] = df[emg_columns('Emg/Filtered', True)].astype('float') / emg_divisor
        df[emg_columns('Emg/Raw', True)] = df[emg_columns('Emg/Raw', True)].astype('float') / emg_divisor
        df[emg_columns('Emg/Contact', True)] = df[emg_columns('Emg/Contact', True)].astype('float') / impedance_divisor
        df[imu_columns('Imu/Accelerometer.')] = df[imu_columns('Imu/Accelerometer.')].astype('float') / acceleration_divisor
        #df[imu_columns('Imu/Magnetometer.')] = df[imu_columns('Imu/Magnetometer.')].astype('float') / magnetometer_divisor
        df[imu_columns('Imu/Gyroscope.')] = df[imu_columns('Imu/Gyroscope.')].astype('float') / gyroscope_divisor

    df[['HeartRate/Average']] = df[['HeartRate/Average']].astype('float') 

    #calculate unix timestamps
    unix_timestamps = time_offset + 946684800
    unix_timestamps = df['Time'].astype('float') + unix_timestamps 
    df.insert(loc=2, column='UnixTime', value=unix_timestamps)
    return df, emg_order

def synchronize_data_with_event_markers(df, path_to_event_markers):
    '''
    """ The function synchronize the sensors data with the event markers,
    based on unix timestamps. The resulting dataframe contains 'event' column
    reffering to the event marker.
    
    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing the sensors data. 
    path_to_event_markers : str
        Path to where the event markers .json file is stored. 
        
    Returns
    -------
    df : pandas.DataFrame
        A formatted dataframe containing the sensors data and an 'event' column
        reffering to the event marker.
    events : list
        A list containing all event labels.
    events_timestamps : list
        A list containing all event timestamps (starting and ending time 
        of each video/event).
    """
    '''
    
    annotation = pd.read_json(path_to_event_markers, convert_dates=False)
    annotation["unix"] = ((annotation["Timestamp"] + 946684800000)/1000).astype("float")
    
    annotation.drop(annotation.loc[annotation['Label'].str.startswith("MaskOnFace")].index, inplace=True)
    annotation.drop(annotation.loc[annotation['Label'].str.startswith("MaskOffFace")].index, inplace=True)
    annotation.reset_index(inplace=True)
    
    for event in annotation.Label.unique():
        tmp_ann = annotation.loc[annotation.Label == event]
        if tmp_ann.loc[tmp_ann['Type'] == 0].shape[0] != 0:
            for ann_idx in range(tmp_ann.loc[tmp_ann['Type'] == 0].shape[0]):
                timestamp = tmp_ann.loc[tmp_ann['Type'] == 0].iloc[ann_idx]['TimestampUnix']
                df.at[df.loc[df.Time <= timestamp].index[-1], 'event'] =  event
        tmp_ann = tmp_ann.drop(tmp_ann.loc[tmp_ann.Type == 0].index)
        for ann_idx in range(int(tmp_ann.shape[0]/2)):
            start = tmp_ann.loc[tmp_ann['Type'] == 1].iloc[ann_idx]['TimestampUnix']
            end = tmp_ann.loc[tmp_ann['Type'] == 2].iloc[ann_idx]['TimestampUnix']
            df.loc[(df.Time >= start) & (df.Time <= end), 'event'] = event
    return df, np.array(annotation['Label']), np.array(annotation['TimestampUnix'])

def remove_rows(dataframe, events, events_timestamps, videoIDsToRemove):
    df = dataframe[dataframe['event'].notna()]
    df = df[df['event']!='participant']
    for i in videoIDsToRemove:
        df = df[df['event']!=i]

    new_events = []
    new_events_timestamps = []
    for j in range(len(events)):
        if (events[j] not in videoIDsToRemove) and (events[j] != 'participant'):
            new_events.append(events[j])
            new_events_timestamps.append(events_timestamps[j])
    
    return df, new_events, new_events_timestamps

def plot_sensor_data_events(title,y_label,sensor_data,sensor_timestamps,event_labels,event_timings,channel_names=''):
    '''
    """ This function is used to plot sensors data and mark events starting and ending time.
    
    Parameters
    ----------
    title : str
        The title of the figure. 
    y_label : str
        The label of the y axis on the figure.
    sensor_data : pandas.DataFrame
        A dataframe containing the sensor data used for the plot.
    sensor_timestamps : pandas.Series
        Sensor data timestamps.
    event_labels : list
        Names of events.
    event_timings : list
        List of starting and ending timestamps of each event.
    channel_names :  list
        List of names of the sensors in the plot.
        
    Returns
    -------
    None.
    """
    '''

    xMin = event_timings[1]
    xMax = event_timings[-2]
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, len(event_timings))]
    ax = plt.figure(figsize=(15, 5.5), dpi=100)
    ax = plt.plot(sensor_timestamps, sensor_data)
    for idx, event in enumerate(event_timings):
        color = colors[idx]
        ax = plt.axvline(event, label = event_labels[idx], c = color)
    ax = plt.title(title)
    #plt.xticks(event_timings, [datetime.utcfromtimestamp(ts).strftime('%H:%M:%S') for ts in event_timings])
    #plt.locator_params(nbins=9)
    legend1 = plt.legend(prop={'size':9}, title='Event Markers', bbox_to_anchor=(1.0, 1.01), loc='upper left')
    ax = plt.gca().add_artist(legend1)
    if channel_names != '':
        legend2 = plt.legend(channel_names)
        ax = plt.gca().add_artist(legend2)
    plt.ylabel(r'$'+y_label+'$', labelpad=6)
    plt.xlabel(r'$Time$', labelpad=6)
    plt.show()
    
def plot(df, event_timings, event_labels, column, label, legend="", ylim="", y_threshold='', y_threshold_label=''):
    '''
    This function plots the given column in a matplotlib plot
    
    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing the data for the wanted plot
        
    column : str
        A string containing the name of the plotting column
    
    label : str
        The y-axes title that should be displayed on the plot
        
    ylim : list of numbers
        List that contains two numbers that describe the y-axes,
        the first number being the lower limit and the second one, the upper limit of the plot
    Returns
    -------
    None.
    '''
    if column not in df.columns:
        return
    
    df = df.iloc[::1000,:]  #one prediction per second
    df.reset_index(drop=True, inplace=True)
    plt.figure(figsize=(20,3))
    sns.lineplot(data=df, x=df.Time, y=df[column])
    plt.ylabel(label)
    plt.xlabel('Time (s)')
    legend_vals = [label]
    if ylim != "":
        plt.ylim(ylim[0], ylim[1])
    if y_threshold != '':
        plt.hlines(y_threshold, linestyle='dashed', colors='r', alpha=0.5, xmin=df.Time.iloc[0], xmax=df.Time.iloc[-1])
        legend_vals = [label, y_threshold_label]
    if event_timings != '':
        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, len(event_timings))]
        for idx, event in enumerate(event_timings):
            ax = plt.axvline(event, label = event_labels[idx], c = colors[idx])
        legend1 = plt.legend(prop={'size':9}, title='Event Markers', bbox_to_anchor=(1.0, 1.01), loc='upper left')
        ax = plt.gca().add_artist(legend1)
    if legend == '':
        legend2 = plt.legend(legend_vals)
        ax = plt.gca().add_artist(legend2)
    else:
        legend2 = plt.legend([legend])
        ax = plt.gca().add_artist(legend2)
    plt.show()



av_zygo_left_low_ar = 0
av_zygo_left_high_ar = 0
av_zygo_left_negative = 0

av_corr_low_ar = 0
av_corr_high_ar = 0
av_corr_negative = 0

av_zygo_right_low_ar = 0
av_zygo_right_high_ar = 0
av_zygo_right_negative = 0

for i in range(8):
    #path_to_csv_file = 'C:\\Users\\Master\\Desktop\\Emteq\\DataExpe\\DataExpe\\'+str(i)+'_Part1.csv'
    path_to_csv_file = 'C:\\Users\\Master\\Desktop\\Emteq\\DataExpe\\DataExpe\\'+str(i)+'_Part2.csv'
    #path_to_event_markers_file = 'C:\\Users\\Master\\Desktop\\Emteq\\DataExpe\\DataExpe\\'+str(i)+'_Part1.json'
    path_to_event_markers_file = 'C:\\Users\\Master\\Desktop\\Emteq\\DataExpe\\DataExpe\\'+str(i)+'_Part2.json'
    data_csv, emg_order = importdataCSV2DF(path_to_csv_file)
    data, events_labels, events_timestamps = synchronize_data_with_event_markers(data_csv, path_to_event_markers_file)
    timestamps = data['Time']

    zygo_left = 'Emg/Amplitude[LeftZygomaticus]'
    zygo_right = 'Emg/Amplitude[RightZygomaticus]'
    corr = 'Emg/Amplitude[CenterCorrugator]'
        
    emg_zygo_corr = data[[zygo_left, zygo_right, corr]]
    emg_channel_names = ['Zygomaticus - left', 'Zygomaticus - right', 'Corrugator']

    '''
    plot_sensor_data_events(str(i) + ' : EMG Amplitude (RMS)',
                                'Volts',
                                emg_zygo_corr,
                                timestamps,
                                events_labels,
                                events_timestamps,
                                emg_channel_names)

    heart_rate_data = data[['HeartRate/Average']]

    plot_sensor_data_events(str(i) + ' : Heart Rate', 
                                'Beats per minute',
                                heart_rate_data,
                                timestamps,
                                events_labels,
                                events_timestamps)
    '''

    #low_arousal_video_label = ['1', '2']
    low_arousal_video_label = ['7', '8']
    #high_arousal_video_label = ['5', '6']
    high_arousal_video_label = ['11', '12']
    #negative_video_label = ['3','4']
    negative_video_label = ['9','10']
    
    videoIDsToRemove = [['5'],['3'],[],[],[],['0'],[],['2', '9', '11']]

    data, events_labels, events_timestamps = remove_rows(data, events_labels, events_timestamps, videoIDsToRemove[i])

    low_ar_video = pd.DataFrame()
    for label in low_arousal_video_label:
        low_ar_video = pd.concat([low_ar_video, data.loc[data.event == label]], ignore_index=True)

    zygo_left_low_ar = low_ar_video[zygo_left]
    zygo_right_low_ar = low_ar_video[zygo_right]
    corr_low_ar = low_ar_video[corr]

    high_ar_video = pd.DataFrame()
    for label in high_arousal_video_label:
        high_ar_video = pd.concat([high_ar_video, data.loc[data.event == label]], ignore_index=True)

    zygo_left_high_ar = high_ar_video[zygo_left]
    zygo_right_high_ar = high_ar_video[zygo_right]
    corr_high_ar = high_ar_video[corr]

    negative_video = pd.DataFrame()
    for label in negative_video_label:
        negative_video = pd.concat([negative_video, data.loc[data.event == label]], ignore_index=True)

    zygo_left_negative = negative_video[zygo_left]
    zygo_right_negative = negative_video[zygo_right]
    corr_negative = negative_video[corr]

    videos=['Low Arousal Videos','High Arousal Videos', 'Low Valence Videos']
    average_zygo_left_low_ar = zygo_left_low_ar.mean()
    average_zygo_left_high_ar = zygo_left_high_ar.mean()
    average_zygo_left_negative = zygo_left_negative.mean()
    if av_zygo_left_low_ar == 0:
        av_zygo_left_low_ar = (av_zygo_left_low_ar + average_zygo_left_low_ar)
    else:
        av_zygo_left_low_ar = (av_zygo_left_low_ar + average_zygo_left_low_ar) / 2
    if av_zygo_left_high_ar == 0:
        av_zygo_left_high_ar = (av_zygo_left_high_ar + average_zygo_left_high_ar)
    else:
        av_zygo_left_high_ar = (av_zygo_left_high_ar + average_zygo_left_high_ar) / 2
    if av_zygo_left_negative == 0:
        av_zygo_left_negative = (av_zygo_left_negative + average_zygo_left_negative)
    else:
        av_zygo_left_negative = (av_zygo_left_negative + average_zygo_left_negative) / 2

    average_corr_low_ar = corr_low_ar.mean()
    average_corr_high_ar = corr_high_ar.mean()
    average_corr_negative = corr_negative.mean()
    if av_corr_low_ar == 0:
        av_corr_low_ar = (av_corr_low_ar + average_corr_low_ar)
    else:
        av_corr_low_ar = (av_corr_low_ar + average_corr_low_ar) / 2
    if av_corr_high_ar == 0:
        av_corr_high_ar = (av_corr_high_ar + average_corr_high_ar)
    else:
        av_corr_high_ar = (av_corr_high_ar + average_corr_high_ar) / 2
    if av_corr_negative == 0:
        av_corr_negative = (av_corr_negative + average_corr_negative)
    else:
        av_corr_negative = (av_corr_negative + average_corr_negative) / 2

    average_zygo_right_low_ar = zygo_right_low_ar.mean()
    average_zygo_right_high_ar = zygo_right_high_ar.mean()
    average_zygo_right_negative = zygo_right_negative.mean()
    if av_zygo_right_low_ar == 0:
        av_zygo_right_low_ar = (av_zygo_right_low_ar + average_zygo_right_low_ar)
    else:
        av_zygo_right_low_ar = (av_zygo_right_low_ar + average_zygo_right_low_ar) / 2
    if av_zygo_right_high_ar == 0:
        av_zygo_right_high_ar = (av_zygo_right_high_ar + average_zygo_right_high_ar)
    else:
        av_zygo_right_high_ar = (av_zygo_right_high_ar + average_zygo_right_high_ar) / 2
    if av_zygo_right_negative == 0:
        av_zygo_right_negative = (av_zygo_right_negative + average_zygo_right_negative)
    else:
        av_zygo_right_negative = (av_zygo_right_negative + average_zygo_right_negative) / 2

    '''
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))

    ZL = [average_zygo_left_low_ar, average_zygo_left_high_ar, average_zygo_left_negative]
    ZR = [average_zygo_right_low_ar, average_zygo_right_high_ar, average_zygo_right_negative]
    CG = [average_corr_low_ar, average_corr_high_ar, average_corr_negative]

    br1 = np.arange(len(ZL))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    plt.bar(br1, ZL, color ='r', width = barWidth,
            edgecolor ='grey', label ='Zygomaticus - Left')
    plt.bar(br2, CG, color ='b', width = barWidth,
            edgecolor ='grey', label ='Corrugator')
    plt.bar(br3, ZR, color ='g', width = barWidth,
            edgecolor ='grey', label ='Zygomaticus - Right')


    plt.xticks([r + barWidth for r in range(len(ZL))],
            videos)
    plt.title(str(i) + ' : Comparison of Average Zygomaticus and Corrugator Activation in Positive and Negative Videos')
    plt.legend()
    plt.show()
    '''

barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

ZL = [av_zygo_left_low_ar, av_zygo_left_high_ar, av_zygo_left_negative]
ZR = [av_zygo_right_low_ar, av_zygo_right_high_ar, av_zygo_right_negative]
CG = [av_corr_low_ar, av_corr_high_ar, av_corr_negative]

br1 = np.arange(len(ZL))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]

plt.bar(br1, ZL, color ='r', width = barWidth,
        edgecolor ='grey', label ='Zygomaticus - Left')
plt.bar(br2, CG, color ='b', width = barWidth,
        edgecolor ='grey', label ='Corrugator')
plt.bar(br3, ZR, color ='g', width = barWidth,
        edgecolor ='grey', label ='Zygomaticus - Right')


plt.xticks([r + barWidth for r in range(len(ZL))],
        videos)
plt.title('Comparison of Average Zygomaticus and Corrugator Activation in Positive and Negative Videos')
plt.legend()
plt.show()