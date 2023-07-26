import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from io import StringIO

'''EMG_SENSOR_NAMES = ['RightOrbicularis', 'RightZygomaticus', 'RightFrontalis', 'CenterCorrugator',
                    'LeftFrontalis', 'LeftZygomaticus', 'LeftOrbicularis']'''
EMG_SENSOR_NAMES = ['RightZygomaticus', 'CenterCorrugator', 'LeftZygomaticus']
AXES = ['x', 'y', 'z']

nbParticipants = 15

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

    df = pd.read_csv(StringIO(data), skip_blank_lines=True, delimiter = ',', na_filter=False, low_memory=True)
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

def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)
         
    plt.plot([], c=color_code, label=label)
    plt.legend()

#orbi_left = 'Emg/Amplitude[LeftOrbicularis]'
#orbi_right = 'Emg/Amplitude[RightOrbicularis]'
zygo_left = 'Emg/Amplitude[LeftZygomaticus]'
zygo_right = 'Emg/Amplitude[RightZygomaticus]'
#front_left = 'Emg/Amplitude[LeftFrontalis]'
#front_right = 'Emg/Amplitude[RightFrontalis]'
corr = 'Emg/Amplitude[CenterCorrugator]'
hr = 'HeartRate/Average'
    
#emg_channel_names = ['Orbicularis - left', 'Frontalis - left', 'Frontalis - right', 'Orbicularis - right']
emg_channel_names = ['Zygomaticus - left', 'Corrugator', 'Zygomaticus - right']
hr_channel_name = 'Heart Rate - Average'

videoIDsToRemove = [['5'],['3'],[],[],[],['0'],[],['2', '9', '11'],['8'],[],[],[],['10'],[],[]]

low_valence_video_label = ['3', '4', '9', '10']
high_valence_video_label = ['7', '0', '2', '6', '8', '5', '11', '1', '12']
low_arousal_video_label = ['1', '7', '2', '8']
high_arousal_video_label = ['5', '11', '6', '12']

videoGroup=['Low Valence Videos', 'High Valence Videos', 'Low Arousal Videos', 'High Arousal Videos']

low_valence_video = pd.DataFrame()
high_valence_video = pd.DataFrame()
low_arousal_video = pd.DataFrame()
high_arousal_video = pd.DataFrame()

for i in range(nbParticipants):
    print(i)
    path_to_csv_file = 'C:\\Users\\Master\\Desktop\\Emteq\\DataExpe\\DataExpe\\'+str(i)+'_Part1.csv'
    path_to_csv_file2 = 'C:\\Users\\Master\\Desktop\\Emteq\\DataExpe\\DataExpe\\'+str(i)+'_Part2.csv'
    path_to_event_markers_file = 'C:\\Users\\Master\\Desktop\\Emteq\\DataExpe\\DataExpe\\'+str(i)+'_Part1.json'
    path_to_event_markers_file2 = 'C:\\Users\\Master\\Desktop\\Emteq\\DataExpe\\DataExpe\\'+str(i)+'_Part2.json'
    data_csv, emg_order = importdataCSV2DF(path_to_csv_file)
    data_csv2, emg_order2 = importdataCSV2DF(path_to_csv_file2)
    data, events_labels, events_timestamps = synchronize_data_with_event_markers(data_csv, path_to_event_markers_file)
    data2, events_labels2, events_timestamps2 = synchronize_data_with_event_markers(data_csv2, path_to_event_markers_file2)
    data, events_labels, events_timestamps = remove_rows(data, events_labels, events_timestamps, videoIDsToRemove[i])
    data2, events_labels2, events_timestamps2 = remove_rows(data2, events_labels2, events_timestamps2, videoIDsToRemove[i])
    timestamps = data['Time']
    timestamps2 = data2['Time']

    #columnsList = [orbi_left, orbi_right, front_left, front_right]
    columnsList = [zygo_left, zygo_right, corr]
    data.loc[:, columnsList] = data[columnsList].apply(lambda x: (x-x.min())/(x.max()-x.min()))
    data2.loc[:, columnsList] = data2[columnsList].apply(lambda x: (x-x.min())/(x.max()-x.min()))

    for label in low_valence_video_label:
        low_valence_video = pd.concat([low_valence_video, data.loc[data.event == label], data2.loc[data2.event == label]], ignore_index=True)

    for label in high_valence_video_label:
        high_valence_video = pd.concat([high_valence_video, data.loc[data.event == label], data2.loc[data2.event == label]], ignore_index=True)

    for label in low_arousal_video_label:
        low_arousal_video = pd.concat([low_arousal_video, data.loc[data.event == label], data2.loc[data2.event == label]], ignore_index=True)
    
    for label in high_arousal_video_label:
        high_arousal_video = pd.concat([high_arousal_video, data.loc[data.event == label], data2.loc[data2.event == label]], ignore_index=True)
    
#orbi_left_data = [low_valence_video[orbi_left], high_valence_video[orbi_left], low_arousal_video[orbi_left], high_arousal_video[orbi_left]]
#orbi_right_data = [low_valence_video[orbi_right], high_valence_video[orbi_right], low_arousal_video[orbi_right], high_arousal_video[orbi_right]]
zygo_left_data = [low_valence_video[zygo_left], high_valence_video[zygo_left], low_arousal_video[zygo_left], high_arousal_video[zygo_left]]
zygo_right_data = [low_valence_video[zygo_right], high_valence_video[zygo_right], low_arousal_video[zygo_right], high_arousal_video[zygo_right]]
#front_left_data = [low_valence_video[front_left], high_valence_video[front_left], low_arousal_video[front_left], high_arousal_video[front_left]]
#front_right_data = [low_valence_video[front_right], high_valence_video[front_right], low_arousal_video[front_right], high_arousal_video[front_right]]
corr_data = [low_valence_video[corr], high_valence_video[corr], low_arousal_video[corr], high_arousal_video[corr]]
hr_data_mean = [low_valence_video[hr].mean(), high_valence_video[hr].mean(), low_arousal_video[hr].mean(), high_arousal_video[hr].mean()]
hr_data_std = [low_valence_video[hr].std(), high_valence_video[hr].std(), low_arousal_video[hr].std(), high_arousal_video[hr].std()]
hr_data = [low_valence_video[hr], high_valence_video[hr], low_arousal_video[hr], high_arousal_video[hr]]

zygo_left_plot = plt.boxplot(zygo_left_data, positions=np.array(np.arange(len(zygo_left_data)))*2.0-0.35, widths=0.3, showfliers=False)
corr_plot = plt.boxplot(corr_data, positions=np.array(np.arange(len(corr_data)))*2.0, widths=0.3, showfliers=False)
zygo_right_plot = plt.boxplot(zygo_right_data, positions=np.array(np.arange(len(zygo_right_data)))*2.0+0.35, widths=0.3, showfliers=False)

define_box_properties(zygo_left_plot, 'red', emg_channel_names[0])
define_box_properties(corr_plot, 'blue', emg_channel_names[1])
define_box_properties(zygo_right_plot, 'green', emg_channel_names[2])

plt.title('Comparison of signal amplitude per video type')
plt.xticks(np.arange(0, len(videoGroup) * 2, 2), videoGroup)
plt.ylabel('Normalized amplitude')
plt.show()

'''orbi_left_plot = plt.boxplot(orbi_left_data, positions=np.array(np.arange(len(orbi_left_data)))*2.0-0.55, widths=0.3, showfliers=False)
front_left_plot = plt.boxplot(front_left_data, positions=np.array(np.arange(len(front_left_data)))*2.0-0.2, widths=0.3, showfliers=False)
front_right_plot = plt.boxplot(front_right_data, positions=np.array(np.arange(len(front_right_data)))*2.0+0.2, widths=0.3, showfliers=False)
orbi_right_plot = plt.boxplot(orbi_right_data, positions=np.array(np.arange(len(orbi_right_data)))*2.0+0.55, widths=0.3, showfliers=False)

define_box_properties(orbi_left_plot, 'red', emg_channel_names[0])
define_box_properties(front_left_plot, 'blue', emg_channel_names[1])
define_box_properties(front_right_plot, 'green', emg_channel_names[2])
define_box_properties(orbi_right_plot, 'orange', emg_channel_names[3])

plt.title('Comparison of signal amplitude per video type')
plt.xticks(np.arange(0, len(videoGroup) * 2, 2), videoGroup)
plt.ylabel('Normalized amplitude')
plt.show()'''

hr_plot = plt.boxplot(hr_data, positions=np.array(np.arange(len(hr_data))), widths=0.3, showfliers=False)
plt.title('Comparison of heart rate per video type')
plt.xticks(np.arange(0, len(videoGroup) * 1, 1), videoGroup)
plt.ylabel('Heart rate')
plt.show()

hr_plot = plt.bar(np.array(np.arange(len(hr_data_mean)))*1.0, hr_data_mean, yerr=hr_data_std)
plt.title('Comparison of mean heart rate and std per video type')
plt.xticks(np.arange(0, len(videoGroup) * 1, 1), videoGroup)
plt.ylabel('Heart rate')
plt.show()