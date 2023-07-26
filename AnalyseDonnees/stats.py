import pandas as pd
import math
import matplotlib.pyplot as plt
import os
import threading
import logging
from datetime import datetime
import numpy as np
from pandas.api.types import is_numeric_dtype
import re
from io import StringIO

#folderPath = 'C:\\Users\\Master\\Desktop\\Emteq\\DataExpe\\DataExpe\\Part1\\'
folderPath = 'C:\\Users\\Master\\Desktop\\Emteq\\DataExpe\\DataExpe\\Part2\\'
nbParticipant = 15
#nbVideos = 7
nbVideos = 13

#Find path for a specific file
def findPath(idParticipant, idVideo, name:str):
    fileList = os.listdir(folderPath)
    for i in range(len(fileList)):
        if fileList[i] == str(idParticipant)+'_'+str(idVideo)+'_'+name:
            return folderPath + fileList[i]
    return folderPath

#Get timestamps
def getTimestamps(path):
    timestamps = pd.read_csv(path, sep=';')
    start = timestamps[timestamps.string == 'Démarrage chrono'].iat[0,0]
    end = timestamps[timestamps.string == 'Fin chrono'].iat[-1,0]
    return start, end

#Get Static expressiveness score dataframe
def staticDataframe(dataPath, start, end, label):
    dataframe = pd.read_csv(dataPath, sep=';')
    dataframe = dataframe.loc[:, ~dataframe.columns.str.contains('^Unnamed')] #Remove unnamed columns
    newDataframe = dataframe.loc[(dataframe['timestamp'] >= start) & (dataframe['timestamp'] <= end)] #Only keep the timestamps interval
    newDataframe = newDataframe.drop(columns=['timestamp'])
    newDataframe = newDataframe[['Mouth smile right (0-1 key)', 'Mouth smile left (0-1 key)', 'Mouth sad right (0-1 key)', 'Mouth sad left (0-1 key)']]
    newDataframe = newDataframe.reset_index(drop=True)
    resDf = pd.DataFrame(index=newDataframe.index.copy())
    resDf[label] = ''
    for idx in newDataframe.index:
        sum = 0
        n = 0
        for x in newDataframe.loc[idx]:
            sum += math.exp(float(x))-1
            n += 1
        res = 100/(n*(math.exp(1)-1)) * sum
        resDf[label][idx] = res
    return resDf

#Get Dynamic expressiveness score dataframe + normalized version
def dynamicDataframe(dataPath, start, end, label):
    dataframe = pd.read_csv(dataPath, sep=';')
    dataframe = dataframe.loc[:, ~dataframe.columns.str.contains('^Unnamed')] #Remove unnamed columns
    newDataframe = dataframe.loc[(dataframe['timestamp'] >= start) & (dataframe['timestamp'] <= end)] #Only keep the timestamps interval
    newDataframe = newDataframe.drop(columns=['timestamp'])
    newDataframe = newDataframe[['Mouth smile right (0-1 key)', 'Mouth smile left (0-1 key)', 'Mouth sad right (0-1 key)', 'Mouth sad left (0-1 key)']]
    newDataframe = newDataframe.reset_index(drop=True)
    resDf = pd.DataFrame(index=newDataframe.index.copy())
    resDf[label] = ''
    resDf.iat[0,0] = 0
    for i in range(1, len(newDataframe.index)):
        tExp = 0
        n = 0
        for j in range(len(newDataframe.iloc[i])):
            dX = float(newDataframe.iat[i,j]) - float(newDataframe.iat[i-1,j])
            dT = 1
            dV = dX / dT
            tExp += math.exp(dV)-1
            n += 1
        res = tExp / (n*(math.exp(1)-1))
        resDf.iat[i,0] = res
    normalizedResDf = resDf.copy()
    if normalizedResDf[label].abs().max() == 0:
        return resDf, normalizedResDf
    else:
        normalizedResDf[label] = normalizedResDf[label].abs() / normalizedResDf[label].abs().max()
    return resDf, normalizedResDf

#Get QFE score dataframe + normalized version
def qfeDataframe(staticDf, dynamicDf, label):
    resDf = pd.DataFrame(index=staticDf.index.copy())
    resDf[label] = ''
    for idx in range(len(staticDf.index)):
        res = staticDf.iat[idx, 0] * (1 + dynamicDf.iat[idx, 0])
        resDf.iat[idx, 0] = res
    normalizedResDf = resDf.copy()
    if normalizedResDf[label].abs().max() == 0:
        return resDf, normalizedResDf
    else:
        normalizedResDf[label] = normalizedResDf[label].abs() / 200
    return resDf, normalizedResDf

EMG_SENSOR_NAMES = ['RightOrbicularis', 'RightZygomaticus', 'RightFrontalis', 'CenterCorrugator',
                    'LeftFrontalis', 'LeftZygomaticus', 'LeftOrbicularis']
AXES = ['x', 'y', 'z']

def emg_columns(emg_sensor, numerical=False):
    return [f'{emg_sensor}[{x}]' for x in EMG_SENSOR_NAMES]

def imu_columns(imu_sensor):
    return [f'{imu_sensor}{x}' for x in AXES]
    
def importdataCSV2DF(filename_csv): 
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

def threadFunc(idParticipant):
    global nbVideos
    os.makedirs('Stats', exist_ok=True)

    #path_to_csv_file = 'C:\\Users\\Master\\Desktop\\Emteq\\DataExpe\\DataExpe\\'+str(idParticipant)+'_Part1.csv'
    path_to_csv_file = 'C:\\Users\\Master\\Desktop\\Emteq\\DataExpe\\DataExpe\\'+str(idParticipant)+'_Part2.csv'
    #path_to_event_markers_file = 'C:\\Users\\Master\\Desktop\\Emteq\\DataExpe\\DataExpe\\'+str(idParticipant)+'_Part1.json'
    path_to_event_markers_file = 'C:\\Users\\Master\\Desktop\\Emteq\\DataExpe\\DataExpe\\'+str(idParticipant)+'_Part2.json'
    videoIDsToRemove = [['5'],['3'],[],[],[],['0'],[],['2', '9', '11'],['8'],[],[],[],['10'],[],[]]
    data_csv, _ = importdataCSV2DF(path_to_csv_file)
    data, events_labels, events_timestamps = synchronize_data_with_event_markers(data_csv, path_to_event_markers_file)
    data, events_labels, events_timestamps = remove_rows(data, events_labels, events_timestamps, videoIDsToRemove[idParticipant])
    front_left = 'Emg/Amplitude[LeftFrontalis]'
    front_right = 'Emg/Amplitude[RightFrontalis]'
    orbi_left = 'Emg/Amplitude[LeftOrbicularis]'
    orbi_right = 'Emg/Amplitude[RightOrbicularis]'
    zygo_left = 'Emg/Amplitude[LeftZygomaticus]'
    zygo_right = 'Emg/Amplitude[RightZygomaticus]'
    corr = 'Emg/Amplitude[CenterCorrugator]'
    hr = 'HeartRate/Average'
    emteqData = data[[front_left, front_right, orbi_left, orbi_right, zygo_left, zygo_right, corr, hr, 'event']]
    
    columnsList = [orbi_left, orbi_right, zygo_left, zygo_right, front_left, front_right, corr]
    emteqData.loc[:, columnsList] = (emteqData[columnsList]-emteqData[columnsList].min())/(emteqData[columnsList].max()-emteqData[columnsList].min())
    
    #for idVideo in range(nbVideos):
    for idVideo in range(7, nbVideos):
        if findPath(idParticipant, idVideo, 'EventStarted.csv') != folderPath:
            if str(idVideo) not in videoIDsToRemove[idParticipant]:
                start, end = getTimestamps(findPath(idParticipant, idVideo , 'EventStarted.csv'))
                staticDf = staticDataframe(findPath(idParticipant, idVideo , 'mouth_shape.csv'), start, end, str(idVideo))
                dynamicDf, _ = dynamicDataframe(findPath(idParticipant, idVideo , 'mouth_shape.csv'), start, end, str(idVideo))
                _, qfeDf = qfeDataframe(staticDf, dynamicDf, str(idVideo))
                qfeDf = qfeDf.astype(float)
                resFT = qfeDf.describe()
                resFT = resFT.drop('count')
                resFT.to_csv('Stats/'+str(idParticipant)+'_'+str(idVideo)+'_face_tracker_stats2.csv',';')
            
            emteqDf = emteqData[emteqData['event'] == str(idVideo)]
            if not emteqDf.empty:
                emteqDf = emteqDf.drop('event', axis=1)
                resEmteq = emteqDf.describe()
                resEmteq = resEmteq.drop('count')
                resEmteq.to_csv('Stats/'+str(idParticipant)+'_'+str(idVideo)+'_emteq_stats2.csv',';')
    return

threadList = [] * nbParticipant

'''
for i in range(nbParticipant):
    thread = threading.Thread(target=threadFunc, args=[i], name=('Thread_'+str(i)))
    threadList.append(thread)
    thread.start()

for t in threadList:
    t.join()
'''

for i in range(13, nbParticipant):
    threadFunc(i)