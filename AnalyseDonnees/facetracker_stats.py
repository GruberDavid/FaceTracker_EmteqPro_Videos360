import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

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

folderPath = 'C:\\Users\\Master\\Desktop\\Gruber\\DonneesExpe\\DataExpe\\Part1\\'
folderPath2 = 'C:\\Users\\Master\\Desktop\\Gruber\\DonneesExpe\\DataExpe\\Part2\\'

lowValenceList = [3, 9, 4, 10]
highArousalList = [12, 6, 11, 5]
lowArousalList = [1, 7, 2, 8]
highValence = [12, 6, 11, 5, 1, 7, 2, 8]

videoGroup=['Low Valence Videos','High Valence Videos', 'Low Arousal Videos', 'High Arousal Videos']

nbVideos = 13
nbParticipants = 15

videoIDsToRemove = [['5'],['3'],[],[],[],['0'],[],['2', '9', '11'],['8'],[],[],[],['10'],[],[]]

low_valence_video = pd.DataFrame()
high_valence_video = pd.DataFrame()
low_arousal_video = pd.DataFrame()
high_arousal_video = pd.DataFrame()

for idParticipant in range(nbParticipants):
    for idVideo in range(nbVideos):
        print(str(idParticipant), str(idVideo))
        folder = ''
        if idVideo < 7:
            folder = folderPath
        else:
            folder = folderPath2
        
        if str(idVideo) not in videoIDsToRemove[idParticipant]:
                start, end = getTimestamps(folder+str(idParticipant)+'_'+str(idVideo)+'_EventStarted.csv')
                staticDf = staticDataframe(folder+str(idParticipant)+'_'+str(idVideo)+'_mouth_shape.csv', start, end, str(idVideo))
                dynamicDf, _ = dynamicDataframe(folder+str(idParticipant)+'_'+str(idVideo)+'_mouth_shape.csv', start, end, str(idVideo))
                _, qfeDf = qfeDataframe(staticDf, dynamicDf, str(idVideo))
                qfeDf.rename(columns={ qfeDf.columns[0]: 'QFE' }, inplace = True)
                qfeDf = qfeDf.astype(float)
                if idVideo in lowValenceList:
                    low_valence_video = pd.concat([low_valence_video, qfeDf], ignore_index=True)
                if idVideo in highValence:
                    high_valence_video = pd.concat([high_valence_video, qfeDf], ignore_index=True)
                if idVideo in lowArousalList:
                    low_arousal_video = pd.concat([low_arousal_video, qfeDf], ignore_index=True)
                if idVideo in highArousalList:
                    high_arousal_video = pd.concat([high_arousal_video, qfeDf], ignore_index=True)

lv = low_valence_video['QFE']
hv = high_valence_video['QFE']
la = low_arousal_video['QFE']
ha = high_arousal_video['QFE']

data = [lv, hv, la, ha]

plt.boxplot(data, positions=np.array(np.arange(len(data))),showfliers=False, showmeans=True, meanline=True)
plt.title('Comparison of QFE score per video type')
plt.ylabel('Normalized QFE score')
plt.xticks(np.arange(0, len(videoGroup)), videoGroup)
plt.show()
plt.boxplot(data, positions=np.array(np.arange(len(data))), showmeans=True, meanline=True)
plt.title('Comparison of QFE score per video type')
plt.ylabel('Normalized QFE score')
plt.xticks(np.arange(0, len(videoGroup)), videoGroup)
plt.show()
plt.boxplot(data, positions=np.array(np.arange(len(data))),whis=(0,100), showmeans=True, meanline=True)
plt.title('Comparison of QFE score per video type')
plt.ylabel('Normalized QFE score')
plt.xticks(np.arange(0, len(videoGroup)), videoGroup)
plt.show()