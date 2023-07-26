import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

nbVideos = 13
nbParticipants = 15

videoIDsToRemove = [['5'],['3'],[],[],[],['0'],[],['2', '9', '11'],['8'],[],[],[],['10'],[],[]]

dataframe = pd.DataFrame({'VideoID':list(range(nbVideos)),
                          'Dataset Valence':[6.07, 6.57, 6.13, 2.38, 3.69, 6.27, 6.17, 6, 6.19, 3.33, 4.39, 6.46, 6.75],
                          'Dataset Arousal':[4, 1.57, 1.8, 4.25, 3.94, 6.18, 7.17, 1.6, 1.81, 3.33, 2.77, 6.91, 7.42]}, 
                          columns=['VideoID', 'Dataset Valence', 'Subjective Valence', 'Dataset Arousal', 'Subjective Arousal'])

for idVideo in range(7):
    averageArousal = 0
    averageValence = 0
    for idParticipant in range(nbParticipants):
        if str(idVideo) not in videoIDsToRemove[idParticipant]:
            df = pd.read_csv('C:\\Users\\Master\\Desktop\\Emteq\\DataExpe\\DataExpe\\Part1\\'+str(idParticipant)+'_rating.csv', sep=';')
            arousal = df.loc[df['VideoID'] == idVideo, 'Arousal'].values[0]
            valence = df.loc[df['VideoID'] == idVideo, 'Valence'].values[0]
            if averageArousal == 0:
                averageArousal = arousal
            else:
                averageArousal = (averageArousal + arousal) / 2
            if averageValence == 0:
                averageValence = valence
            else:
                averageValence = (averageValence + valence) / 2
    dataframe.loc[dataframe['VideoID'] == idVideo, 'Subjective Valence'] = averageValence
    dataframe.loc[dataframe['VideoID'] == idVideo, 'Subjective Arousal'] = averageArousal

for idVideo in range(7, nbVideos):
    averageArousal = 0
    averageValence = 0
    for idParticipant in range(nbParticipants):
        if str(idVideo) not in videoIDsToRemove[idParticipant]:
            df = pd.read_csv('C:\\Users\\Master\\Desktop\\Emteq\\DataExpe\\DataExpe\\Part2\\'+str(idParticipant)+'_rating.csv', sep=';')
            arousal = df.loc[df['VideoID'] == idVideo, 'Arousal'].values[0]
            valence = df.loc[df['VideoID'] == idVideo, 'Valence'].values[0]
            if averageArousal == 0:
                averageArousal = arousal
            else:
                averageArousal = (averageArousal + arousal) / 2
            if averageValence == 0:
                averageValence = valence
            else:
                averageValence = (averageValence + valence) / 2
    dataframe.loc[dataframe['VideoID'] == idVideo, 'Subjective Valence'] = averageValence
    dataframe.loc[dataframe['VideoID'] == idVideo, 'Subjective Arousal'] = averageArousal

print(dataframe)

fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
table = ax.table(cellText=dataframe.values, colLabels=dataframe.columns, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
plt.show()