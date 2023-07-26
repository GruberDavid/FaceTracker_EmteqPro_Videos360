import pandas as pd
import matplotlib.pyplot as plt

#pd.set_option("display.max_rows", 999)

folderPart1 = 'C:\\Users\\Master\\Desktop\\Emteq\\DataExpe\\DataExpe\\Part1\\'
folderPart2 = 'C:\\Users\\Master\\Desktop\\Emteq\\DataExpe\\DataExpe\\Part2\\'

nbVideos = 13
nbParticipants = 15

videoIDsToRemove = [['5'],['3'],[],[],[],['0'],[],['2', '9', '11'],['8'],[],[],[],['10'],[],[]]

valenceDB = [6.07, 6.57, 6.13, 2.38, 3.69, 6.27, 6.17, 6, 6.19, 3.33, 4.39, 6.46, 6.75]
arousalDB = [4, 1.57, 1.8, 4.25, 3.94, 6.18, 7.17, 1.6, 1.81, 3.33, 2.77, 6.91, 7.42]

#region Plot Subjective / Objective Arousal + Valence
dataframe = pd.DataFrame()

for idParticipant in range(0, nbParticipants):
    df = pd.read_csv(folderPart1+str(idParticipant)+'_rating.csv', sep=';')
    df = df[~df['VideoID'].isin(list(map(int, videoIDsToRemove[idParticipant])))]
    df2 = pd.read_csv(folderPart2+str(idParticipant)+'_rating.csv', sep=';')
    df2 = df2[~df2['VideoID'].isin(list(map(int, videoIDsToRemove[idParticipant])))]
    dataframe = pd.concat([dataframe, df, df2])

dataframe = dataframe.reset_index(drop=True)

valDf = dataframe.loc[dataframe['VideoID'] == 0, 'Valence'].reset_index(drop=True).rename(str(0))
for idVideo in range(1, nbVideos):
    df = dataframe.loc[dataframe['VideoID'] == idVideo, 'Valence'].reset_index(drop=True).rename(str(idVideo))
    valDf = pd.concat([valDf, df], axis=1)
valenceSub = valDf.mean().to_list()

arDf = dataframe.loc[dataframe['VideoID'] == 0, 'Arousal'].reset_index(drop=True).rename(str(0))
for idVideo in range(1, nbVideos):
    df = dataframe.loc[dataframe['VideoID'] == idVideo, 'Arousal'].reset_index(drop=True).rename(str(idVideo))
    arDf = pd.concat([arDf, df], axis=1)
arousalSub = arDf.mean().to_list()



fig, ax = plt.subplots()
plt.title('Comparison of arousal and valence per video')
ax.scatter(valenceDB, arousalDB, c='blue', marker='+', label='Dataset value', s=100)
ax.scatter(valenceSub, arousalSub, c='red', marker='+', label='Subjective value', s=100)
ax.set_xlim(0, 10)
ax.set_xlabel('Valence')
ax.set_ylim(0,10)
ax.set_ylabel('Arousal')
ax.legend()
plt.show()
#endregion

#region Arousal / Valence boxplots
dataframe = pd.DataFrame()

for idParticipant in range(0, nbParticipants):
    df = pd.read_csv(folderPart1+str(idParticipant)+'_rating.csv', sep=';')
    df = df[~df['VideoID'].isin(list(map(int, videoIDsToRemove[idParticipant])))]
    df2 = pd.read_csv(folderPart2+str(idParticipant)+'_rating.csv', sep=';')
    df2 = df2[~df2['VideoID'].isin(list(map(int, videoIDsToRemove[idParticipant])))]
    dataframe = pd.concat([dataframe, df, df2])

newDataframe = dataframe.loc[dataframe['VideoID'] == 0, 'Valence'].reset_index(drop=True).rename(str(0))
for idVideo in range(1, nbVideos):
    df = dataframe.loc[dataframe['VideoID'] == idVideo, 'Valence'].reset_index(drop=True).rename(str(idVideo))
    newDataframe = pd.concat([newDataframe, df], axis=1)

fig, axes = plt.subplots(2)
fig.suptitle('Comparison of arousal and valence per video')
axes[0].set_title('Valence')
axes[0].set_ylim(0, 10)
newDataframe.plot.box(ax=axes[0], whis=(0, 100), showmeans=True, meanline=True)
axes[0].scatter(list(range(1, len(valenceDB)+1)), valenceDB, color='red', marker='+', label='Dataset value', s=100)
axes[0].legend()

newDataframe = dataframe.loc[dataframe['VideoID'] == 0, 'Arousal'].reset_index(drop=True).rename(str(0))
for idVideo in range(1, nbVideos):
    df = dataframe.loc[dataframe['VideoID'] == idVideo, 'Arousal'].reset_index(drop=True).rename(str(idVideo))
    newDataframe = pd.concat([newDataframe, df], axis=1)

axes[1].set_title('Arousal')
axes[1].set_ylim(0, 10)
newDataframe.plot.box(ax=axes[1], whis=(0, 100), showmeans=True, meanline=True)
axes[1].scatter(list(range(1, len(arousalDB)+1)), arousalDB, color='red', marker='+', label='Dataset value', s=100)
axes[1].legend()
plt.show()
#endregion

