import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, cross_validate
from sklearn.svm import SVC, SVR
from sklearn import neighbors, preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support, make_scorer, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import f_classif, SelectKBest, VarianceThreshold
from sklearn.dummy import DummyClassifier
from sklearn.multioutput import MultiOutputRegressor

pd.set_option("display.max_rows", 999)

folderPath = 'Stats/'
folderPart1 = 'C:\\Users\\Master\\Desktop\\Gruber\\DonneesExpe\\DataExpe\\Part1\\'
folderPart2 = 'C:\\Users\\Master\\Desktop\\Gruber\\DonneesExpe\\DataExpe\\Part2\\'

nbVideos = 13
nbParticipants = 15

lowValenceList = [3, 9, 4, 10]
highArousalList = [12, 6, 11, 5]
lowArousalList = [1, 7, 2, 8]
highValence = [12, 6, 11, 5, 1, 7, 2, 8]

datasetValence = [6.07, 6.57, 6.13, 2.38, 3.69, 6.27, 6.17, 6, 6.19, 3.33, 4.39, 6.46, 6.75]
datasetArousal = [4, 1.57, 1.8, 4.25, 3.94, 6.18, 7.17, 1.6, 1.81, 3.33, 2.77, 6.91, 7.42]

'''featureNames = ['LeftFrontalis:Mean', 'LeftFrontalis:Std', 'LeftFrontalis:Min', 'LeftFrontalis:25%', 'LeftFrontalis:50%', 'LeftFrontalis:75%', 'LeftFrontalis:Max',
                'RightFrontalis:Mean', 'RightFrontalis:Std', 'RightFrontalis:Min', 'RightFrontalis:25%', 'RightFrontalis:50%', 'RightFrontalis:75%', 'RightFrontalis:Max',
                'LeftOrbicularis:Mean', 'LeftOrbicularis:Std', 'LeftOrbicularis:Min', 'LeftOrbicularis:25%', 'LeftOrbicularis:50%', 'LeftOrbicularis:75%', 'LeftOrbicularis:Max',
                'RightOrbicularis:Mean', 'RightOrbicularis:Std', 'RightOrbicularis:Min', 'RightOrbicularis:25%', 'RightOrbicularis:50%', 'RightOrbicularis:75%', 'RightOrbicularis:Max',
                'LeftZygomaticus:Mean', 'LeftZygomaticus:Std', 'LeftZygomaticus:Min', 'LeftZygomaticus:25%', 'LeftZygomaticus:50%', 'LeftZygomaticus:75%', 'LeftZygomaticus:Max',
                'RightZygomaticus:Mean', 'RightZygomaticus:Std', 'RightZygomaticus:Min', 'RightZygomaticus:25%', 'RightZygomaticus:50%', 'RightZygomaticus:75%', 'RightZygomaticus:Max',
                'CenterCorrugator:Mean', 'CenterCorrugator:Std', 'CenterCorrugator:Min', 'CenterCorrugator:25%', 'CenterCorrugator:50%', 'CenterCorrugator:75%', 'CenterCorrugator:Max',
                'HeartRate:Mean', 'HeartRate:Std', 'HeartRate:Min', 'HeartRate:25%', 'HeartRate:50%', 'HeartRate:75%', 'HeartRate:Max',
                'QFE:Mean', 'QFE:Std', 'QFE:Min', 'QFE:25%', 'QFE:50%', 'QFE:75%', 'QFE:Max']'''
featureNames = ['CenterCorrugator:Mean', 'CenterCorrugator:Std',
                'QFE:Mean', 'QFE:Std']
'''featureNames = ['LeftZygomaticus:Mean', 'LeftZygomaticus:Std',
                'RightZygomaticus:Mean', 'RightZygomaticus:Std',
                'CenterCorrugator:Mean', 'CenterCorrugator:Std',
                'QFE:Mean', 'QFE:Std']'''

#classNames = ['Low Valence', 'Low Arousal', 'High Arousal']
classNames = ['High Valence', 'Low Valence']
#classNames = ['High Arousal', 'Low Arousal']



def findPath(idParticipant, idVideo, name:str):
    fileList = os.listdir(folderPath)
    for i in range(len(fileList)):
        if fileList[i] == str(idParticipant)+'_'+str(idVideo)+'_'+name:
            return folderPath + fileList[i]
    return folderPath

def dataframeToList(df):
    dataArray = df.to_numpy()
    dataList = dataArray.flatten()
    dataList = dataList.tolist()
    return dataList

def dataToXY(idParticipant, idVideo):
    if findPath(idParticipant, idVideo, 'emteq_stats2.csv') != folderPath:
            df = pd.read_csv(folderPart1+str(idParticipant)+'_rating.csv', sep=';')
            df2 = pd.read_csv(folderPart2+str(idParticipant)+'_rating.csv', sep=';')
            dfRating = pd.concat([df, df2])
            valence = dfRating.loc[dfRating['VideoID'] == idVideo, 'Valence'].values[0]
            arousal = dfRating.loc[dfRating['VideoID'] == idVideo, 'Arousal'].values[0]
            '''if valence < 5:
                y = 'Low Valence'
            else:
                if arousal < 5:
                    y = 'Low Arousal'
                else:
                    y = 'High Arousal' 
            if idVideo in lowValenceList:
                y = 'Low Valence'
            else:
                if idVideo in highArousalList:
                    y = 'High Arousal'
                else:
                    y = 'Low Arousal' 
            if idVideo in lowValenceList:
                y = 'Low Valence'
            else:
                y = 'High Valence' '''
            if valence < 5:
                y = 'Low Valence'
            else:
                y = 'High Valence' 
            '''if arousal < 5:
                y = 'Low Arousal'
            else:
                y = 'High Arousal' 
            if datasetArousal[idVideo] < 5:
                y = 'Low Arousal'
            else:
                y = 'High Arousal' '''
            #y1 = datasetValence[idVideo]
            #y2 = datasetArousal[idVideo]
            #y1 = valence
            #y2 = arousal
            #y = [y1, y2]

            emteqDf = pd.read_csv(findPath(idParticipant, idVideo, 'emteq_stats2.csv'), sep=';', index_col=0)
            #emteqDf1 = emteqDf.loc[['mean', 'std'],['Emg/Amplitude[LeftZygomaticus]']]
            #emteqDf2 = emteqDf.loc[['mean', 'std'],['Emg/Amplitude[RightZygomaticus]']]
            emteqDf3 = emteqDf.loc[['mean', 'std'],['Emg/Amplitude[CenterCorrugator]']]
            #emteqList = dataframeToList(emteqDf1) + dataframeToList(emteqDf2)# + dataframeToList(emteqDf3)
            emteqList = dataframeToList(emteqDf3)
            ftDf = pd.read_csv(findPath(idParticipant, idVideo, 'face_tracker_stats2.csv'), sep=';', index_col=0)
            ftDf = ftDf.loc[['mean', 'std']]
            ftList = dataframeToList(ftDf)
            x = emteqList + ftList
            #x = ftList
            return x, y



X = []
Y = []

for i in range(nbParticipants):
    for j in range(1,nbVideos):
        videoIDsToRemove = [['5'],['3'],[],[],[],['0'],[],['2', '9', '11'],['8'],[],[],[],['10'],[],[]]
        if str(j) not in videoIDsToRemove[i]:
            Xij, Yij = dataToXY(i,j)
            X.append(Xij)
            Y.append(Yij)

for seed in [0]:#, 10, 27, 42, 72]:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

    print('X : ', len(X), 'Y : ', len(Y))
    print('X train : ', np.asarray(X_train).shape)
    print('X test : ', np.asarray(X_test).shape)
    print('Y train : ', np.asarray(Y_train).shape)
    print('Y test : ', np.asarray(Y_test).shape)

    print('Y train :')
    for label in classNames:
        print(label, Y_train.count(label))
    print('Y test :')
    for label in classNames:
        print(label, Y_test.count(label))

    scoring = {'accuracy' : make_scorer(accuracy_score), 
        'precision' : make_scorer(precision_score, average = None),
        'recall' : make_scorer(recall_score, average = None), 
        'f1_score' : make_scorer(f1_score, average = None)}

    dummy_clf = DummyClassifier(strategy="most_frequent", random_state=0).fit(X_train, Y_train)
    Y_pred = dummy_clf.predict(X_test)
    accuracy = dummy_clf.score(X_test, Y_test)
    print('Accuracy : ', accuracy)
    '''titles_options = [
        ("Confusion matrix", None),
        ("Normalized confusion matrix", "true"),
    ]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            dummy_clf,
            X_test,
            Y_test,
            cmap=plt.cm.Blues,
            normalize=normalize,
        )
        disp.ax_.set_title(title)
    plt.show()'''


    print('------ SVM ------')
    svcParameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'gamma': [0.0001, 0.001, 0.01, 0.1,'scale','auto'],
            'kernel':['rbf'],
            'class_weight':['balanced', None]}
    clfSVC = GridSearchCV(SVC(random_state=0), svcParameters, cv=10, n_jobs=8).fit(X_train, Y_train)
    '''svrParameters = {'estimator__C': np.arange(0.1,100.1,0.1),
            'estimator__gamma': [0.0001, 0.001, 0.01, 0.1,'scale','auto'],
            'estimator__kernel':['linear', 'poly', 'rbf', 'sigmoid']}
    clfSVR = GridSearchCV(MultiOutputRegressor(SVR()), svrParameters, cv=10, n_jobs=8).fit(X_train, Y_train)'''
    print(clfSVC.best_params_)
    print(clfSVC.best_score_)

    '''model = MultiOutputRegressor(SVR(C=4.1, gamma='scale', kernel='rbf'))
    model.fit(X_train, Y_train)
    results = permutation_importance(model, X_test, Y_test, scoring='neg_mean_squared_error')
    importance = pd.Series(results.importances_mean, featureNames).sort_values(ascending=True).plot(kind='barh', title='Feature Importances')
    plt.show()'''

    print('\nRandom Forest')
    clfRF = RandomForestClassifier(max_depth=None, max_features=None, random_state=0).fit(X_train, Y_train)
    print(clfRF.score(X_test, Y_test))
    pd.Series((clfRF.feature_importances_), index=featureNames).sort_values(ascending=True).plot(kind='barh', title='Feature Importances')
    '''result = permutation_importance(clfRF, X_test, Y_test, n_repeats=10, random_state=0, n_jobs=2)
    forest_importances = pd.Series(result.importances_mean, index=featureNames).sort_values(ascending=True).plot(kind='barh', title='Feature Importances',xerr=result.importances_std)
    '''
    plt.show()



    print('SVM')
    clfSVC = SVC(kernel="rbf", C=10, gamma=0.1, random_state=0, class_weight='balanced').fit(X_train, Y_train)
    Y_pred = clfSVC.predict(X_test)
    accuracy = clfSVC.score(X_test, Y_test)
    precision, recall, f1score, _ = precision_recall_fscore_support(Y_test, Y_pred, labels=classNames, average=None)
    print('Accuracy : ', accuracy)
    print('Precision : ', precision)
    print('Recall : ', recall)
    print('f-score : ', f1score)
    titles_options = [
        ("Confusion matrix", None),
        ("Normalized confusion matrix", "true"),
    ]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            clfSVC,
            X_test,
            Y_test,
            cmap=plt.cm.Blues,
            normalize=normalize,
        )
        disp.ax_.set_title(title)
    plt.show()


# linear
'''def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()
f_importances(clfSVC.coef_[0], featureNames)'''

# rbf
'''perm_importance = permutation_importance(clfSVC, X_test, Y_test)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(np.array(featureNames)[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.show()'''