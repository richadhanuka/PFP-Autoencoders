from keras.models import Model
from sklearn.metrics import hamming_loss
from keras.layers import Input, Dense, Activation
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc
from numpy import loadtxt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gc
from Metrics import metrics
from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler
from keras.layers.normalization import BatchNormalization

mmmm

data_test=pd.read_csv("Reduced_MLDA_400_test-bp.csv", header=None)  #read  test fil
nFeature = 351
cls=932  
X_data_test=data_test.values

Y_test=X_data_test[:,nFeature:]

X_new_train = pd.read_csv("trainLoss.csv", header=None)
X_new_test = pd.read_csv("testLoss.csv", header=None)

X_new_test=preprocessing.normalize(X_new_test, axis=1, copy=True, return_norm=False) #normalize x test
X_new_train=preprocessing.normalize(X_new_train, axis=1, copy=True, return_norm=False)#normalize x train

scaler = MinMaxScaler()
scaler.fit(X_new_train)
X_new_train = scaler.transform(X_new_train)
X_new_test = scaler.transform(X_new_test)

#X_new_train.to_csv("train_input.csv",index=None)
#X_new_test.to_csv("test_input.csv", index=None)

filepath="model_sigmoid_hope_best.h5"
model = load_model(filepath)

Y_predicted = model.predict(X_new_test)
threshold = 0.21
Y_predicted[Y_predicted >= threshold] = 1
Y_predicted[Y_predicted < threshold] = 0
avgPrecision, avgRecall, avgF1Score, F1Score, hammingLoss = metrics(Y_predicted, Y_test)

print("Average Precision : " + str(avgPrecision))

print("Average Recall : "  + str (avgRecall))

print("Average F1-Score : " + str(avgF1Score))


print("F1-score : " + str(F1Score))

print("Hamming Loss : " + str(hammingLoss))

thresholds = [0.1, 0.15, 0.2, 0.25,
      0.3, 0.35, 0.4, 0.45,
      0.5, 0.55, 0.6, 0.65]
plist,rlist,flist = [],[],[]
for t in thresholds:
    Y_predicted = model.predict(X_new_test)
    Y_predicted[Y_predicted>= t] = 1
    Y_predicted[Y_predicted< t] = 0
    avgPrecision, avgRecall, avgF1Score, F1Score, hammingLoss = metrics(Y_predicted, Y_test)

    plist.append(avgPrecision)
    rlist.append(avgRecall)
    flist.append(F1Score)


import matplotlib.pyplot as plt
plt.plot(plist)
plt.plot(rlist)
plt.plot(flist)

plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11], thresholds)
plt.grid()
plt.ylabel('percentage')
plt.xlabel('threshold')
plt.legend(['precision','recall','f1-score'], loc = 'upper right')
#plt.show()
plt.savefig("PRgrapha.png")

#aupr
model = load_model(filepath)
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
# predict probabilities
lr_probs = model.predict_proba(X_new_test)
# keep probabilities for the positive outcome only
#lr_probs = lr_probs[:, 1]
        
from sklearn.metrics import average_precision_score

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(932):
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                        lr_probs[:, i])
    average_precision[i] = average_precision_score(Y_test[:, i], lr_probs[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
    lr_probs.ravel())
average_precision["micro"] = average_precision_score(Y_test, lr_probs,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

print(classification_report(Y_test[:,791] ,Y_predicted[:,791] ))
