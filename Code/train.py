from keras.models import Model
from keras.layers import Input, Dense, Activation
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.utils import class_weight
from sklearn.metrics import hamming_loss
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc
from numpy import loadtxt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gc
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint
from Metrics import metrics
from keras.models import Sequential
from keras.callbacks import EarlyStopping

data_train=pd.read_csv("Reduced_MLDA_400_train-bp.csv", header=None) #read train file

data_test=pd.read_csv("Reduced_MLDA_400_test-bp.csv", header=None)  #read  test fil
nFeature = 351
cls=932 #(0-585, total 586 classes)

#data_train = data_train.drop(data_train.columns[725],axis=1) 
#data_test = data_test.drop(data_test.columns[725],axis=1) 


X_data_train=data_train.values
X_data_test=data_test.values
 
X_train=X_data_train[:,:nFeature]
Y_train=X_data_train[:,nFeature:]
X_test=X_data_test[:,:nFeature]
Y_test=X_data_test[:,nFeature:]
mmmm

'''
### scaled with either 0 or 1
mean = []
for a in range(nFeature):
    mean.append(X_train[:,a].mean())

for i in range(X_train.shape[0]):
    for j in range(nFeature):
        if X_train[i,j] < mean[j] :
            X_train[i,j] = 0
        else :
            X_train[i,j] = 1

for i in range(X_test.shape[0]):
    for j in range(nFeature):
        if X_test[i,j] < mean[j] :
            X_test[i,j] = 0
        else :
            X_test[i,j] = 1
'''
'''
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
'''
'''
X_train_p = X_train[Y_train[:,cls]==1]
X_train_n = X_train[Y_train[:,cls]==0]
#s = X_train_p.shape[0]
#X_train_n = X_train_n[np.random.randint(X_train_n.shape[0], size =s),:]

#X_train_n[:,500+cls] = 1
#X_train_p[:,500:] = 0
#X_train_p[:,505] = 1 
#X_train_p[:,500].sum()
#X_train_p[:,500:].sum()
X_test_p = X_test[Y_test[:,cls]==1]
X_test_n = X_test[Y_test[:,cls]==0]
#s = X_test_p.shape[0]
#X_test_n = X_test_n[np.random.randint(X_test_n.shape[0], size =s),:]

#X_test_p[:,500:] = 0
#----X_test_p[:,400+cls] = 1
#X_test_n[:,500:] = 0
#----X_test_n[:,400+cls] = 1
#X_test_p[:,500:].sum()
#X_test_n[:,500:].sum()
#X_test_p=preprocessing.normalize(X_test_p, axis=0, copy=True, return_norm=False) #normalize x test
#X_test_n=preprocessing.normalize(X_test_n, axis=0, copy=True, return_norm=False) #normalize x test
#X_train=preprocessing.normalize(X_train, axis=0, copy=True, return_norm=False)#normalize x train
'''
#scaler.transform(X_test_p)
#scaler.transform(X_test_n)
'''
#eachTestSampleLoss_n = []
#eachTestSampleLoss_p = []
eachTrainSampleLoss = []
eachTrainSampleLoss_p = []
eachTrainSampleLoss_n = []
label_train=[]

eachTestSampleLoss = []
label_test=[]
'''
f_path = '/home/acer/Desktop/hope/BP/Autoencoder/AEmodels/'
totalTrainLoss = []
totalTestLoss=[]
for j in range(cls):
    eachTrainSampleLoss = []
    eachTestSampleLoss = []
    #goTerm =  goTermsList[j] 
    filePath = f_path + "AEmodel" + str(j) + ".h5"
    autoencoder = load_model(filePath)
    print(j)
    
    for i in range(len(X_train)): 
        eachTrainSampleLoss.append(autoencoder.evaluate(X_train[i].reshape(1,nFeature),X_train[i].reshape(1,nFeature),verbose=0))
    totalTrainLoss.append(eachTrainSampleLoss)
    
    
    for k in range(len(X_test)): 
        eachTestSampleLoss.append(autoencoder.evaluate(X_test[k].reshape(1,nFeature),X_test[k].reshape(1,nFeature),verbose=0))
    totalTestLoss.append(eachTestSampleLoss)
    
    #del autoencoder
    #gc.collect()
#logger.info("transposing")    
totalTrainLoss = np.array(totalTrainLoss).T.tolist()
totalTestLoss = np.array(totalTestLoss).T.tolist()

#logger.info("exiting loop")


X_new_train = pd.DataFrame(totalTrainLoss, dtype=np.float32)
X_new_train = X_new_train *10000
X_new_train.to_csv("trainLoss.csv")
X_new_test = pd.DataFrame(totalTestLoss, dtype=np.float32)
X_new_test = X_new_test *10000
X_new_test.to_csv("testLoss.csv")



X_new_train = pd.read_csv("trainLoss.csv", header=None)
X_new_test = pd.read_csv("testLoss.csv", header=None)


X_new_test=preprocessing.normalize(X_new_test, axis=1, copy=True, return_norm=False) #normalize x test
X_new_train=preprocessing.normalize(X_new_train, axis=1, copy=True, return_norm=False)#normalize x train

scaler = MinMaxScaler()
scaler.fit(X_new_train)
X_new_train = scaler.transform(X_new_train)
X_new_test = scaler.transform(X_new_test)

filepath="model_sigmoid_hope_mse.h5"
checkpoint = ModelCheckpoint(filepath,monitor="val_loss" ,verbose=1, save_best_only=True, mode='min')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
callbacks_list = [checkpoint]

model = Sequential()
model.add(Dense(810, input_dim=cls, kernel_initializer="uniform"))#, activation='relu'))
#model.add(BatchNormalization())
#model.add(Activation("relu"))
#model.add(Dense(100))#, activation='relu'))
#model.add(BatchNormalization())
model.add(Activation("relu"))
#model.add(Dense(500, activation='relu'))
model.add(Dense(cls, activation='sigmoid', kernel_initializer="uniform"))

model.compile(optimizer="rmsprop",loss='mean_squared_error')


m = model.fit(X_new_train,Y_train,  class_weight='balance', epochs=500,batch_size=32, validation_split=0.3, callbacks=callbacks_list, shuffle=True)

plt.plot(m.history['loss'])
plt.plot(m.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train Loss', 'Validation Loss'], loc='upper right')
plt.show() 
model = load_model(filepath)

Y_predicted = model.predict(X_new_test)
threshold = 0.23
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
plt.savefig("PRgraph.png")
