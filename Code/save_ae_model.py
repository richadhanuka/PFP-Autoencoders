from keras.layers import Input, Dense, Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import ModelCheckpoint
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


data_train=pd.read_csv("Reduced_MLDA_400_train-bp.csv", header=None) #read train file
#data_test=pd.read_csv("transformed-test-bp.csv")  #read  test fil

X_data_train=data_train.values
#X_data_test=data_test.values
nFeature=351
cls=932 #(0-585, total 586 classes)

X_train=X_data_train[:,:nFeature]
Y_train=X_data_train[:,nFeature:]
#X_test=X_data_test[:,:nFeature]
#Y_test=X_data_test[:,500:]

#from collections import Counter
#print('Resampled dataset shape %s' % Counter(Y_train))

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
'''
'''
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
'''

#X_test=preprocessing.normalize(X_test, axis=0, copy=True, return_norm=False) #normalize x test
#X_train=preprocessing.normalize(X_train, axis=0, copy=True, return_norm=False)#normalize x train
#pd.DataFrame(X_train).to_csv("normalised_train.csv")
#pd.DataFrame(X_test).to_csv("normalised_test.csv")

for i in range(552,cls): 
    # this is our input placeholder
    input_data = Input(shape=(nFeature,))
    # "encoded" is the encoded representation of the input
    
    encoded = Dense(300, activation='tanh')(input_data)
    #encoded = Dropout(0.2)(encoded)
    encoded = Dense(200, activation='tanh')(encoded)
    #encoded = Dropout(0.3)(encoded)
    #encoded = Dense(150, activation='tanh')(encoded)
    #encoded = Dense(500, activation='tanh')(encoded)
    #encoded = Dense(200, activation='tanh')(encoded)
    # "decoded" is the lossy reconstruction of the input
    #decoded = Dense(500, activation='tanh')(encoded)
    #decoded = Dense(1000, activation='tanh')(decoded)
    decoded = Dense(300, activation='tanh')(encoded)
    #decoded = Dropout(0.3)(decoded)
    #decoded = Dense(350, activation='tanh')(decoded)
    #decoded = Dropout(0.5)(decoded)
    decoded = Dense(nFeature, activation='tanh')(decoded)
    
    # this model maps an input to its reconstruction
    autoencoder = Model(input_data, decoded)
    
    # this model maps an input to its encoded representation
    #encoder = Model(input_data, encoded)
    filepath="AEmodel" + str(i) + ".h5"
    checkpoint = ModelCheckpoint(filepath,monitor="val_loss" ,verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    X_train_p = X_train[Y_train[:,i]==1]
    #X_train_p[:,500:] = 0
    #X_train_p[:,i+500] = 1 
    
    autoencoder.compile(optimizer='rmsprop', loss='mean_squared_error')
    autoencoder_train = autoencoder.fit(X_train_p, X_train_p,
                    epochs=100,
                    batch_size=8,
                    shuffle=True,
                    validation_split=0.1, callbacks=callbacks_list)
    
# summarize history for loss
    plt.plot(autoencoder_train.history['loss'])
    plt.plot(autoencoder_train.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train Loss', 'Validation Loss'], loc='upper right')
    plt.show() 
    del  autoencoder
    del  autoencoder_train
    #autoencoder.save("AEmodel600.h5")  # creates a HDF5 file 'my_model.h5'
