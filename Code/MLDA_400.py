# -*- coding: utf-8 -*-
"""
Ashish Ranjan
"""
import pandas as pd
import numpy as np
from collections import Counter
import math
import logging 
import gc

from NFV import NFV400
#Create and configure logger 
logging.basicConfig(filename="Logs_MLDA.txt", 
                    format='%(asctime)s %(message)s', 
                    filemode='w') 
  
#Creating an object 
logger=logging.getLogger() 
  
#Setting the threshold of logger to DEBUG 
logger.setLevel(logging.DEBUG) 
'''
def nGram(seq, chunk_size):
    dict2 = list()
    for i in range(len(seq) - chunk_size + 1):
        dict2.append(seq[i:i + chunk_size])  
    return(dict2)

def dictionary(dataset,chunk_size):
    logger.info('Creating Dictionary:')
    dict = {}
    value = 0
    for row in dataset:
        value = value + 1
        for i in range(len(row) - chunk_size + 1):
            key = row[i:i + chunk_size]
            if key in dict:
                item = dict[key]
                if value not in item:
                    dict.setdefault(key, [])
                    dict[key].append(value)
            else:
                dict.setdefault(key, [])
                dict[key].append(value)
    return(dict)
    
def tf_vector(row,dict1):
    for key in dict1:
        dict1[key] = 0
    count = Counter(row)
    for key in dict1.keys():
        value = float(count[key])/float(len(row))
        dict1[key] = value*1000
    tf = list(dict1.values())
    return(tf)

def idf_vector(dict1,dict_index,N):
    for key in dict1:
        dict1[key] = 0
    for key in dict_index.keys():
        d_f = len(dict_index[key])
        var = float(N) / float(d_f)
        value = math.log10(var)
        dict1[key] = value
        del d_f,var,value
        gc.collect()
    idf = list(dict1.values())
    return(idf)

def tf_idf(dict1, n_gram, idf_value):
    tf_value = tf_vector(n_gram,dict1)
    tf_idf_value = [(a*b) for a, b in zip(tf_value, idf_value)]
    return(dict1.keys(), tf_idf_value)

#dataframe = pd.read_csv("model_data.txt", header = None)
chunk_size = 3
'''
f_dim = 400
dataframe = pd.read_csv("train-bp.csv")

'''
ls =[]
for i in range(1,dataframe.shape[1]):
    s = dataframe.iloc[:,i].sum()
    if s==231:
        ls.append(i)
        print(str(i) + "--" + str(s))

dataframe = dataframe.drop(dataframe.columns[408],axis=1)
dataframe = dataframe.drop(dataframe.columns[408],axis=1)
dataframe = dataframe.drop(dataframe.columns[408],axis=1)
'''

dataset = dataframe.values
seq_dataset = NFV400(dataset[:,0])
input_len = len(dataset[0])
#seq_dataset = dataset[:,0]
Y = dataset[:,1:input_len]
del dataframe, dataset
gc.collect()
logger.info('Original Dataset Size : %s' %len(seq_dataset))
N = len(seq_dataset)

'''
"""CREATE DICTIONARY"""
logger.info('*************************************************************')
dict_index = dictionary(seq_dataset,chunk_size)
logger.info('Dictionary Size: %s' %len(dict_index.keys()))
f_dim = len(dict_index)  # Feature Dimension

dout = {k:v for k,v in dict_index.items()}
for key in dout:
    dout[key] = 0
    
idf_value = idf_vector(dout, dict_index,N)
'''

"""CLASS PARTITION"""
cls_partiton = [ ]
for cols in range(Y.shape[1]):
    c_part = [ ]
    for i, col_val in enumerate(Y[:,cols]):
        if col_val == 1:
            c_part.append(i)
    cls_partiton.append(c_part)
del cols,i,col_val
gc.collect()

"""step 1: Calculating class mean and global mean."""
def class_mean():
    cls_mean = [ ]
    for cols in range(Y.shape[1]):
        data = [ ]
        for i in cls_partiton[cols]:
            logger.info('CLASS: %s, ROW: %s' %(cols,i))
            #n_gram = nGram(seq_dataset[i], chunk_size)
            #key_term, X = tf_idf(dout, n_gram, idf_value)
            X = seq_dataset[i]
            data.append(X)
        cls_mean.append(list(np.mean(np.array(data),axis=0)))
    return np.array(cls_mean)

def global_mean():
    cls_sum = [ ]
    count = 0
    for cols in range(Y.shape[1]):
        data = [ ]
        for i in cls_partiton[cols]:
            logger.info('CLASS: %s, ROW: %s' %(cols,i))
            #n_gram = nGram(seq_dataset[i], chunk_size)
            #key_term, X = tf_idf(dout, n_gram, idf_value)
            X = seq_dataset[i]
            data.append(X)
            count = count + 1
        cls_sum.append(list(np.sum(np.array(data),axis=0)))
    tot_sum = np.sum(np.array(cls_sum), axis=0)
    return(tot_sum/count)

c_mean = class_mean()
g_mean = global_mean()

"""step 2: Calculating scatter matrices."""
def scatter_w():
    #f_dim = X.shape[1]
    sca_w = np.zeros((f_dim,f_dim))
    for cols in range(Y.shape[1]):
        logger.info('Class: %s' %cols)
        class_sc_mat = np.zeros((f_dim,f_dim)) 
        for i in cls_partiton[cols]:
            #n_gram = nGram(seq_dataset[i], chunk_size)
            #key_term, X = tf_idf(dout, n_gram, idf_value)
            X = seq_dataset[i]
            pro = np.array(X).reshape(f_dim,1) - c_mean[cols].reshape(c_mean.shape[1],1)
            class_sc_mat += (pro).dot((pro).T)
        sca_w += class_sc_mat
    return(sca_w)
    
def scatter_b():
    sca_b = np.zeros((f_dim,f_dim))
    for cols in range(Y.shape[1]):
        logger.info('Class: %s' %cols)
        count= 0
        for i in cls_partiton[cols]:
            count = count + 1
        pro = c_mean[cols].reshape(c_mean.shape[1],1) - g_mean.reshape(g_mean.shape[0],1)
        sca_b += count * (pro).dot((pro).T)
    return(sca_b)

sca_w_mat = scatter_w()
sca_b_mat = scatter_b()

"""step 2: Calculating eigen vector and eigen value."""
#eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(sca_w_mat).dot(sca_b_mat))
eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(sca_w_mat).dot(sca_b_mat))
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
logger.info('Variance explained:')
eigv_sum = sum(eig_vals)
for i,j in enumerate(eig_pairs):
    if(i<400):
        logger.info('eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))

# Choosing k eigenvectors with the largest eigenvalues
W = np.hstack((eig_pairs[0][1].reshape(f_dim,1), 
               eig_pairs[1][1].reshape(f_dim,1)))
logger.info('Matrix W:\n', W.real)
T = [ ]
for i in range(351):
    t = list(np.hstack(eig_pairs[i][1].reshape(f_dim,1)).real)
    T.append(t)
T = np.array(T)
trans_mat = pd.DataFrame((T).T)
trans_mat.to_csv('transform_mat_mlda_400.csv', header = None, index = None)

logger.info("Storing Transformed Train Dataset.....")
import os
from csv import writer
dir = os.path.join('output')
try:
    os.makedirs(dir)
except OSError:
    pass
myFile = open(os.path.join(dir, "Reduced_MLDA_400_train-bp.csv"), 'w', newline = '')
with myFile:
    for t_tag, seq in enumerate(seq_dataset):
        #n_gram = nGram(seq, chunk_size)
        #key_term, X = tf_idf(dout, n_gram, idf_value)
        X = seq
        Z = list((T).dot(np.array(X)))
        for item in Y[t_tag]:
            Z.append(item)
        csv_writer = writer(myFile)
        csv_writer.writerow(Z)
myFile.close()

logger.info("Storing Transformed Test Dataset.....")
dataframe = pd.read_csv("test-bp.csv")
'''
dataframe = dataframe.drop(dataframe.columns[408],axis=1)
dataframe = dataframe.drop(dataframe.columns[408],axis=1)
dataframe = dataframe.drop(dataframe.columns[408],axis=1)
'''

dataset = dataframe.values
test_seq_dataset = NFV400(dataset[:,0])
input_len = len(dataset[0])
#test_seq_dataset = dataset[:,0]
test_Y = dataset[:,1:input_len]
myFile = open(os.path.join(dir, "Reduced_MLDA_400_test-bp.csv"), 'w', newline = '')
with myFile:
    for tag, seq in enumerate(test_seq_dataset):
        #n_gram = nGram(seq, chunk_size)
        #key_term, X = tf_idf(dout, n_gram, idf_value)
        X = seq
        Z = list((T).dot(np.array(X)))
        for item in test_Y[tag]:
            Z.append(item)
        csv_writer = writer(myFile)
        csv_writer.writerow(Z)
myFile.close()
