#################################################################
### file name text classification vectorization.py
### October 7, 2019
### data export from test_create_new_dataset.py
### load data -> vetorication 
### STEP 2
#################################################################
#text cleaning #https://stackabuse.com/text-classification-with-python-and-scikit-learn/
import pickle
import time
from sklearn import preprocessing
import numpy as np 


def get_pickle(filename): 
    with open(filename, 'rb') as f:
        x, y = pickle.load(f)
    return x, y

def get_LabelEncoder(y):
    le = preprocessing.LabelEncoder()
    new_y = le.fit_transform(y)
    #print(le.classes_)
    return new_y,le.classes_

start_time = time.time()
###########################
#### load data ############
###########################

### file name fron STEP1 text classification create dataset
train_name = "religion_train.pickle"
test_name = "religion_test.pickle"

X_train_raw,y_train_raw  = get_pickle(train_name)
X_test_raw,y_test_raw  = get_pickle(test_name)

list_lable = ['atheist', 'christian' ]
## get specific lable if there are more than 2 class
X_train_raw = [X_train_raw.iloc[i] for i,label in enumerate(y_train_raw)  if label in list_lable]
y_train_raw = [label for i,label in enumerate(y_train_raw)  if label in list_lable ]
X_test_raw = [X_test_raw.iloc[i] for i,label in enumerate(y_test_raw)  if label in list_lable]
y_test_raw = [label for i,label in enumerate(y_test_raw)  if label in list_lable]

y_train, label = get_LabelEncoder(y_train_raw)
y_test, label = get_LabelEncoder(y_test_raw)

##############################
####### vectorization ########
##############################
from sklearn.feature_extraction.text import TfidfVectorizer 
tfidf_vectorizer=TfidfVectorizer(use_idf=True,stop_words="english")
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(X_train_raw)

## vector for traning (tfidf)
X_train = tfidf_vectorizer_vectors 
print("# train word",X_train.shape)
      
## vector for test
## use only transform method
tfidf_vectorizer_vectors=tfidf_vectorizer.transform(X_test_raw)
X_test = tfidf_vectorizer_vectors
print("# test word",X_test.shape)

#count numbr of each class      
unique, counts = np.unique(y_train, return_counts=True)
print("# y_train",unique,counts)
unique, counts = np.unique(y_test, return_counts=True)
print("# y_test",unique,counts)
      
train_name = train_name.split(".")
train_name_new = train_name[0]+"_tfidf."+train_name[1]
test_name = test_name.split(".")
test_name_new = test_name[0]+"_tfidf."+test_name[1]

with open(train_name_new, 'wb') as f:
    pickle.dump([X_train, y_train], f)
with open(test_name_new, 'wb') as f:
    pickle.dump([X_test, y_test], f)

    
print("Create vectorization dataset: "+train_name_new+ " and "+test_name_new)

elapsed_time = time.time() - start_time
print("Total time: ",elapsed_time, "seconds")