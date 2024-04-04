################################################
### October 7, 2019
### data preprocessing 
### convert raw file to dataset
### STEP 1
############################################## 
import pickle
import time
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
    #Importing the Dataset

start_time = time.time()
        
csv_file = r"D:\1_code\data\religion.csv" #sys.argv[1]
dataset = pd.read_csv(csv_file) 
    
filename = csv_file.split("\\")[-1]
filename = filename.split(".")[0]

    
X  = dataset['text']
y = dataset['religion']
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.5)
    ## save variable

train_name = filename+"_train.pickle"
test_name = filename+"_test.pickle"

with open(train_name, 'wb') as f:
    pickle.dump([X_train_raw, y_train], f)
with open(test_name, 'wb') as f:
    pickle.dump([X_test_raw, y_test], f)

# count instance 
print("# trian ",len(X_train_raw))
print("# test ",len(X_test_raw))

#count numbr of each class      
unique, counts = np.unique(y_train, return_counts=True)
print("# y_train",unique,counts)
unique, counts = np.unique(y_test, return_counts=True)
print("# y_test",unique,counts)
    
print("Create dataset: "+train_name+ " and "+test_name)
elapsed_time = time.time() - start_time
print("Total time: ",elapsed_time, "seconds")
