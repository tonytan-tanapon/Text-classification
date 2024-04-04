#################################################################
### October 7, 2019
### file name text classifciation modelling.py
### data export from test_create_new_dataset.py
### load data -> machine learning -> evaluation
### STEP 3
#################################################################

import pickle
import numpy as np 
import time 
def get_pickle(filename): 
    with open(filename, 'rb') as f:
        x, y = pickle.load(f)
    return x, y
def openfile(file_name): 
    file = open(file_name, "r") 
    data =  file.readlines()

    return data
def writefile(file_name,data):
    with open(file_name, "w") as f: 
        f.write(data) 
 
def display(title,value):
    if title =="":
        print(str(value))
        out.append(str(value)+"\n")
    else: 
        print(title+": "+str(value))
        out.append(title+": "+str(value)+"\n")
start_time = time.time()
out = []
###########################
#### load data ############
###########################   

#### name come from STEP 2 text classification vectorization
train_name = "religion_train_tfidf.pickle"
test_name = "religion_test_tfidf.pickle"
X_train,y_train  = get_pickle(train_name)
X_test,y_test  = get_pickle(test_name)

display("number of training instance",len(X_train.toarray()))
display("number of testing instance",len(X_test.toarray()))

###############################
##### Machine learning ########
###############################
#ref=> https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568


from sklearn.naive_bayes import MultinomialNB ##************** recommend 66.83%
from sklearn.neural_network import MLPClassifier # time comsume 71.73%
from sklearn.neighbors import KNeighborsClassifier # 60.75%
from sklearn.svm import SVC # 49%
from sklearn.tree import DecisionTreeClassifier # 62.70%
from sklearn.ensemble import RandomForestClassifier # 58.68%
from sklearn.ensemble import AdaBoostClassifier # 67.517%
from sklearn.linear_model import SGDClassifier #************** recommend 71.44%
from sklearn.linear_model import LogisticRegression #********* recommend 69.97%


clf = SGDClassifier().fit(X_train, y_train)
## testing 
predicted = clf.predict(X_test)
display("Classifier",clf.__init__)

#############################
##### evaluation ############
#############################
from sklearn import metrics

def evaluation(test,predicted):
    ## Show in metrics, another 
    accuracy = (np.mean(predicted == test) *100) 
#    print("\nAccuracy: %.4f"%accuracy," %")
    display("\nAccuracy", accuracy )
    display("",metrics.classification_report(test, predicted))
    display("",metrics.confusion_matrix(test, predicted))
    
   
evaluation(y_test,predicted)
##
## plot graph
#show_graph = 0
#if show_graph ==1:
#    plottext(X_train,y_train)
#    plottext(X_test,y_test)
##    

filename =  "result_"+clf.__class__.__name__+".txt"
writefile(filename,"".join(out))

elapsed_time = time.time() - start_time
print("Total time: ",elapsed_time, "seconds")