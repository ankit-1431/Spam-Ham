# Libraries 

import string
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from email.parser import Parser
import re
import nltk
from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

nltk.download('stopwords')
ENGLISH_STOP_WORDS = set(stopwords.words('english'))

# Data Processing

def process_data(temp):
  temp=temp.lower() # Removing all Uppercases 
  temp=temp.translate(str.maketrans(dict.fromkeys(string.punctuation))) # Removing the puctutations
  temp=re.sub('\d+',"",temp)# removing numbers 
  # Removing Stop words
  s="" 
  for i in temp.split():
    if i not in ENGLISH_STOP_WORDS:
      s+=i
      s+=' '
  return s

dataset = pd.read_csv('spam_ham_dataset.csv')
dataset = pd.DataFrame(dataset)

for i in range(0,len(dataset)):
  x=str(dataset.iloc[i][2])
  y=process_data(x)
  dataset.loc[i,'text']=y

dataset.sample(frac=1) # random shuffle 
data=pd.DataFrame({'text':list(dataset['text']),'Spam/Ham':list(dataset['label_num'])})


# train-test split ( 40% + 60%)
X=data['text']
Y=data['Spam/Ham']
x_train,x_test,y_train,y_test=train_test_split(X,Y,stratify=Y,test_size=0.6)

# Graph
grp = data['Spam/Ham'].value_counts()
grp.plot(kind="bar")
plt.xticks(np.arange(2), ('Ham', 'Spam'),rotation=0)
plt.show()

# features extraction 
#tf ( term frequency) gives the recurrences of words and idf (inverse data frequency ) gives heaviness to uncommon words.
feature_extraction = TfidfVectorizer() # Gives High value to unique words and low value to common words.
X_train = feature_extraction.fit_transform(x_train) # fit_transform scale out features based on mean and variance of all data in train set 
X_test = feature_extraction.transform(x_test) # transform use the mean and variance of train data to scale out test to keep it un-biased

# Function to Print Results 
def print_result(accuracy,matrix):
  print("  Confusion Matrix : \n",matrix)
  print("  Accuracy : ", accuracy*100)
  print("  Precision : ",matrix[0][0]/(matrix[0][0]+matrix[1][0]) )
  print("  Recall : ",matrix[0][0]/(matrix[0][0]+matrix[0][1]) )
  print("")



# Logistic Regression 

print("Apply Logistic Regression Algorithm ")
model = LogisticRegression()
model.fit(X_train, y_train)
predict_train= model.predict(X_train)
accuracy_train = accuracy_score(y_train, predict_train)
mat_train=confusion_matrix(y_train, predict_train)

print("For Training Data :")
print_result(accuracy_train,mat_train)

predict_test= model.predict(X_test)
accuracy_test = accuracy_score(y_test, predict_test)
mat_test=confusion_matrix(y_test, predict_test)

print("For Test Data :")
print_result(accuracy_test,mat_test)





# Perceptron Learning Algorithm

print("Apply Perceptron Learning Algorithm ")
p= Perceptron()
p.fit(X_train,y_train)
predi_train=p.predict(X_train)
p_accuracy_train =p.score(X_train,y_train)
mat_train1=confusion_matrix(y_train, predi_train)

print("For Training Data :")
print_result(p_accuracy_train,mat_train1)

predi_test=p.predict(X_test)
p_accuracy_test =p.score(X_test,y_test)
mat_test1=confusion_matrix(y_test, predi_test)

print("For Test Data :")
print_result(p_accuracy_test,mat_test1)





# Single Layer Perceptron

print("Apply Single Layer Perceptron Algorithm ")
slp= MLPClassifier(hidden_layer_sizes=(), max_iter=500).fit(X_train, y_train) #hidden_layer_sizes=() means no hidden layer hence SLP 
slp_predi_train=slp.predict(X_train)
slp_accuracy_train =slp.score(X_train,y_train)
mat_train2=confusion_matrix(y_train,slp_predi_train)

print("For Training Data :")
print_result(slp_accuracy_train,mat_train2)

slp_predi_test=slp.predict(X_test)
slp_accuracy_test =slp.score(X_test,y_test)
mat_test2=confusion_matrix(y_test,slp_predi_test)

print("For Test Data :")
print_result(slp_accuracy_test,mat_test2)





# Multi Layer Perceptron

print("Apply Multi Layer Perceptron Algorithm ")
mlp= MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=500).fit(X_train, y_train) # MLP with 2 hidden layers each of 100 perceptrons 
mlp_predi_train=mlp.predict(X_train)
mlp_accuracy_train =mlp.score(X_train,y_train)
mat_train3=confusion_matrix(y_train,mlp_predi_train)

print("For Training Data :")
print_result(mlp_accuracy_train,mat_train3)

mlp_predi_test=mlp.predict(X_test)
mlp_accuracy_test =mlp.score(X_test,y_test)
mat_test3=confusion_matrix(y_test,mlp_predi_test)

print("For Test Data :")
print_result(mlp_accuracy_test,mat_test3)



