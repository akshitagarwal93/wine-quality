from keras.models import Sequential
from keras.layers import Dense,Dropout
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from keras import metrics
np.random.seed(42)


#training set
dataset=np.genfromtxt('winequality-red.csv',delimiter=';', skip_header=1)


#print(dataset)
#print(dataset.shape)
X = dataset[:,0:11]
Y = dataset[:,11]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)

model=Sequential()
model.add(Dense(11, input_dim=11, activation='sigmoid'))

model.add(Dense(11, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(11, activation='softmax'))
#print(model.summary())
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=[metrics.sparse_categorical_accuracy])
model.fit(X_train, Y_train, epochs=100, batch_size=64, verbose=2, validation_data=(X_test,Y_test))




#Validation Set
dataset=np.genfromtxt('winequality-white.csv', delimiter=';', skip_header=1)
X = dataset[:,0:11]
Y = dataset[:,11]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)
s,a=model.evaluate(X_train, Y_train, batch_size=16, verbose = 2)      
print("score - %.2f" % (s))

print("accuracy - %.2f" % (a))

prediction=model.predict(X, batch_size=16) #Prediction Matrix








#print(prediction[1:4,].shape)

i=122
j=0
l=[]

print (prediction[0:10,])
"""
print(prediction[i,j])
print('\ndebug')
"""
#print(prediction[1:10,])

#for i in prediction[i]:
val=np.argmax(prediction[i])
print(val)
#i+=1
