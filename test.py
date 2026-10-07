from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import layers
import pickle
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
from keras.models import Model
from sklearn.neural_network import MLPClassifier


dataset = pd.read_csv("Dataset/swat_dataset.csv")
dataset.fillna(0, inplace = True)
dataset = dataset.values
X = dataset[:,0:dataset.shape[1]-1]
Y = dataset[:,dataset.shape[1]-1]

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
with open('model/minmax.txt', 'wb') as file:
    pickle.dump(scaler, file)
file.close()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
'''
encoding_dim = 256 # encoding dimesnion is 32 which means each image will be filtered 32 times to get important features from images
input_size = keras.Input(shape=(X.shape[1],)) #we are taking input size
encoded = layers.Dense(encoding_dim, activation='relu')(input_size) #creating dense layer to start filtering dataset with given 32 filter dimension
decoded = layers.Dense(y_train.shape[1], activation='softmax')(encoded) #creating another layer with input size as 784 for encoding
autoencoder = keras.Model(input_size, decoded) #creating decoded layer to get prediction result
encoder = keras.Model(input_size, encoded)#creating encoder object with encoded and input images
encoded_input = keras.Input(shape=(encoding_dim,))#creating another layer for same input dimension
decoder_layer = autoencoder.layers[-1] #holding last layer
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))#merging last layer with encoded input layer
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])#compiling model
print(autoencoder.summary())#printing model summary
hist = autoencoder.fit(X, Y, epochs=300, batch_size=16, shuffle=True, validation_data=(X_test, y_test))#now start generating model with given Xtrain as input 
autoencoder.save_weights('model/encoder_model_weights.h5')#above line for creating model will take 100 iterations            
model_json = autoencoder.to_json() #saving model
with open("model/encoder_model.json", "w") as json_file:
    json_file.write(model_json)
json_file.close    
f = open('model/encoder_history.pckl', 'wb')
pickle.dump(hist.history, f)
f.close()
encoder_acc = hist.history
acc = encoder_acc['accuracy']
'''

with open('model/encoder_model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    autoencoder = model_from_json(loaded_model_json)
json_file.close()
autoencoder.load_weights("model/encoder_model_weights.h5")
autoencoder._make_predict_function()
print(autoencoder.summary())
        
predict = autoencoder.predict(X_test)
predict = np.argmax(predict, axis=1)
testY = np.argmax(y_test, axis=1)
f = f1_score(testY, predict,average='macro') * 100
a = accuracy_score(testY,predict)*100    
print(f)
print(a)

encoder_model = Model(autoencoder.inputs, autoencoder.layers[-1].output)#creating autoencoder model
vector = encoder_model.predict(X)  #extracting features using autoencoder
print(vector.shape)
pca = PCA(n_components = 7)
vector = pca.fit_transform(vector)
print(vector)
print(vector.shape)
Y1 = np.argmax(Y, axis=1)

X_train, X_test, y_train, y_test = train_test_split(vector, Y1, test_size=0.2)

dt = DecisionTreeClassifier()
dt.fit(vector, Y1)
predict = dt.predict(X_test)
f = f1_score(y_test, predict,average='macro') * 100
a = accuracy_score(y_test,predict)*100    
print(f)
print(a)


attack_type = []
for i in range(len(vector)):
    temp = []
    temp.append(vector[i])
    attack = dt.predict(np.asarray(temp))
    attack_type.append(attack[0])
attack_type = np.asarray(attack_type)
print(attack_type)
print(np.unique(attack_type))


X_train, X_test, y_train, y_test = train_test_split(vector, attack_type, test_size=0.2)

dnn = MLPClassifier()
dnn.fit(vector, attack_type)
predict = dnn.predict(X_test)
f = f1_score(y_test, predict,average='macro') * 100
a = accuracy_score(y_test,predict)*100    
print(f)
print(a)



