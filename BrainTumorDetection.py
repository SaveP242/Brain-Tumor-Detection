import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

path= os.listdir('brain_tumor_dataset/Training/')
classes= {'no':0 , 'yes':1 }

X=[]
Y=[]
for c in classes:
    pth= 'brain_tumor_dataset/Training/'+c
    for j in os.listdir(pth):
        img= cv2.imread(pth+'/'+j, 0)
        img= cv2.resize(img, (200,200))
        X.append(img)
        Y.append(classes[c])

X= np.array(X)
Y= np.array(Y)

#Preparing Data

X_updated= X.reshape(len(X), -1)
print(X_updated.shape)

#Split Data

X_train, X_test, Y_train, Y_test = train_test_split(X_updated, Y, random_state=10, test_size=.20)

#Feature Scaling

X_train = X_train/255
X_test = X_test/255

#Feature Selection

#from sklearn.decomposition import PCA

#pca= PCA(0.98)
pca_train= X_train
pca_test= X_test

#Train Model

from sklearn.svm import SVC

sv= SVC()
sv.fit(pca_train, Y_train)

pred= sv.predict(pca_test)
np.where(Y_test != pred)


#Test Model

dec= {0:'no', 1:'yes'}
plt.figure(figsize=(12,8))
cl=1
test_set= []
test_pth= os.listdir('brain_tumor_dataset/Testing/')
for i in os.listdir('brain_tumor_dataset/Testing/no/')[:9]:
    plt.subplot(3, 3, cl)
    img= cv2.imread('brain_tumor_dataset/Testing/no/'+i, 0)
    img1 = cv2.resize(img, (200, 200))
    img1= img1.reshape(1, -1) / 255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    cl += 1
plt.show()
plt.figure(figsize=(12,8))

cl1= 1

for i in os.listdir('brain_tumor_dataset/Testing/yes/')[:9]:
    plt.subplot(3, 3, cl1)
    img= cv2.imread('brain_tumor_dataset/Testing/yes/'+i, 0)
    img2 = cv2.resize(img, (200,200))
    img2= img2.reshape(1,-1)/255
    p= sv.predict(img2)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap= 'gray')
    plt.axis('off')
    cl1 += 1
plt.show()
plt.figure(figsize=(12,8))