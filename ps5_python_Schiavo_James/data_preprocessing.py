from itertools import count
import numpy as np
import cv2 as cv 
import glob
from matplotlib import pyplot as plt 

from sklearn.model_selection import train_test_split

#2
X = []
X_train = []
X_test = []
for i in range(1,41):
    person_images = []
    images = [cv.imread(file) for file in glob.glob("./input/all/HW5_att_faces/s" + str(i) + "/*.pgm")] 
    for j in range(len(images)):
        person_images.append(images[j])
        X_train_img,X_test_img = train_test_split(person_images,test_size=.20)
    X.append(person_images)
    X_train.append(X_train_img)
    X_test.append(X_test_img)
print("X:")
print(len(X))
print(len(X[0]))
print("X_train:")
print(len(X_train))
print(len(X_train[0]))
print("X_test:")
print(len(X_test))
print(len(X_test[0]))
for i in range(len(X_train)):
    for j in range(len(X_train[0])):
        X_train[i][j] = cv.cvtColor(X_train[i][j],cv.COLOR_BGR2GRAY)
        path = "./input/train/" + str(i + 1) + "_" + str(j + 1) + ".pgm"
        print(path)
        cv.imwrite(path,X_train[i][j])
for i in range(len(X_test)):
    for j in range(len(X_test[0])):
        X_test[i][j] = cv.cvtColor(X_test[i][j],cv.COLOR_BGR2GRAY)
        path = "./input/test/" + str(i + 1) + "_" + str(j + 1) + ".pgm"
        cv.imwrite(path,X_test[i][j])

img_2 = cv.imread("./input/train/1_1.pgm")
cv.imwrite("./output/ps5-2-0.png",img_2)
print("person's id: " + str(1))




def PCA(X_train,eigenvalues=np.load("./output/eigenvalues.npy"),eigenvectors=np.load("./output/eigenvectors.npy")):
    T = np.zeros((10304,320))
    for i in range(len(X_train)):
        for j in range(len(X_train[0])):
            img_vec = np.transpose([np.ndarray.flatten((X_train[i][j]))])
            T = np.append(T,img_vec,axis=1)
    T = np.delete(T,list(range(0,320)),axis=1)
    print(T.shape)
    print(T)
    cv.imwrite("./output/ps5-1-a.png",T)  
    m = np.zeros((10304,1))
    for i in range(T.shape[0]):
        m[i] = np.mean(T[i][:])
    print("m: " + str(m))
    m_disp = m.reshape(112,92)
    cv.imwrite("./output/ps5-2-1-b.png",m_disp)
    for j in range(T.shape[1]):
        A = T[:][j] - m
    C = np.matmul(A,np.transpose(A))
    print("C shape: " + str(C.shape))
    C_norm = (C-np.min(C))/(np.max(C)-np.min(C))*255
    cv.imwrite("./output/ps5-2-1-c.png",C_norm)
    #print("EIGEN!!!!!")
    #eigenvalues,eigenvectors = np.linalg.eig(C)
    #np.save("./output/eigenvalues.npy",eigenvalues)
    #np.save("./output/eigenvectors.npy",eigenvectors)
    print("eigenvalues: " + str(eigenvalues.real))
    print("eigenvectors: " + str(eigenvectors.real))
    eigenvalues_sorted = np.flip(np.sort(eigenvalues.real))
    print("eigenvalues sorted: " + str(eigenvalues_sorted.real))
    N = 320
    count = 1
    v_k = []
    n_sum = np.sum(eigenvalues_sorted.real)
    for n in range(N):
        k_sum = np.sum(eigenvalues_sorted[0:n].real)
        v_k.append(k_sum/n_sum)
    print("v_k: " + str(v_k))
    plt.figure()
    k = np.arange(0,len(v_k),1)
    plt.plot(k,v_k)
    plt.savefig("./output/ps5-2-1-d.png")
    K = 100
    U = np.zeros((eigenvectors.shape[0],K))
    for i in range(K):
        ev_index = 0
        while(eigenvalues_sorted[i] != eigenvalues[ev_index].real):
            ev_index += 1
        U[:,i] = eigenvectors[:,ev_index]
    img_nine = np.zeros((112,92*9))
    #TODO: 
    #for i in range(9):
        #img_nine = np.reshape(U[:][i],(X_train[0][0].shape[0],X_train[0][0].shape))
    #img_nine_norm = (img_nine-np.min(img_nine))/(np.max(img_nine)-np.min(img_nine))*255
    #cv.imwrite("./output/ps5-2-1-e.png",img_nine_norm)
    print("U shape: " + str(U.shape))
    print("U: " + str(U))
    return U, m

def feature_extraction(X_train,X_test,U,m):
    W_training = X_train
    W_testing = X_test
    y_train = np.zeros((320,1))
    y_test = np.zeros((80,1))
    for i in range(len(X_train)):
        for j in range(len(X_train[0])):
            I = np.transpose([np.ndarray.flatten((X_train[i][j]))])
            w = np.matmul(np.transpose(U),(I - U))
            print("w: " + str(w.shape))
            W_training[i][j] = w
            y_train[i*j] = i + 1
    print("y_train: " + str(y_train))
    for i in range(len(X_test)):
        for j in range(len(X_test[0])):
            I = np.transpose([np.ndarray.flatten((X_test[i][j]))])
            w = np.matmul(np.transpose(U),(I - U))
            W_testing[i][j] = w
            y_test[i*j] = i + 1
    print("y_test: " + str(y_test))
    print("W_training SHAPE!!!!!")
    print(len(W_training[0][0].shape))
    return W_training, W_testing, y_train, y_test



#2.1
U,m = PCA(X_train)     
     
#2.2
W_training, W_testing, y_train, y_test = feature_extraction(X_train,X_test,U,m)

#2.3.a
from sklearn.neighbors import KNeighborsClassifier
K = [1,3,5,7,9,11]
from sklearn import metrics
for i in range(len(K)):
    knn = KNeighborsClassifier(K[i])
    knn.fit(W_training,y_train)
    y_pred = knn.predict(W_testing)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    print("K: " + str(K[i]) + " ACCURACY: " + str(accuracy))

#2.3.b
from sklearn import svm
svm_classifier1 = svm.SVC(kernel='linear',decision_function_shape='ovo')
svm_classifier1.fit(W_training,y_train)
y_pred1 = svm_classifier1.predict(W_testing)
svm_classifier2 = svm.SVC(kernel='poly',decision_function_shape='ovo')
svm_classifier2.fit(W_training,y_train)
y_pred2 = svm_classifier2.predict(W_testing)
svm_classifier3 = svm.SVC(kernel='rbf',decision_function_shape='ovo')
svm_classifier3.fit(W_training,y_train)
y_pred3 = svm_classifier3.predict(W_testing)
svm_classifier4 = svm.SVC(kernel='linear',decision_function_shape='ovr')
svm_classifier4.fit(W_training,y_train)
y_pred4 = svm_classifier4.predict(W_testing)
svm_classifier5 = svm.SVC(kernel='poly',decision_function_shape='ovr')
svm_classifier5.fit(W_training,y_train)
y_pred5 = svm_classifier5.predict(W_testing)
svm_classifier6 = svm.SVC(kernel='rbf',decision_function_shape='ovr')
svm_classifier6.fit(W_training,y_train)
y_pred6 = svm_classifier6.predict(W_testing)