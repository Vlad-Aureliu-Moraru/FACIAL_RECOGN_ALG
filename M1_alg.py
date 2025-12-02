import os
import numpy as np
from statistics import mode
from statistics import StatisticsError
import matplotlib.pyplot as plt
import cv2
import time 

#fixed data
data_path = 'orl/' 
h, w = 112, 92
num_pixels = h * w          
num_persons = 40


def displayImages(poza_test,knn_result,nn_result):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(poza_test, cmap='gray')
    plt.title("Input")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(knn_result, cmap='gray')
    plt.title("kNN Result")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(nn_result, cmap='gray')
    plt.title("NN Result")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def constructTrainingMatrix(num_persons = 40,num_img_per_person = 10,procentage_of_images_to_train = 8):
    coloane_A = []
    coloane_B = []
    labels_A = []  
    labels_B = []
    count = 0
    countB = 0
    num_total_train = num_persons * num_img_per_person  
    A = np.zeros((num_pixels, num_total_train), dtype=np.float32)
    B = np.zeros((num_pixels,num_total_train) , dtype = np.float32)
    for i in range(1, num_persons + 1):          
        person_path = os.path.join(data_path, f's{i}')
        
        for j in range(1, num_img_per_person + 1):  
            img_path = os.path.join(person_path, f'{j}.pgm')
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32)
            img_vector = img.reshape(num_pixels, 1)
            if j<=procentage_of_images_to_train:
                coloane_A.append( img_vector[:, 0])
                labels_A.append(i)
                count+=1
            else:
                coloane_B.append( img_vector[:, 0])
                labels_B.append(i)
                countB+=1

    A = np.column_stack(coloane_A)
    B = np.column_stack(coloane_B)
    labels_A = np.array(labels_A, dtype=int)
    labels_B = np.array(labels_B, dtype=int)
    return A,B,labels_A,labels_B


def NN(pozaTest,A,norma):
    start = time.time()
    z = np.zeros(A.shape[1])
    for i in range (0,A.shape[1]):
        if(norma==2):            
            z[i] = np.linalg.norm(pozaTest - A[:, i], 2) 
        elif(norma==1):
            z[i] = np.linalg.norm(pozaTest - A[:,i],1)
        elif(norma==3):
            z[i] = np.linalg.norm(pozaTest - A[:,i],np.inf)
        elif(norma==4):
            a = pozaTest
            b = A[:, i]
            cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
            z[i] = 1 - cos_sim
        else:
            return
    end = time.time()

    io = np.argmin(z)
    return io,end-start

def kNN(pozaTest,A,labels_A,norma,k=3):
    start = time.time()
    z = np.zeros(A[0].size)
    for i in range (0,A[0].size):
        if(norma==2):            
            z[i] = np.linalg.norm(pozaTest - A[:, i], 2) 
        elif(norma==1):
            z[i] = np.linalg.norm(pozaTest - A[:,i],1)
        elif(norma==3):
            z[i] = np.linalg.norm(pozaTest - A[:,i],np.inf)
        elif(norma==4):
            a = pozaTest
            b = A[:, i]
            cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
            z[i] = 1 - cos_sim
        else:
            return
    indici = np.argsort(z)
    indici_k = labels_A[indici[:k]]
    #clase_k = indici_k//8+1
    try:
        po = mode(indici_k)
    except StatisticsError:
        po = indici_k[0]
    end = time.time()
    
    return po,indici[0],end-start

