import algorithm_related as alg
import fileResultsWriter as frw
import time 
h, w = 112, 92


A,B,labels_A,labels_B = alg.constructTrainingMatrix(40,10)

nn_count_correct = 0
knn_count_correct = 0

nn_result_array = []
knn_result_array = []

knn_timetaken_array = []
nn_timetaken_array = []

person_num = B.shape[1]
start = time.time()
for i in range (int(person_num)):
    vectorized_img = B[:,i]
    poza_test = vectorized_img.reshape(h, w)
    label,io,knnTime = alg.kNN(vectorized_img, A,labels_A, 2, 3)  
    knn_result = A[:, io].reshape(h, w)
    knn_timetaken_array.append(knnTime)
    
    if label == labels_B[i]:
        knn_count_correct+=1
    i1,nnTime =alg.NN(vectorized_img, A, 2)       
    nn_result = A[:, i1].reshape(h, w)
    nn_timetaken_array.append(nnTime)
    if labels_A[i1] == labels_B[i]:
        nn_count_correct+=1
    
end = time.time()
timpul_de_executie = end-start
nn_rate = nn_count_correct / B.shape[1]
knn_rate = knn_count_correct / B.shape[1]

print("┌ STATISTICA")
print(f"├ Timpul de executare al algoritmului pentru {person_num:.0f} persoane  : {timpul_de_executie:.4f}s")
print(f"├ Timpul Aproximat De Executie Pentru Procesarea Unei Img : {timpul_de_executie/person_num:.2f}s")
print(f"├  Matricea De Antrenare Contine\t : {400-person_num:.0f} imagini ")
print(f"├  Matricea De Test Contine\t\t : {person_num} imagini ")
print(f"├ Rata de Recunoastere Pentru Acest Run (NN) : {nn_rate*100:.2f}%")
print(f"└ Rata de Recunoastere Pentru Acest Run (kNN): {knn_rate*100:.2f}%")
frw.plot_execution_times(nn_timetaken_array,knn_timetaken_array)


#alg.displayImages(poza_test,knn_result,nn_result)


