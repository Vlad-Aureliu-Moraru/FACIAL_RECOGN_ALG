from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from M1_alg import *
import numpy as np
import time
import M2_EigenFaces as m2eig
import M2_Lanczos as m2lan
from collections import defaultdict
from tkinter import ttk

def run_eigenfaces(procentage, K,norma_val=1,knn_k_val=1,knn=False):

    if procentage == '60%':
        proc_value = 6
    elif procentage == '70%':
        proc_value = 7
    elif procentage == '80%':
        proc_value = 8
    else:
        raise ValueError("Invalid percentage")

    A, B, labels_A, labels_B = constructTrainingMatrix(
        procentage_of_images_to_train=proc_value
    )

    time_extract, mean_face, eigenfaces_matrix, training_projections = m2eig.eigenfaces(A, K)
    
    training_projections = training_projections.T

    correct = 0
    total = B.shape[1]
    times = []

    for i in range(total):
        b = B[:, i]
        b_centered = b - mean_face

        w_test = eigenfaces_matrix.T @ b_centered
        
        if knn is False:
            idx, t = NN(w_test, training_projections, norma=norma_val)
            predicted_label = labels_A[idx]
            true_label = labels_B[i]
            if predicted_label == true_label:
                correct += 1
            times.append(t)
        else:
            label, io, t=kNN(w_test,training_projections, labels_A,norma_val,knn_k_val)
            if label == labels_B[i]:
                correct += 1
            times.append(t)



    accuracy = correct / total
    avg_time = np.mean(times)

    print("Total test images:", total)
    print("Correct predictions:", correct)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Average recognition time: {avg_time:.6f} sec")
    print(f"norm {norma_val}")

    return A.shape[1],B.shape[1],accuracy, avg_time,time_extract
def run_eigenfaces_class_rep(procentage, K,norma_val=1,knn_k_val=1,knn=False):

    if procentage == '60%':
        proc_value = 6
    elif procentage == '70%':
        proc_value = 7
    elif procentage == '80%':
        proc_value = 8
    else:
        raise ValueError("Invalid percentage")

    A, B, labels_A, labels_B = constructTrainingMatrix(
        procentage_of_images_to_train=proc_value
    )

    time_extract, mean_face, eigenfaces_matrix, training_projections, rep_matrix, class_list = \
        m2eig.eigenfaces_class_representatives(A, labels_A, K)

    num_tests = B.shape[1]
    correct = 0
    total_classification_time = 0.0

    for i in range(num_tests):

        b = B[:, i]
        b_centered = b - mean_face
        w_test = eigenfaces_matrix.T @ b_centered

        if knn is False:
            idx, t = NN(w_test, training_projections, norma=norma_val)
            predicted_label = labels_A[idx]
            true_label = labels_B[i]
            if predicted_label == true_label:
                correct += 1
            total_classification_time+=t
        else:
            label, io, t=kNN(w_test,training_projections, labels_A,norma_val,knn_k_val)
            if label == labels_B[i]:
                correct += 1
            total_classification_time+=t

    accuracy = correct / num_tests
    avg_classification_time = total_classification_time/num_tests
    print("Correct predictions:",correct)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Average recognition time: {avg_classification_time:.6f} sec")

    return A.shape[1], B.shape[1], accuracy, avg_classification_time, time_extract
def run_lanczos(procentage, K,norma_val=1,knn_k_val=1,knn=False):

    if procentage == '60%':
        proc_value = 6
    elif procentage == '70%':
        proc_value = 7
    elif procentage == '80%':
        proc_value = 8
    else:
        raise ValueError("Invalid percentage")

    A, B, labels_A, labels_B = constructTrainingMatrix(
        procentage_of_images_to_train=proc_value
    )

    t0 = time.time()
    eigvals, eigenfaces = m2lan.lanczos(A, K)  
    preprocessing_time = time.time() - t0

    mean_face = np.mean(A, axis=1, keepdims=True)      
    A_centered = A - mean_face
    B_centered = B - mean_face

    training_projections = eigenfaces.T @ A_centered   

    num_tests = B.shape[1]
    correct = 0
    total_classification_time = 0.0

    for i in range(num_tests):
        b = B_centered[:, i]             
        w_test = eigenfaces.T @ b         

        if knn is False:
            idx, t = NN(w_test, training_projections, norma=norma_val)
            predicted_label = labels_A[idx]
            true_label = labels_B[i]
            if predicted_label == true_label:
                correct += 1
            total_classification_time += t
        else:
            label, io, t=kNN(w_test,training_projections, labels_A,norma_val,knn_k_val)
            if label == labels_B[i]:
                correct += 1
            total_classification_time += t

    accuracy = correct / num_tests
    avg_recognition_time = total_classification_time / num_tests

    print("Total test images:", num_tests)
    print("Correct predictions:", correct)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Average recognition time: {avg_recognition_time:.6f} sec")
    print(f"Preprocessing time: {preprocessing_time:.6f} sec")

    return A.shape[1], num_tests, accuracy, avg_recognition_time, preprocessing_time
def run_nn(procentage,norma_val):
    if procentage == '60%':
        proc_value = 6
    elif procentage == '70%':
        proc_value = 7
    elif procentage == '80%':
        proc_value = 8
    else:
        raise ValueError("Invalid percentage")

    A, B, labels_A, labels_B = constructTrainingMatrix(
        procentage_of_images_to_train=proc_value
    )
    nn_count_correct =0 
    nn_timetaken = []  
    person_num = B.shape[1]
    start = time.time()

    for i in range(person_num):
        vectorized_img = B[:, i]
        poza_test = vectorized_img.reshape(h, w)
        idx, nnTime = NN(vectorized_img, A, norma_val)
        nn_timetaken.append(nnTime)
        if labels_A[idx] == labels_B[i]:
            nn_count_correct += 1

    end = time.time()
    timpul_de_executie = end - start

    nn_rate = nn_count_correct / person_num
    avg_recognition_time = timpul_de_executie/person_num

    print("┌ STATISTICA")
    print(f"├ Timpul total de executare pentru {person_num} persoane: {timpul_de_executie:.4f}s")
    print(f"├ Timpul mediu per imagine: {timpul_de_executie/person_num:.4f}s")
    print(f"├  Matricea de antrenare conține: {400-person_num} imagini")
    print(f"├  Matricea de test conține: {person_num} imagini")
    print("├─────────── Rezultate NN ───────────")
    print(f"│ Norma {norma_val}: {nn_rate*100:.2f}%")
    print("└─────────────────────────────────────")
    return A.shape[1],person_num,nn_rate*100, avg_recognition_time
def run_knn(procentage,norma_val,knn_k_val):
    if procentage == '60%':
        proc_value = 6
    elif procentage == '70%':
        proc_value = 7
    elif procentage == '80%':
        proc_value = 8
    else:
        raise ValueError("Invalid percentage")

    A, B, labels_A, labels_B = constructTrainingMatrix(
        procentage_of_images_to_train=proc_value
    )
    knn_count_correct = 0 
    knn_timetaken = []
    person_num = B.shape[1]
    start = time.time()
    for i in range(person_num):
        vectorized_img = B[:, i]
        label, io, knnTime = kNN(vectorized_img, A, labels_A,norma_val, knn_k_val)
        knn_timetaken.append(knnTime)
        if label == labels_B[i]:
            knn_count_correct += 1

    end = time.time()
    timpul_de_executie = end - start
    knn_rate =knn_count_correct / person_num 
    avg_recognition_time = timpul_de_executie/person_num
    print("┌ STATISTICA")
    print(f"├ Timpul total de executare pentru {person_num} persoane: {timpul_de_executie:.4f}s")
    print(f"├ Timpul mediu per imagine: {timpul_de_executie/person_num:.4f}s")
    print(f"├  Matricea de antrenare conține: {400-person_num} imagini")
    print(f"├  Matricea de test conține: {person_num} imagini")
    print("├─────────── Rezultate kNN ───────────")
    print(f"│ Norma {norma_val}, k={knn_k_val}: {knn_rate*100:.2f}%")
    print("└─────────────────────────────────────")
    return A.shape[1],person_num,knn_rate*100, avg_recognition_time

def run_single_eigenface(photo_index, frame_to_add_graphs, procentage, K_var,norma_val=1,knn_k_val=1,knn=False):
    if procentage == '60%':
        proc_value = 6
    elif procentage == '70%':
        proc_value = 7
    elif procentage == '80%':
        proc_value = 8
    else:
        raise ValueError("Invalid percentage")

    A, B, labels_A, labels_B = constructTrainingMatrix(procentage_of_images_to_train=proc_value)

    time_extract,mean_face, eigenfaces_matrix, training_projections = m2eig.eigenfaces(A, K=K_var)

    b = B[:, photo_index]
    b_centered = b - mean_face
    w_test = eigenfaces_matrix.T @ b_centered

    training_projections_T = training_projections.T  
    if knn is False:
        idx, t = NN(w_test, training_projections_T, norma=norma_val)
        found_image = A[:, idx]
    else:
        label, io, t=kNN(w_test,training_projections_T, labels_A,norma_val,knn_k_val)
        found_image = A[:, io]


    eigenface_image = mean_face + eigenfaces_matrix @ w_test

    height = 112
    width = 92

    original_img = b.reshape((height, width))
    eigenface_img = eigenface_image.reshape((height, width))
    found_img = found_image.reshape((height, width))

    for widget in frame_to_add_graphs.winfo_children():
        widget.destroy()

    fig, axs = plt.subplots(1, 2, figsize=(6, 2))
    axs[0].imshow(original_img, cmap='gray')
    axs[0].set_title("Original")
    axs[0].axis('off')
    axs[1].imshow(found_img, cmap='gray')
    axs[1].set_title("Poza Gasita")
    axs[1].axis('off')

    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=frame_to_add_graphs)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
def run_single_eigenface_classrep(photo_index, frame_to_add_graphs, procentage, K_var,norma_val=1,knn_k_val=1,knn=False):
    if procentage == '60%':
        proc_value = 6
    elif procentage == '70%':
        proc_value = 7
    elif procentage == '80%':
        proc_value = 8
    else:
        raise ValueError("Invalid percentage")

    A, B, labels_A, labels_B = constructTrainingMatrix(procentage_of_images_to_train=proc_value)

    time_extract, mean_face, eigenfaces_matrix, training_projections, rep_matrix, class_list = \
        m2eig.eigenfaces_class_representatives(A, labels_A, K_var)

    b = B[:, photo_index]
    b_centered = b - mean_face
    w_test = eigenfaces_matrix.T @ b_centered

    if knn == False:
        predicted_class_idx, _ = NN(w_test, rep_matrix, norma_val)
    else:
        _,predicted_class_idx,holder = kNN(w,test,rep_matrix,labels_A,norma_val,knn_k_val)
    predicted_class_label = class_list[predicted_class_idx]

    found_indices = [i for i, label in enumerate(labels_A) if label == predicted_class_label]
    found_image = A[:, found_indices[0]]  

    eigenface_image = mean_face + eigenfaces_matrix @ w_test

    height = 112
    width = 92

    original_img = b.reshape((height, width))
    eigenface_img = eigenface_image.reshape((height, width))
    found_img = found_image.reshape((height, width))

    for widget in frame_to_add_graphs.winfo_children():
        widget.destroy()

    fig, axs = plt.subplots(1, 2, figsize=(6, 2))
    axs[0].imshow(original_img, cmap='gray')
    axs[0].set_title("Original")
    axs[0].axis('off')
    axs[1].imshow(found_img, cmap='gray')
    axs[1].set_title("Poza Gasita")
    axs[1].axis('off')

    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=frame_to_add_graphs)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
def run_single_lanczos(photo_index, frame_to_add_graphs, procentage, K_var,norma_val=1,knn_k_val=1,knn=False):
    if procentage == '60%':
        proc_value = 6
    elif procentage == '70%':
        proc_value = 7
    elif procentage == '80%':
        proc_value = 8
    else:
        raise ValueError("Invalid percentage")

    A, B, labels_A, labels_B = constructTrainingMatrix(procentage_of_images_to_train=proc_value)

    mean_face = np.mean(A, axis=1, keepdims=True)
    A_centered = A - mean_face
    B_centered = B - mean_face

    eigvals, eigenfaces_matrix = m2lan.lanczos(A, K_var)  

    training_projections = eigenfaces_matrix.T @ A_centered  

    b = B[:, photo_index]                     
    b_centered = b - mean_face.flatten()     
    w_test = (eigenfaces_matrix.T @ b_centered).flatten()  

    if knn == False:
        predicted_idx, _ = NN(w_test, training_projections, norma_val)
    else:
        _, predicted_idx,_ = kNN(w_test,training_projections,labels_A,norma_val,knn_k_val)
    found_image = A[:, predicted_idx]

    eigenface_image = mean_face.flatten() + eigenfaces_matrix @ w_test

    height = 112
    width = 92

    original_img = b.reshape((height, width))
    eigenface_img = eigenface_image.reshape((height, width))
    found_img = found_image.reshape((height, width))

    for widget in frame_to_add_graphs.winfo_children():
        widget.destroy()

    fig, axs = plt.subplots(1, 2, figsize=(6, 2))
    axs[0].imshow(original_img, cmap='gray')
    axs[0].set_title("Original")
    axs[0].axis('off')
    axs[1].imshow(found_img, cmap='gray')
    axs[1].set_title("Poza Gasita")
    axs[1].axis('off')
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=frame_to_add_graphs)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
def run_single_nn(photo_index,frame_to_add_graphs,procentage,norma_val):
    height = 112
    width = 92
    if procentage == '60%':
        proc_value = 6
    elif procentage == '70%':
        proc_value = 7
    elif procentage == '80%':
        proc_value = 8
    else:
        raise ValueError("Invalid percentage")

    A, B, labels_A, labels_B = constructTrainingMatrix(procentage_of_images_to_train=proc_value)
    vectorized_img = B[:, photo_index]
    idx, nnTime = NN(vectorized_img, A, norma_val)
    if labels_A[idx] == labels_B[photo_index]:
        pass
    found_image = A[:,idx]

    original_img = vectorized_img.reshape((height, width))
    found_img = found_image.reshape((height, width))

    for widget in frame_to_add_graphs.winfo_children():
        widget.destroy()

    fig, axs = plt.subplots(1, 2, figsize=(6, 2))
    axs[0].imshow(original_img, cmap='gray')
    axs[0].set_title("Original")
    axs[0].axis('off')
    axs[1].imshow(found_img, cmap='gray')
    axs[1].set_title("Poza Gasita")
    axs[1].axis('off')
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=frame_to_add_graphs)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
    print(f"NORM {norma_val} ")
def run_single_knn(photo_index,frame_to_add_graphs,procentage,norma_val,knn_k_val):
    height = 112
    width = 92
    if procentage == '60%':
        proc_value = 6
    elif procentage == '70%':
        proc_value = 7
    elif procentage == '80%':
        proc_value = 8
    else:
        raise ValueError("Invalid percentage")

    A, B, labels_A, labels_B = constructTrainingMatrix(procentage_of_images_to_train=proc_value)
    vectorized_img = B[:, photo_index]
    label, idx, knnTime = kNN(vectorized_img, A, labels_A,norma_val, knn_k_val)
    if labels_A[idx] == labels_B[photo_index]:
        pass
    found_image = A[:,idx]

    original_img = vectorized_img.reshape((height, width))
    found_img = found_image.reshape((height, width))

    for widget in frame_to_add_graphs.winfo_children():
        widget.destroy()

    fig, axs = plt.subplots(1, 2, figsize=(6, 2))
    axs[0].imshow(original_img, cmap='gray')
    axs[0].set_title("Original")
    axs[0].axis('off')
    axs[1].imshow(found_img, cmap='gray')
    axs[1].set_title("Poza Gasita")
    axs[1].axis('off')
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=frame_to_add_graphs)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
    print(f"NORM {norma_val} K {knn_k_val}")


def display_graphs_based_on_text_file(frame_to_add_graphs, n, alg_type):

    # ---------------- File selection ----------------
    if alg_type == "nn":
        file_path = "nn_results.txt"
    elif alg_type == "knn":
        file_path = "knn_results.txt"
    elif alg_type == "eig":
        file_path = f"eigen_faces_result_n{n}.txt"
    elif alg_type == "eic":
        file_path = f"eigen_faces_classrep_result_n{n}.txt"
    elif alg_type == "lan":
        file_path = f"lanczos_result_n{n}.txt"
    else:
        ttk.Label(frame_to_add_graphs, text="Unknown algorithm type").pack()
        return

    # ---------------- Clear frame ----------------
    for widget in frame_to_add_graphs.winfo_children():
        widget.destroy()

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # ================= NN / KNN ==========================
        if alg_type in ("nn", "knn"):
            data = defaultdict(lambda: {
                "x": [],      
                "accuracy": [],
                "time": []
            })

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if line.lower().startswith("norma"):
                    continue

                parts = line.split("\t")

                try:
                    norm = int(parts[0])

                    if alg_type == "knn":
                        x_val = int(parts[1])
                        accuracy = float(parts[4])
                        avg_time = float(parts[5])
                        label = f"Norma {norm}"

                    else:  # NN
                        x_val = norm
                        accuracy = float(parts[3])
                        avg_time = float(parts[4])
                        label = "NN"

                except (ValueError, IndexError):
                    continue

                data[label]["x"].append(x_val)
                data[label]["accuracy"].append(accuracy)
                data[label]["time"].append(avg_time)

            if not data:
                ttk.Label(frame_to_add_graphs, text="No NN/KNN data found").pack()
                return

            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            fig.suptitle(alg_type.upper() + "Rezultate", fontsize=14, weight="bold")

            # Rata de recunoastere plot
            for label, values in data.items():
                axs[0].plot(values["x"], values["accuracy"], marker="o", label=label)

            axs[0].set_title("Rata de Recunoaștere (%)")
            axs[0].set_xlabel("K" if alg_type == "knn" else "Norma")
            axs[0].set_ylabel("Rata de recunoastere")
            axs[0].grid(True)
            axs[0].legend()

            for label, values in data.items():
                axs[1].plot(values["x"], values["time"], marker="s", label=label)

            axs[1].set_title("Timp mediu de recunoaștere (s)")
            axs[1].set_xlabel("K" if alg_type == "knn" else "Norma")
            axs[1].set_ylabel("Timp / imagine")
            axs[1].grid(True)
            axs[1].legend()

            plt.tight_layout(rect=[0, 0, 1, 0.95])

        # ================ EIG / EIC / LAN ====================
        else:
            Ks, accuracies, avg_times, preprocessing_times = [], [], [], []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if line.lower().startswith("k"):
                    continue

                parts = line.split("\t")
                if len(parts) < 7:
                    continue

                try:
                    K = int(parts[0])
                    accuracy = float(parts[4])
                    avg_time = float(parts[5])
                    preprocessing = float(parts[6].replace("s", ""))
                except ValueError:
                    continue

                Ks.append(K)
                accuracies.append(accuracy)
                avg_times.append(avg_time)
                preprocessing_times.append(preprocessing)

            if not Ks:
                ttk.Label(frame_to_add_graphs, text="No EIG/EIC/LAN data found").pack()
                return

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            fig.suptitle(alg_type.upper() + f" Rezultate Norma:{n}", fontsize=14, weight="bold")

            axs[0].plot(Ks, accuracies, marker="o")
            axs[0].set_title("Rata de Recunoaștere (%)")
            axs[0].set_xlabel("K")
            axs[0].set_ylabel("Rata de recunoastere")
            axs[0].grid(True)

            axs[1].plot(Ks, avg_times, marker="s")
            axs[1].set_title("Timp mediu de recunoaștere (s)")
            axs[1].set_xlabel("K")
            axs[1].set_ylabel("Timp / imagine")
            axs[1].grid(True)

            axs[2].plot(Ks, preprocessing_times, marker="^")
            axs[2].set_title("Timp de preprocesare (s)")
            axs[2].set_xlabel("K")
            axs[2].set_ylabel("Timp")
            axs[2].grid(True)

            plt.tight_layout(rect=[0, 0, 1, 0.95])

        # ---------------- Embed in Tkinter ----------------
        canvas = FigureCanvasTkAgg(fig, master=frame_to_add_graphs)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    except FileNotFoundError:
        ttk.Label(frame_to_add_graphs, text=f"File {file_path} not found").pack()
    except Exception as e:
        ttk.Label(frame_to_add_graphs, text=f"Error: {e}").pack()
