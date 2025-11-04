import numpy as np
import matplotlib.pyplot as plt

def plot_execution_times(nn_times, knn_times, title="Timpul De Exec Pt O Imagine"):
    nn_times = np.array(nn_times)
    knn_times = np.array(knn_times)
    
    num_images = len(nn_times)
    x = np.arange(1, num_images + 1)  

    plt.figure(figsize=(12, 6))
    plt.plot(x, nn_times, label='NN', marker='o', linestyle='-', color='blue')
    plt.plot(x, knn_times, label='kNN', marker='x', linestyle='--', color='orange')
    
    plt.xlabel('Index imagine')
    plt.ylabel('Timp exec (s)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

