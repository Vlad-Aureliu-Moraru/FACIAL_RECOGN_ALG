import M2_helper as m2h

norm_vals = [1,2,3,4]
k_vals = [20,40,60,80,100]
def create_files_eigen():
    for k in norm_vals:
        output_file = f"eigen_faces_result_n{k}.txt"
        with open(output_file, "w") as f:
            f.write("K\tTraining Images\tTesting Images\tAccuracy\tAvg Recognition Time(s)\tPreprocessing Time\n")

            for k_value in k_vals:
                    A_size, B_size, accuracy, avg_time,preprocessing_time= m2h.run_eigenfaces("80%",k_value ,k)
                    
                    f.write(f"{k_value}\t{k}\t{A_size}\t{B_size}\t{accuracy:.2f}\t{avg_time:.6f}\t{preprocessing_time:.6f}s\n")
                    

create_files_eigen()

