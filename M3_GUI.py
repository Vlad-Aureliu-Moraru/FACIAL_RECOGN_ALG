from tkinter import *
from tkinter import ttk
import M2_helper as m2h
from functools import partial

def select_photos():
    adapt_input_panel()
    num_test_images = 160
    if train_var.get() == '70%':
        num_test_images = 120
    elif train_var.get() == '80%':
        num_test_images = 80

    for widget in photo_panel.winfo_children():
        widget.destroy()

    for i in range(num_test_images):
        btn = Button(photo_panel, text=f"Photo {i+1}",
                     width=5, command=partial(run_single_test ,i,statistic_panel))
        row = i //12
        col = i % 12
        btn.grid(row=row, column=col, padx=2, pady=3)

def run_single_test(index,graph_panel):
    print(f"Running single test for Photo {index+1}")
    if algo_var.get() == 'Eigenfaces clasic':
        m2h.run_single_eigenface(index,graph_panel,train_var.get(),k_var.get())
    elif algo_var.get()=='Eigenfaces cu repr. de clasa':
        m2h.run_single_eigenface_classrep(index,graph_panel,train_var.get(),k_var.get())
    elif algo_var.get()=='Lanczos':
        m2h.run_single_lanczos(index,graph_panel,train_var.get(),k_var.get())
    elif algo_var.get() == "NN":
        m2h.run_single_nn(index,graph_panel,train_var.get(),nn_norm_var.get())
    elif algo_var.get() == "KNN":
        m2h.run_single_knn(index,graph_panel,train_var.get(),nn_norm_var.get(),knn_k_var.get())

def search_photo():
    input = search_text_box.get("1.0","end-1c")
    print(f"index of photo, {input}")
    num_test_images = 160
    if train_var.get() == '70%':
        num_test_images = 120
    elif train_var.get() == '80%':
        num_test_images = 80
    try:
        photo_index = int(input)
        if photo_index>num_test_images:
            raise ValueError
        for widget in photo_panel.winfo_children():
            widget.destroy()
        btn = Button(photo_panel, text=f"Photo {photo_index}",
                     width=5, command=partial(run_single_test ,photo_index-1,statistic_panel))
        btn.grid(row=1, column=1, padx=2, pady=3)
    except ValueError:
        print("Invalid INPUT")
        search_text_box.delete('1.0', END)
    
def compute():
    if algo_var.get() == "NN":
        A_size ,B_size,corect,avg_time= m2h.run_nn(train_var.get(),nn_norm_var.get())
        avg_time_label.config(text= f"Timpul mediu de recunoastere = {avg_time:.4f} sec")
        correctitude_label.config(text= f"Rata de identificare = {corect:.2f}%")
        preprocesing_label.config(text= f"")
    elif algo_var.get() == "KNN":
        A_size ,B_size,corect,avg_time= m2h.run_knn(train_var.get(),nn_norm_var.get(),knn_k_var.get())
        avg_time_label.config(text= f"Timpul mediu de recunoastere = {avg_time:.4f} sec")
        correctitude_label.config(text= f"Rata de identificare = {corect:.2f}%")
        preprocesing_label.config(text= f"")
    elif algo_var.get() =='Eigenfaces clasic':
        A_size ,B_size ,corect,avg_time,time_extract=m2h.run_eigenfaces(train_var.get(),k_var.get(),nn_norm_var.get())
        avg_time_label.config(text= f"Timpul mediu de recunoastere = {avg_time:.4f} sec")
        correctitude_label.config(text= f"Rata de identificare = {corect*100:.2f}%")
        preprocesing_label.config(text= f"Timpul de preprocesare = {time_extract:.2f} sec")
    elif algo_var.get() =='Eigenfaces cu repr. de clasa':
        A_size ,B_size ,corect,avg_time,time_extract=m2h.run_eigenfaces_class_rep(train_var.get(),k_var.get(),nn_norm_var.get())
        avg_time_label.config(text= f"Timpul mediu de recunoastere = {avg_time:.4f} sec")
        correctitude_label.config(text= f"Rata de identificare = {corect*100:.2f}%")
        preprocesing_label.config(text= f"Timpul de preprocesare = {time_extract:.2f} sec")
    elif algo_var.get() =='Lanczos' :
        A_size ,B_size ,corect,avg_time,time_extract=m2h.run_lanczos(train_var.get(),k_var.get(),nn_norm_var.get())
        avg_time_label.config(text= f"Timpul mediu de recunoastere = {avg_time:.4f} sec")
        correctitude_label.config(text= f"Rata de identificare = {corect*100:.2f}%")
        preprocesing_label.config(text= f"Timpul de preprocesare = {time_extract:.2f} sec")

def adapt_input_panel():
    print("running adapt ")
    for widget in input_panel.winfo_children():
        widget.destroy()

    if algo_var.get() == "NN":
        print("NN PICKED")
        ttk.Label(input_panel, text="NN Norma:").pack(padx=5, pady=5, side=TOP)
        norm_combobox = ttk.Combobox(
            input_panel, 
            textvariable=nn_norm_var, 
            values=list(range(1, 5)), 
            state="readonly",
            width=5
        )
        norm_combobox.pack(padx=5, pady=5, side=TOP)
        norm_combobox.current(0)
        calculeaza_button = ttk.Button(input_panel,text="Calculeaza",width=30,command=compute)
        calculeaza_button.pack(padx=5, pady=5, side=TOP)
    elif algo_var.get() == "KNN":
        print("KNN PICKED")
        ttk.Label(input_panel, text="NN Norm:").pack(padx=5, pady=5, side=TOP)
        norm_combobox = ttk.Combobox(
            input_panel, 
            textvariable=nn_norm_var, 
            values=list(range(1, 5)), 
            state="readonly",
            width=5
        )
        norm_combobox.pack(padx=5, pady=5, side=TOP)
        ttk.Label(input_panel, text="K valori:").pack(padx=5, pady=5, side=TOP) # <<< Added Label
        k_combobox = ttk.Combobox(
            input_panel, 
            textvariable=knn_k_var, 
            values=[3, 5, 7], 
            state="readonly",
            width=5
        )
        k_combobox.pack(padx=5, pady=5, side=TOP)
        norm_combobox.set(nn_norm_var) 
        k_combobox.set(knn_k_var)
        norm_combobox.current(0)
        k_combobox.current(0)
        calculeaza_button = ttk.Button(input_panel,text="Calculeaza",width=30,command=compute)
        calculeaza_button.pack(padx=5, pady=5, side=TOP)
    elif algo_var.get() in['Eigenfaces clasic','Eigenfaces cu repr. de clasa','Lanczos'] :
        print(f"{k_var.get()} TO SS")
        k_label = ttk.Label(input_panel, text="Nr de eigenfaces (K):", width=23, wraplength=300,font=("Arial",11))
        k_label.pack(padx=5, pady=5, side=TOP)
        k_slider = Scale(input_panel, from_=1, to=100, orient=HORIZONTAL, variable=k_var)
        k_slider.pack(padx=5, pady=5, side=TOP)
        ttk.Label(input_panel, text="NN Norm:").pack(padx=5, pady=5, side=TOP)
        norm_combobox = ttk.Combobox(
            input_panel, 
            textvariable=nn_norm_var, 
            values=list(range(1, 5)), 
            state="readonly",
            width=5
        )
        norm_combobox.pack(padx=5, pady=5, side=TOP)
        norm_combobox.current(0)
        calculeaza_button = ttk.Button(input_panel,text="Calculeaza",width=30,command=compute)
        calculeaza_button.pack(padx=5, pady=5, side=TOP)

window = Tk()
window.geometry("900x840")
window.title("ML Algoritmi")

train_var = StringVar()  
k_var = IntVar(value=20)  
nn_norm_var = IntVar(value=1)
knn_k_var = IntVar(value=3)
algo_var = StringVar(value="NN") 

#blue ---------- Control Panel ----------

control_panel = ttk.Frame(window, relief=RAISED, borderwidth=1)
control_panel.place(x=20, y=20, width=860, height=100)

    # blue |---------- Split The DB----------

train_label = ttk.Label(control_panel, text="Selecteaza % din matricea de antrenare:", width=40, wraplength=500,font=("Arial",12))
train_label.place(x=10, y=50)

train_combo = ttk.Combobox(control_panel, width=10, textvariable=train_var, state="readonly",font=("Arial",12))
train_combo['values'] = ('60%', '70%', '80%')
train_combo.current(1)  
train_combo.place(x=300, y=50)

    # blue|---------- Combobox Algoritmi----------

algo_var = StringVar()  

algo_label = ttk.Label(control_panel, text="Selecteaza tipul algoritmului:", width=40, wraplength=300,font=("Arial",12))
algo_label.place(x=10, y=10)

algo_combo = ttk.Combobox(control_panel, width=20, textvariable=algo_var, state="readonly",font=("Arial",12))
algo_combo['values'] = ('Eigenfaces clasic', 'Eigenfaces cu repr. de clasa', 'Lanczos','NN','KNN')
algo_combo.current(0)  
algo_combo.place(x=300, y=10)

apply_alg_btn = ttk.Button(control_panel,text="Aplica",width=30,command=select_photos)
apply_alg_btn.place(x=530,y=10)

    #blue| ---------- Search Photo ----------

search_photo_val= StringVar()  

search_photo_label= ttk.Label(control_panel, text="Search :", width=10, wraplength=500,font=("Arial",12))
search_photo_label.place(x=530, y=50)

search_text_box = Text(control_panel, height=1,width=10,font=("Arial",12))
search_text_box.place(x=600, y=50)

search_btn = ttk.Button(control_panel,text=" ",width=5,command=search_photo)
search_btn.place(x=700,y=50)

    #blue| ---------- Show Statistics Button ----------

show_statistic_btn = Button(control_panel, text="Statistica  ",width=8 )
show_statistic_btn.place(x=770,y=9)

# ---------- Single Photo Panel ----------

canvas = Canvas(window)
scrollbar = Scrollbar(window, orient=VERTICAL, command=canvas.yview)
photo_panel = Frame(canvas, relief=RAISED, borderwidth=1)

photo_panel.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=photo_panel, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)


canvas.place(x=20, y=140, width=860, height=150)
scrollbar.place(x=870, y=140, height=150)

# ---------- input_panel ----------

input_panel= ttk.Frame(window, relief=RAISED, borderwidth=1)
input_panel.place(x=20, y=310, width=150, height=400)


# ---------- statistic_panel ----------

statistic_panel= ttk.Frame(window, relief=RAISED, borderwidth=1)
statistic_panel.place(x=180, y=310, width=680, height=400)



# ---------- Display Stats ----------

display_stats_panel = ttk.Frame(window, relief=RAISED, borderwidth=1)
display_stats_panel.place(x=20, y=720, width=860, height=100)

avg_time_label = ttk.Label(display_stats_panel,text="",foreground="blue",font=("Arial",11))
avg_time_label.pack(padx=5, pady=5, side=TOP)

correctitude_label = ttk.Label(display_stats_panel,text="",foreground="blue",font=("Arial",11))
correctitude_label.pack(padx=5, pady=5, side=TOP)

preprocesing_label = ttk.Label(display_stats_panel,text="",foreground="blue",font=("Arial",11))
preprocesing_label.pack(padx=5, pady=5, side=TOP)

window.mainloop()
