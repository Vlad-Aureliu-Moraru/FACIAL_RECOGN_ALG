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
                     width=5, command=partial(run_single_test, i))
        row = i //12
        col = i % 12
        btn.grid(row=row, column=col, padx=2, pady=3)

def run_single_test(index):
    print(f"Running single test for Photo {index+1}")
    if algo_var.get() == 'Eigenfaces clasic':
        m2h.run_single_eigenface(index,graph_panel,train_var.get(),k_var.get())
    elif algo_var.get()=='Eigenfaces cu repr. de clasa':
        m2h.run_single_eigenface_classrep(index,graph_panel,train_var.get(),k_var.get())
    elif algo_var.get()=='Lanczos':
        m2h.run_single_lanczos(index,graph_panel,train_var.get(),k_var.get())

def adapt_input_panel():
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
            
            norm_combobox.set(nn_norm_var)
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
        
        norm_combobox.set(nn_norm_var)

window = Tk()
window.geometry("900x800")
window.title("ML Algoritmi")

k_var = IntVar(value=20)  
nn_norm_var = IntVar(value=1)
knn_k_var = IntVar(value=3)
algo_var = StringVar(value="NN") 

# ---------- Control Panel ----------
control_panel = ttk.Frame(window, relief=RAISED, borderwidth=1)
control_panel.place(x=20, y=20, width=860, height=100)

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

# ---------- Combobox Algoritmi----------

algo_var = StringVar()  

algo_label = ttk.Label(control_panel, text="Selecteaza tipul algoritmului:", width=40, wraplength=300,font=("Arial",12))
algo_label.place(x=10, y=10)

algo_combo = ttk.Combobox(control_panel, width=20, textvariable=algo_var, state="readonly",font=("Arial",12))
algo_combo['values'] = ('Eigenfaces clasic', 'Eigenfaces cu repr. de clasa', 'Lanczos','NN','KNN')
algo_combo.current(0)  
algo_combo.place(x=300, y=10)

apply_alg_btn = ttk.Button(control_panel,text="Aplica",width=30,command=select_photos)
apply_alg_btn.place(x=530,y=10)

# ---------- Split The DB----------
train_var = StringVar()  

train_label = ttk.Label(control_panel, text="Selecteaza % din matricea de antrenare:", width=40, wraplength=500,font=("Arial",12))
train_label.place(x=10, y=50)

train_combo = ttk.Combobox(control_panel, width=10, textvariable=train_var, state="readonly",font=("Arial",12))
train_combo['values'] = ('60%', '70%', '80%')
train_combo.current(1)  
train_combo.place(x=300, y=50)

# ---------- Search Photo ----------
search_photo_val= StringVar()  

search_photo_label= ttk.Label(control_panel, text="Search :", width=10, wraplength=500,font=("Arial",12))
search_photo_label.place(x=530, y=50)

search_text_box = Text(control_panel, height=1,width=10,font=("Arial",12))
search_text_box.place(x=600, y=50)

search_btn = ttk.Button(control_panel,text=" ",width=5)
search_btn.place(x=700,y=50)


window.mainloop()
