from tkinter import *
from tkinter import ttk
import M2_helper as m2h
from functools import partial

window = Tk()
window.geometry("1000x800")
window.title("ML Algoritmi")

# ---------- Control Panel ----------
control_panel = ttk.Frame(window, relief=RAISED, borderwidth=1)
control_panel.place(x=0, y=0, width=430, height=1000)

# ---------- Graphs Panel ----------
graph_panel= ttk.Frame(window, relief=RAISED, borderwidth=1)
graph_panel.place(x=670, y=0, width=1200, height=250)

# ---------- Graphs_statistic Panel ----------
graph_canvas = Canvas(window, bg="white")
graph_scrollbar_v = Scrollbar(window, orient=VERTICAL, command=graph_canvas.yview)
graph_scrollbar_h = Scrollbar(window, orient=HORIZONTAL, command=graph_canvas.xview)

graph_statistic_panel = Frame(graph_canvas, relief=RAISED, borderwidth=3, bg="lightgray")

def on_frame_configure(event):
    graph_canvas.configure(scrollregion=graph_canvas.bbox("all"))

graph_statistic_panel.bind("<Configure>", on_frame_configure)

graph_canvas.create_window((0, 0), window=graph_statistic_panel, anchor="nw")

graph_canvas.configure(yscrollcommand=graph_scrollbar_v.set, xscrollcommand=graph_scrollbar_h.set)

graph_canvas.place(x=680, y=300, width=1200, height=700)
graph_scrollbar_v.place(x=1880, y=0, width=10, height=800)
graph_scrollbar_h.place(x=730, y=1000, width=1010, height=10)

# ---------- Photo Panel ----------
canvas = Canvas(window)
scrollbar = Scrollbar(window, orient=VERTICAL, command=canvas.yview)
photo_panel = Frame(canvas, relief=RAISED, borderwidth=1)

photo_panel.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=photo_panel, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.place(x=430, y=0, width=210, height=1000)
scrollbar.place(x=660, y=0, height=800)

# ---------- Training Percentage ----------
train_var = StringVar()  

train_label = ttk.Label(control_panel, text="Selecteaza % de img din matricea de antrenare:", width=40, wraplength=500,font=("Arial",14))
train_label.place(x=10, y=10)

train_combo = ttk.Combobox(control_panel, width=27, textvariable=train_var, state="readonly",font=("Arial",12))
train_combo['values'] = ('60%', '70%', '80%')
train_combo.current(1)  
train_combo.place(x=10, y=50)

# ---------- Algorithm Type ----------
algo_var = StringVar()  
search_algo_var = StringVar()  
norm_algo_var = StringVar()  
knn_k_var = StringVar()  

algo_label = ttk.Label(control_panel, text="Selecteaza tipul algoritmului:", width=40, wraplength=300,font=("Arial",14))
algo_label.place(x=10, y=90)

algo_combo = ttk.Combobox(control_panel, width=27, textvariable=algo_var, state="readonly",font=("Arial",12))
algo_combo['values'] = ('Eigenfaces clasic', 'Eigenfaces cu repr. de clasa', 'Lanczos')
algo_combo.current(0)  
algo_combo.place(x=10, y=120)


search_algo_combo = ttk.Combobox(control_panel, width=10, textvariable=search_algo_var, state="readonly",font=("Arial",12))
search_algo_combo['values'] = ('NN', 'kNN')
search_algo_combo.current(0)  
search_algo_combo.place(x=280, y=120)


norm_label = ttk.Label(control_panel, text="Norma :", width=10, wraplength=300,font=("Arial",10))
norm_label.place(x=280, y=140)
norm_combo = ttk.Combobox(control_panel, width=3, textvariable=norm_algo_var, state="readonly",font=("Arial",12))
norm_combo['values'] = ('1', '2','inf','cos')
norm_combo.current(0)  
norm_combo.place(x=280, y=160)

knn_k_label = ttk.Label(control_panel, text="Valoare K:", width=10, wraplength=300,font=("Arial",10))
knn_k_label.place(x=344, y=140)

knn_k_combo = ttk.Combobox(control_panel, width=3, textvariable=knn_k_var, state="readonly",font=("Arial",12))
knn_k_combo['values'] = ('1', '2','3')
knn_k_combo.current(0)  
knn_k_combo.place(x=344, y=160)

# ----------- Slider pt K ---------
k_var = IntVar(value=20)  

k_label = ttk.Label(control_panel, text="Numarul de eigenfaces (K):", width=23, wraplength=300,font=("Arial",15))
k_label.place(x=10, y=160)

k_slider = Scale(control_panel, from_=1, to=100, orient=HORIZONTAL, variable=k_var)
k_slider.place(x=10, y=190, width=200)

#--------------- Matrix contain ---------
A_size_label = ttk.Label(control_panel,text="",foreground="green",font=("Arial",15))
A_size_label.place(x=10,y=300)

B_size_label = ttk.Label(control_panel,text="",foreground="red",font=("Arial",15))
B_size_label.place(x=160,y=300)


avg_time_label = ttk.Label(control_panel,text="",foreground="blue",font=("Arial",15))
avg_time_label.place(x=10,y=350)

correctitude_label = ttk.Label(control_panel,text="",foreground="blue",font=("Arial",15))
correctitude_label.place(x=10,y=400)

preprocesing_label = ttk.Label(control_panel,text="",foreground="blue",font=("Arial",15))
preprocesing_label.place(x=10,y=450)
# ---------- Run Button ----------
custom_font = ("Helvetica", 16, "bold italic")

def run_algorithm():
    usingKNN=search_algo_var.get()=='kNN'
    int_norm_var = 1;
    if norm_algo_var.get() == '2':
        int_norm_var=2
    elif norm_algo_var.get() =='inf':
        int_norm_var=3
    elif norm_algo_var.get() =='cos':
        int_norm_var=4
    if algo_var.get() == 'Eigenfaces clasic':
        A_size ,B_size ,corect,avg_time,time_extract= m2h.run_eigenfaces(train_var.get(),k_var.get(),int_norm_var,int(knn_k_var.get()),usingKNN)
        A_size_label.config(text= f"A poze = {A_size}")
        B_size_label.config(text= f"B poze = {B_size}")
        avg_time_label.config(text= f"Timpul mediu de recunoastere = {avg_time:.4f} sec")
        correctitude_label.config(text= f"Rata de identificare = {corect*100:.2f}%")
        preprocesing_label.config(text= f"Timpul de preprocesare = {time_extract:.2f} sec")

    elif algo_var.get()=='Eigenfaces cu repr. de clasa':
        A_size ,B_size ,corect,avg_time,time_extract= m2h.run_eigenfaces_class_rep(train_var.get(),k_var.get(),int_norm_var,int(knn_k_var.get()),usingKNN)
        A_size_label.config(text= f"A poze = {A_size}")
        B_size_label.config(text= f"B poze = {B_size}")
        avg_time_label.config(text= f"Timpul mediu de recunoastere = {avg_time:.4f} sec")
        correctitude_label.config(text= f"Rata de identificare = {corect*100:.2f}%")
        preprocesing_label.config(text= f"Timpul de preprocesare = {time_extract:.2f} sec")
    elif algo_var.get()=='Lanczos':
        A_size ,B_size ,corect,avg_time,time_extract= m2h.run_lanczos(train_var.get(),k_var.get(),int_norm_var,int(knn_k_var.get()),usingKNN)
        A_size_label.config(text= f"A poze = {A_size}")
        B_size_label.config(text= f"B poze = {B_size}")
        avg_time_label.config(text= f"Timpul mediu de recunoastere = {avg_time:.4f} sec")
        correctitude_label.config(text= f"Rata de identificare = {corect*100:.2f}%")
        preprocesing_label.config(text= f"Timpul de preprocesare = {time_extract:.2f} sec")

def run_single_test(index):
    print(f"Running single test for Photo {index+1}")
    if algo_var.get() == 'Eigenfaces clasic':
        m2h.run_single_eigenface(index,graph_panel,train_var.get(),k_var.get())
    elif algo_var.get()=='Eigenfaces cu repr. de clasa':
        m2h.run_single_eigenface_classrep(index,graph_panel,train_var.get(),k_var.get())
    elif algo_var.get()=='Lanczos':
        m2h.run_single_lanczos(index,graph_panel,train_var.get(),k_var.get())

def select_photos():
    num_test_images = 160
    if train_var.get() == '70%':
        num_test_images = 120
    elif train_var.get() == '80%':
        num_test_images = 80

    for widget in photo_panel.winfo_children():
        widget.destroy()

    for i in range(num_test_images):
        btn = Button(photo_panel, text=f"Photo {i+1}",
                     width=10, command=partial(run_single_test, i))
        row = i // 2
        col = i % 2
        btn.grid(row=row, column=col, padx=2, pady=3)

def show_prev_statistic():
    m2h.display_graphs_based_on_text_file(graph_statistic_panel)

run_btn = Button(control_panel, text="Ruleaza Algoritmul", command=run_algorithm,font = ("Arial",12))
run_btn.place(x=10, y=250)

select_single_person_btn = Button (control_panel,text="Selecteaza o singura persoana",command=select_photos,font = ("Arial",12))
select_single_person_btn.place(x=170, y=250)

show_statistic_prev_btn = Button (control_panel,text="Afiseaza Statistica",command=show_prev_statistic,font = ("Arial",12))
show_statistic_prev_btn.place(x=10, y=650)


window.mainloop()
