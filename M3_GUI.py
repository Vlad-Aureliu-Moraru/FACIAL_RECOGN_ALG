from tkinter import *
from tkinter import ttk
import M2_helper as m2h
from functools import partial

window = Tk()
window.geometry("840x800")
window.title("ML Algoritmi")

# ---------- Control Panel ----------
control_panel = ttk.Frame(window, relief=RAISED, borderwidth=1)
control_panel.place(x=20, y=20, width=800, height=100)

# ---------- Single Photo Panel ----------
single_photo_panel= ttk.Frame(window, relief=RAISED, borderwidth=1)
single_photo_panel.place(x=20, y=140, width=800, height=150)


# ---------- input_panel ----------
input_panel= ttk.Frame(window, relief=RAISED, borderwidth=1)
input_panel.place(x=20, y=310, width=300, height=400)


# ---------- statistic_panel ----------
statistic_panel= ttk.Frame(window, relief=RAISED, borderwidth=1)
statistic_panel.place(x=340, y=310, width=480, height=400)

# ---------- Combobox Algoritmi----------

algo_var = StringVar()  

algo_label = ttk.Label(control_panel, text="Selecteaza tipul algoritmului:", width=40, wraplength=300,font=("Arial",12))
algo_label.place(x=10, y=10)

algo_combo = ttk.Combobox(control_panel, width=20, textvariable=algo_var, state="readonly",font=("Arial",12))
algo_combo['values'] = ('Eigenfaces clasic', 'Eigenfaces cu repr. de clasa', 'Lanczos','NN','KNN')
algo_combo.current(0)  
algo_combo.place(x=300, y=10)

apply_alg_btn = ttk.Button(control_panel,text="Aplica",width=30)
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
