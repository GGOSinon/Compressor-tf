from tkinter import *
from tkinter.filedialog import askopenfilename, asksaveasfilename
import tkinter.font as font
import os
import sys

path = ' '
new_path = ' '

def f_path():
    global path
    path = askopenfilename()
    t_path['text'] = path
    print("path : "+ path)
    
def f_newpath():
    global new_path
    new_path = asksaveasfilename()
    t_newpath['text'] = new_path
    print("new_path : "+ new_path)

def f_start():        
    print("Started compressing " + path)
    if radioVar.get() == 1: qf = t_qf.get()
    else: qf = t_cr.get()
    S = "python AIC_main.py --qf="+qf+" --path="+path+" --new_path="+new_path+" --mode=com"
    if checkVar.get()==True: S = S + " --extract=True"
    print("Executed :\n" + S)
    os.system(S)

def f_cancel():
    print("Cancelled")
    sys.exit()

def f_qf():
    print("QF")
    t_cr['state']='disabled'
    t_qf['state']='normal'

def f_cr():
    print("CR")
    t_cr['state']='normal'
    t_qf['state']='disabled'

    
root = Tk()
root.title("AIC-compressor")
root.geometry('500x280')
#root.withdraw()

bg = Label(root, text = ' ', anchor=CENTER, bg='white')
bg.place(x=25, y=25, width=450, height=225)

dy = 5
t1 = Label(root, text = 'Image to compress', anchor = W, justify = LEFT, bg='white')
t1.place(x=35, y=35+dy)
t_path = Label(root, text = '', anchor = W, justify = LEFT, bg='white', borderwidth=1, padx=2, pady=2, relief='solid')
t_path.place(x=35, y=60+dy, width=420)
b1 = Button(root, text = "Browse...", anchor = W, justify = RIGHT, padx = 5, command = f_path, borderwidth=1)
b1.place(x=390, y=33+dy)

dy = 60
t2 = Label(root, text = 'Destination', anchor = W, justify = LEFT, bg='white')
t2.place(x=35, y=35+dy)
t_newpath = Label(root, text = '', anchor = W, justify = LEFT, bg='white', borderwidth=1, padx=2, pady=2, relief='solid')
t_newpath.place(x=35, y=60+dy, width=420)
b2 = Button(root, text = "Browse...", anchor = W, justify = RIGHT, padx = 5, command = f_newpath, borderwidth=1)
b2.place(x=390, y=33+dy)

radioVar = IntVar()
radioVar.set(1)
dy = 115
tqf = Radiobutton(root, text = 'Quality Factor', bg='white', command=f_qf, variable=radioVar, value=1)
tqf.place(x=35, y=35+dy)
t_qf = Entry(root, justify = LEFT, bg='white', borderwidth=1,  relief='solid')
t_qf.place(x=35, y=60+dy, width=200)

tcr = Radiobutton(root, text = 'Compression Rate', bg='white', command=f_cr, variable=radioVar, value=2)
tcr.place(x=255, y=35+dy)
t_cr = Entry(root, justify = LEFT, bg='white', borderwidth=1,  relief='solid', state='disabled')
t_cr.place(x=255, y=60+dy, width=200)

checkVar = BooleanVar()
tex = Checkbutton(root, text = 'Additional extract', bg='white', var=checkVar, onvalue=True, offvalue=False)
tex.place(x=35, y=210)

#Bigfont = font.Font(family='Default', size=15)
bStart = Button(root, text = "Start", anchor = CENTER, justify = CENTER, padx = 10, command = f_start)
bStart.place(x=300, y=210, width=65)

bStart = Button(root, text = "Cancel", anchor = CENTER, justify = CENTER, padx = 10, command = f_cancel)
bStart.place(x=390, y=210, width=65)

root.mainloop()
