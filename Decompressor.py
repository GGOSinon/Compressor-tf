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
    print("Executed :\n" + "python AIC_main.py --path="+path+" --new_path="+new_path+" --mode=dec")
    os.system("python AIC_main.py --path="+path+" --new_path="+new_path+" --mode=dec")

def f_cancel():
    print("Cancelled")
    sys.exit()

root = Tk()
root.title("AIC-compressor")
root.geometry('500x220')
#root.withdraw()

bg = Label(root, text = ' ', anchor=CENTER, bg='white')
bg.place(x=25, y=25, width=450, height=165)

dy = 5
t1 = Label(root, text = 'AIC file to decompress', anchor = W, justify = LEFT, bg='white')
t1.place(x=35, y=35+dy)
t_path = Label(root, text = '', anchor = W, justify = LEFT, bg='white', borderwidth=1, padx=2, pady=2, relief='solid')
t_path.place(x=35, y=60+dy, width=420)
b1 = Button(root, text = "Browse...", anchor = W, justify = RIGHT, padx = 5, command = f_path, borderwidth=1)
b1.place(x=390, y=32+dy)

dy = 60
t2 = Label(root, text = 'Destination', anchor = W, justify = LEFT, bg='white')
t2.place(x=35, y=35+dy)
t_newpath = Label(root, text = '', anchor = W, justify = LEFT, bg='white', borderwidth=1, padx=2, pady=2, relief='solid')
t_newpath.place(x=35, y=60+dy, width=420)
b2 = Button(root, text = "Browse...", anchor = W, justify = RIGHT, padx = 5, command = f_newpath, borderwidth=1)
b2.place(x=390, y=32+dy)

#Bigfont = font.Font(family='Default', size=15)
bStart = Button(root, text = "Start", anchor = CENTER, justify = CENTER, padx = 10, command = f_start)
bStart.place(x=300, y=152, width=65)

bStart = Button(root, text = "Cancel", anchor = CENTER, justify = CENTER, padx = 10, command = f_cancel)
bStart.place(x=390, y=152, width=65)

root.mainloop()
