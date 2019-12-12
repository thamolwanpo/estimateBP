from tkinter import *
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure 
from matplotlib.animation import FuncAnimation
import numpy as np
from itertools import count
import pandas as pd
import sys
sys.path.append("..")
from model import create_model
from util import zero_order_holding_first, zero_order_second
import time

index = count()
timewindow = 32
display_length = 250
x_axis = np.arange(0, 250)

fig = Figure(figsize=(15, 10), dpi=50)
ppg_graph = fig.add_subplot(311)
ecg_graph = fig.add_subplot(312)
abp_graph = fig.add_subplot(313)

ppg_count = 32
ecg_count = 32
abp_count = 32

ppg_display = [0] * (display_length-32)
ecg_display = [0] * (display_length-32)
abp_display = [0] * (display_length-32)
pred_display = [0] * (display_length-32)

sys_gt = 0
dia_gt = 0
sys_pred = 0
dia_pred = 0

sys_gtnumlabel = None
dia_gtnumlabel = None
sys_prednumlabel = None
dia_prednumlabel = None
subjectnumlabel = None

counter = 0

def animate_ppg_graph(i):
    global ppg_graph, ppg_display, ppg_count, subjectnumlabel

    data = pd.read_csv('data.csv')
    ls = data['ppg'].values
    subject_ls = data['subject'].values
    subjectnumlabel.config(text=str(subject_ls[0]))

    if ppg_count == 32:
        for i in range(32):
            ppg_display.append(ls[i])
    else:
        ppg_display = ppg_display[1:]
        ppg_display.append(ls[ppg_count])

    ppg_count += 1

    ppg_graph.clear()
    ppg_graph.set_ylabel("Amplitude")
    ppg_graph.plot(x_axis, ppg_display, label='PPG')
    ppg_graph.set_title('PPG Signal')
    ppg_graph.axis(ymin=0.0, ymax=1.0)
    ppg_graph.legend()


def animate_ecg_graph(i):
    global ecg_graph, ecg_count, ecg_display

    data = pd.read_csv('data.csv')
    ls = data['ecg'].values

    if ecg_count == 32:
        for i in range(32):
            ecg_display.append(ls[i])
    else:
        ecg_display = ecg_display[1:]
        ecg_display.append(ls[ecg_count])

    ecg_count += 1

    ecg_graph.clear()
    ecg_graph.set_ylabel("Amplitude")
    ecg_graph.plot(x_axis, ecg_display, label='ECG')
    ecg_graph.set_title('ECG Signal')
    ecg_graph.axis(ymin=0.0,ymax=1.0)
    ecg_graph.legend()

def animate_bp_graph(i):
    global abp_graph, abp_count, abp_display, model, pred_display, sys_gt, dia_gt, sys_gtnumlabel, dia_gtnumlabel

    data = pd.read_csv('data.csv')
    ls = data['abp'].values

    inputs = []
    ppg = ppg_display[-(timewindow):]
    ecg = ecg_display[-(timewindow):]
    for i in range(timewindow):
        inputs.append([(ppg[i]-ppg_all_min)/(ppg_all_max-ppg_all_min), (ecg[i]-ecg_all_min)/(ecg_all_max-ecg_all_min)])

    inputs = np.array([inputs])

    pred = model.predict([inputs, np.zeros((1, timewindow, 1))], batch_size=1)

    pred_value = (pred[0][-1][0] * (abp_all_max-abp_all_min)) + abp_all_min

    if abp_count == 32:
        for i in range(32):
            abp_display.append(ls[i])
            pred_display.append(pred_value)
    else:
        abp_display = abp_display[1:]
        pred_display = pred_display[1:]
        abp_display.append(ls[abp_count])
        pred_display.append(pred_value)

    sys_gt = zero_order_holding_first(abp_display[-125:], delay=125, is_sys=True)
    dia_gt = zero_order_holding_first(abp_display[-125:], delay=125, is_sys=False)
    sys_pred = zero_order_holding_first(pred_display[-125:], delay=125, is_sys=True)
    dia_pred = zero_order_holding_first(pred_display[-125:], delay=125, is_sys=False)

    sys_gt = zero_order_second(sys_gt)
    dia_gt = zero_order_second(dia_gt)
    sys_pred = zero_order_second(sys_pred)
    dia_pred = zero_order_second(dia_pred)

    sys_gtnumlabel.config(text=str("{:.3f}".format(sys_gt[-1])))
    dia_gtnumlabel.config(text=str("{:.3f}".format(dia_gt[-1])))
    sys_prednumlabel.config(text=str("{:.3f}".format(sys_pred[-1])))
    dia_prednumlabel.config(text=str("{:.3f}".format(dia_pred[-1])))

    abp_count += 1

    abp_graph.clear()
    abp_graph.set_ylabel("Amplitude")
    abp_graph.plot(x_axis, abp_display, label='Ground Truth')
    abp_graph.plot(x_axis, pred_display, label='Predicted')
    abp_graph.set_title('Estimated BP Signal')
    abp_graph.axis(ymin=40,ymax=170)
    abp_graph.legend()

def updateALL(frameNum):
    global counter, start_time
    animate_ppg_graph(frameNum)
    animate_ecg_graph(frameNum)
    animate_bp_graph(frameNum)

    counter += 1

    if (time.time() - start_time) > 1 :
        print("FPS: ", counter / (time.time() - start_time))
        counter = 0
        start_time = time.time()
    return counter

def run_display():
    global sys_gtnumlabel, dia_gtnumlabel, sys_prednumlabel, dia_prednumlabel, subjectnumlabel
    window = Tk()
    window.geometry('1200x600')
    window.title('Demo Blood Pressure Estimation')
    window.configure(background='white')

    title = Label(window, text="Demo for Blood Pressure Estimation using Deep Learning", bg="orange red", fg="white", font="none 24 bold")
    title.grid(row=0, columnspan=1200, sticky=W, padx=(150,0))

    space = Label(window, text=' ', bg="white", font="none 64 bold")
    space.grid(row=2, column=2)

    subjectlabel = Label(window, text='Subject:', font="none 12 bold", bg="white")
    subjectlabel.grid(row=3, column=0)
    syslabel = Label(window, text='Systolic BP Estimation     ', font="none 12 bold", bg="white")
    syslabel.grid(row=6, column=0)
    dialabel = Label(window, text='Diastolic BP Estimation     ', font="none 12 bold", bg="white")
    dialabel.grid(row=7, column=0)

    predlabel = Label(window, text='Predicted', font="none 12 bold", bg="white")
    predlabel.grid(row=5, column=1)
    space1 = Label(window, text='         ', bg="white")
    space1.grid(row=5, column=2)
    gtlabel = Label(window, text='Ground Turth', font="none 12 bold", bg="white")
    gtlabel.grid(row=5, column=3)

    subjectnumlabel = Label(window, text=str(0), font="none 12", bg="white")
    subjectnumlabel.grid(row=3, column=1)
    sys_prednumlabel = Label(window, text=str(0), font="none 12", bg="white")
    sys_prednumlabel.grid(row=6, column=1)
    sys_gtnumlabel = Label(window, text=str(0), font="none 12", bg="white")
    sys_gtnumlabel.grid(row=6, column=3)
    dia_prednumlabel = Label(window, text=str(0), font="none 12", bg="white")
    dia_prednumlabel.grid(row=7, column=1)
    dia_gtnumlabel = Label(window, text=str(0), font="none 12", bg="white")
    dia_gtnumlabel.grid(row=7, column=3)

    graph_canvas = FigureCanvasTkAgg(fig, master=window)  # A tk.DrawingArea.
    graph_canvas.draw()
    graph_canvas.get_tk_widget().grid(row=1, column=4, rowspan=100)

    anim = FuncAnimation(fig, updateALL, interval=1, frames=200)

    window.mainloop()

if __name__ == '__main__':
    model = create_model()
    model.load_weights('../weights/model-weight.h5')
    min_max = np.load('../data/min_max.npy')
    ppg_all_min, ppg_all_max = min_max[0]
    ecg_all_min, ecg_all_max = min_max[1]
    abp_all_min, abp_all_max = min_max[2]

    start_time = time.time()

    run_display()