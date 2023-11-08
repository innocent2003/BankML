from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import numpy as np 
import pandas as pd 
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

df = pd.read_csv('UniversalBank.csv')
dt_train,dt_test = train_test_split(df,test_size=0.3,shuffle = False)

# calculate error
def error(y, y_pred):
    sum = 0
    for i in range(0, len(y)):
        sum = sum + abs(y[i] - y_pred[i])
    return sum/len(y)  # tra ve trung binh

min=999999
k = 5
kf = KFold(n_splits=k, random_state=None)
for train_index, validation_index in kf.split(dt_train):
    X_train, X_validation = dt_train.iloc[train_index,:8], dt_train.iloc[validation_index, :8]
    y_train, y_validation = dt_train.iloc[train_index, 8], dt_train.iloc[validation_index, 8]


    svm = SVC().fit(X_train,y_train)
    y_train_pred = svm.predict(X_train)
    y_validation_pred = svm.predict(X_validation)
    y_train = np.array(y_train)
    y_validation = np.array(y_validation)

    sum_error = error(y_train,y_train_pred) + error(y_validation, y_validation_pred)
    
    if(sum_error < min):
        min = sum_error
        regr=svm

y_test_pred = regr.predict(dt_test.iloc[:,:8])
y_test = np.array(dt_test.iloc[:,8])

# print("SVM:")
# print("Accuracy: ",accuracy_score(y_test,y_test_pred))
# print("Precision: ",precision_score(y_test,y_test_pred,zero_division=1))# tỉ lệ số điểm true positive trong số những điểm được phân loại là positive
# print("recall: ",recall_score(y_test,y_test_pred,zero_division=1))# tỉ lệ số điểm true positive trong số những điểm thực sự là positive
# print("f1_score: ",f1_score(y_test,y_test_pred,zero_division=1))

#form
form = Tk()
form.title("Dự đoán liệu khách hàng có khả năng chấp nhận khoản vay cá nhân do ngân hàng đưa ra hay không?")
form.geometry("1000x500")

lable_ten = Label(form, text = "Nhập thông tin cần dự doán:", font=("Arial Bold", 10), fg="red")
lable_ten.grid(row = 1, column = 1, padx = 40, pady = 10)

lable_Age = Label(form, text = "Age:")
lable_Age.grid(row = 2, column = 1, padx = 40, pady = 10)
textbox_Age = Entry(form)
textbox_Age.grid(row = 2, column = 2)

lable_Experience = Label(form, text = "Experience:")
lable_Experience.grid(row = 3, column = 1, pady = 10)
textbox_Experience = Entry(form)
textbox_Experience.grid(row = 3, column = 2)

lable_Income = Label(form, text = "Income:")
lable_Income.grid(row = 4, column = 1,pady = 10)
textbox_Income = Entry(form)
textbox_Income.grid(row = 4, column = 2)

lable_ZIPCode  = Label(form, text = "ZIPCode:")
lable_ZIPCode.grid(row = 5, column = 1, pady = 10)
textbox_ZIPCode  = Entry(form)
textbox_ZIPCode.grid(row = 5, column = 2)

lable_Family  = Label(form, text = "Family :")
lable_Family .grid(row = 6, column = 1, pady = 10 )
textbox_Family  = Entry(form)
textbox_Family .grid(row = 6, column = 2)

lable_CCAvg = Label(form, text = "CCAvg:")
lable_CCAvg.grid(row = 7, column = 1, pady = 10 )
textbox_CCAvg = Entry(form)
textbox_CCAvg.grid(row = 7, column = 2)

lable_Education= Label(form, text = "Education:")
lable_Education.grid(row = 2, column = 3, padx = 90)
textbox_Education= Entry(form)
textbox_Education.grid(row = 2, column = 4)

lable_Mortgage  = Label(form, text = "Mortgage :")
lable_Mortgage.grid(row = 3, column = 3)
textbox_Mortgage  = Entry(form)
textbox_Mortgage.grid(row = 3, column = 4)

#pla
lbl1 = Label(form)
lbl1.grid(column=1, row=8)
lbl1.configure(text="Tỉ lệ dự đoán đúng của SVM: "+'\n'
                           +"Accuracy: " + str(accuracy_score(y_test,y_test_pred)*100) + "%"+ '\n'
                           +"Precision: "+str(precision_score(y_test,y_test_pred ,zero_division=1)*100)+"%"+'\n'
                           +"Recall: "+str(recall_score(y_test,y_test_pred ,zero_division=1)*100)+"%"+'\n'
                           +"F1-score: "+str(f1_score(y_test,y_test_pred, zero_division=1)*100)+"%"+'\n')
def dudoanSVM():
    Age = textbox_Age.get()
    Experience = textbox_Experience.get()
    Income = textbox_Income.get()
    ZIPCode = textbox_ZIPCode.get()
    Family = textbox_Family.get()
    CCAvg = textbox_CCAvg.get()
    Education = textbox_Education.get()
    Mortgage = textbox_Mortgage.get()

    if((Age == '') or (Experience == '') or (Income == '') or (ZIPCode == '') or (Family == '')or (CCAvg == '') or (Education == '') or (Mortgage == '')):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = np.array([Age, Experience, Income,ZIPCode,Family,CCAvg ,Education ,Mortgage],dtype=np.float64).reshape(1, -1)
        y_kqua = regr.predict(X_dudoan)
        lbl.configure(text= y_kqua)
button_SVM = Button(form, text = 'Kết quả dự đoán theo SVM', command = dudoanSVM)
button_SVM.grid(row = 4, column = 3)
lbl = Label(form, text="...")
lbl.grid(column=4, row=4)

form.mainloop()
