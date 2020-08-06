from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy.io.arff import loadarff
import pandas as pd
import numpy as np
from tkinter import *
from tkinter import messagebox
import random

data = loadarff("./hepatitis.arff")
df = pd.DataFrame(data[0])

### DECODE BYTES TO STRING
for i in range(1,13):
    df.iloc[:, i] = df.iloc[:, i].apply(lambda x : x.decode('utf-8'), 1)

df.iloc[:, 18] = df.iloc[:, 18].apply(lambda x : x.decode('utf-8'), 1)
df.iloc[:, 19] = df.iloc[:, 19].apply(lambda x : x.decode('utf-8'), 1)

### REPLACE ? WITH OTHER STRING
def str_repl(str, weights = None):
    if str == '?':
        return random.choices(["yes", "no"], weights = weights)[0]
    else:
        return str

for i in range(2,13):
        row = df.iloc[:, i]
        y = sum(row.str.count("yes"))
        n = sum(row.str.count("no"))
        weights = [y/(y+n), n/(y+n)]
        df.iloc[:, i] = df.iloc[:, i].apply(lambda x : str_repl(x, weights = weights), 1)

row = df.iloc[:, 18]
y = sum(row.str.count("yes"))
n = sum(row.str.count("no"))
weights = [y/(y+n), n/(y+n)]
df.iloc[:, 18] = df.iloc[:, 18].apply(lambda x : str_repl(x, weights = weights), 1)

### CLEAR NAN
df.iloc[:, 0] = df.iloc[:,0].fillna(0)
for i in range(13,18):
    df.iloc[:, i] = df.iloc[:,i].fillna(0)

### GUI STARTS
window = Tk()

window.title("Welcome to this DSS")

window.geometry('300x800')
texts = ["Age", "Sex", "Steroid", "Antivirals", "Fatigue", "Malaise",
         "Anorexia", "Liver Big", " Liver Firm", "Spleen Palpable",
        "Spiders", "Ascites", "Varices", "Bilirubin", "Alk Phosphate",
        "SGOT", "Albumin", "Protime", "Histology"]
vals = []
for tex in texts:
    lbl = Label(window, text=tex)
    lbl.grid(column=0, row=texts.index(tex))
    vals.append(Entry(window, width=10))
    vals[-1].grid(column=1, row=texts.index(tex))


def clicked():
    vals = list(map(lambda x : x.get(), vals))
    vals.append('LIVE')
 #    vals = ['30','male','no','no','no', 'no','no','no', 'no', 'no','no', 'no',
 # 'no',
 # '1',
 # '85',
 # '18',
 # '4',
 # '54',
 # 'no', 'LIVE']

    df2 = df.append(pd.DataFrame(np.array(vals).reshape(1,-1), columns = df.columns)).reset_index(drop=True)

    ### TRANSFORM DATA
    lb = preprocessing.LabelBinarizer()
    for i in range(1,13):
            df2.iloc[:, i] = lb.fit_transform(df2.iloc[:, i])

    df2.iloc[:, 18] = lb.fit_transform(df2.iloc[:, 18])
    df2.iloc[:, 19] = lb.fit_transform(df2.iloc[:, 19])

    df2, ex = df2.iloc[0:-1, :], df2.iloc[-1,0:-1]


    ### NAIVE BAYES
    X = df2.loc[:, "AGE":"HISTOLOGY"]
    y = df2.loc[:, "Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = GaussianNB()
    clf.fit(X_train, y_train)

    results = clf.predict(np.array(list(map(float, ex.to_list()))).reshape(1,-1))
    if results == 1:
        result = "Live"
    else:
        result = "Die"

    messagebox.showinfo('Hepatitis Prediction:', f"{result}")


btn = Button(window, text="Submit", command=clicked)
btn.grid(column=1, row=20)

window.mainloop()
