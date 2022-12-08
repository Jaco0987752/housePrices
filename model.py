# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 12:57:44 2022

@author: Jaco
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("Complete_dataset.csv", sep=';')

# make numeric.
data["Perioden_Title"] = pd.to_numeric(data["Perioden_Title"])
data["gemiddelde verkoopprijs.GemiddeldeVerkoopprijs_1"] = pd.to_numeric(data["gemiddelde verkoopprijs.GemiddeldeVerkoopprijs_1"])

# Get the values of amsterdam and sort them on year.
amsterdam = data[data["RegioS_Title"] == 'Amsterdam']
amsterdam = amsterdam.sort_values("Perioden_Title", ascending=False)
#print(amsterdam[["RegioS_Title", "Perioden_Title"]])

# extract x and y values, so they can be plot.
x_values = amsterdam[["Perioden_Title"]]
y_values = amsterdam[["gemiddelde verkoopprijs.GemiddeldeVerkoopprijs_1"]]

# 
reg = LinearRegression().fit(x_values, y_values)
y_pred = reg.predict(x_values)

x = np.array(x_values["Perioden_Title"])
y = np.array(y_values["gemiddelde verkoopprijs.GemiddeldeVerkoopprijs_1"])

#print(x,y)
plt.plot(x, y, color="red", label="data")
plt.plot(x, y_pred, color="blue", label="pred")


#print(reg.score(x_values, Y_values))

