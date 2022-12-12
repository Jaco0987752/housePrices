# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 12:57:44 2022

@author: Jaco
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data.csv", sep=';')

# Make numeric.
data["Perioden_Title"] = pd.to_numeric(data["Perioden_Title"])
data["gemiddelde verkoopprijs.GemiddeldeVerkoopprijs_1"] = pd.to_numeric(data["gemiddelde verkoopprijs.GemiddeldeVerkoopprijs_1"])
data["GemiddeldeBevolking_2"] = pd.to_numeric(data["GemiddeldeBevolking_2"])
data["Bevolkingsdichtheid_57"] = pd.to_numeric(data["Bevolkingsdichtheid_57"])
data["Woningdichtheid_93"] = pd.to_numeric(data["Woningdichtheid_93"])

# Create meta data.
data["housepressure"] = data["Bevolkingsdichtheid_57"] / data["Woningdichtheid_93"]  


# Get the values of gemeente and sort them on year.
gemeente = data[data["RegioS_Title"] == 'Alblasserdam']
gemeente = gemeente.sort_values("Perioden_Title", ascending=False)
#print(gemeente[["RegioS_Title", "Perioden_Title"]])

# Extract x and y values, so they can be plot.
x_values = gemeente[["Perioden_Title", "GemiddeldeBevolking_2", "housepressure"]]
y_values = gemeente[["gemiddelde verkoopprijs.GemiddeldeVerkoopprijs_1"]]

# Do linear regression.
reg = LinearRegression().fit(x_values, y_values)
y_pred = reg.predict(x_values)

x = np.array(x_values["Perioden_Title"])
y = np.array(y_values["gemiddelde verkoopprijs.GemiddeldeVerkoopprijs_1"])

# print(x,y)
plt.plot(x, y, color="red", label="data")
plt.plot(x, y_pred, color="blue", label="pred")


print("the r squared of the regression:" + str(reg.score(x_values, y_values)))

