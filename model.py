# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 12:57:44 2022

@author: Jaco
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split

data = pd.read_csv("data.csv", sep=';')

# Make numeric.
data["Perioden_Title"] = pd.to_numeric(data["Perioden_Title"])
data["gemiddelde verkoopprijs.GemiddeldeVerkoopprijs_1"] = pd.to_numeric(data["gemiddelde verkoopprijs.GemiddeldeVerkoopprijs_1"])
data["GemiddeldeBevolking_2"] = pd.to_numeric(data["GemiddeldeBevolking_2"])
data["Bevolkingsdichtheid_57"] = pd.to_numeric(data["Bevolkingsdichtheid_57"])
data["Woningdichtheid_93"] = pd.to_numeric(data["Woningdichtheid_93"])
data["Hypotheekrente"] = pd.to_numeric(data["Hypotheekrente"])

data["borrowcapacity"] =1 / data["Hypotheekrente"]

# Create meta data.
data["housepressure"] = data["Bevolkingsdichtheid_57"] / data["Woningdichtheid_93"]  

city = 'Rotterdam'

# Get the values of gemeente and sort them on year.
gemeente = data[data["RegioS_Title"] == city]
gemeente = gemeente.sort_values("Perioden_Title", ascending=False)
#print(gemeente[["RegioS_Title", "Perioden_Title"]])

# Extract x and y values, so they can be plot.
x_values = gemeente[["borrowcapacity","Perioden_Title", "GemiddeldeBevolking_2", "housepressure"]]
y_values = gemeente[["gemiddelde verkoopprijs.GemiddeldeVerkoopprijs_1"]]

X_test, X_train, y_test, y_train, = train_test_split(
    x_values, y_values, test_size=0.75, shuffle=False ,random_state=42)

x_regression_values = [["borrowcapacity","GemiddeldeBevolking_2", "housepressure"]]

# Do linear regression.
reg = ElasticNet(alpha=1.0,  l1_ratio=0.5)
reg.fit(X_train[["borrowcapacity","GemiddeldeBevolking_2", "housepressure"]], y_train)
y_pred = reg.predict(X_test[["borrowcapacity","GemiddeldeBevolking_2", "housepressure"]])


x = np.array(x_values["Perioden_Title"])
y = np.array(y_values["gemiddelde verkoopprijs.GemiddeldeVerkoopprijs_1"])

# Concatenate the unsorted arrays()
unsorted_x = np.append(np.array(X_train["Perioden_Title"]), np.array(X_test["Perioden_Title"])) 
unsorted_y = np.append(np.array(y_train["gemiddelde verkoopprijs.GemiddeldeVerkoopprijs_1"]), np.array(y_pred))


# Sort list on year in order to be able to plot.
items = dict(zip(unsorted_x,unsorted_y)).items()
res = sorted(items)
x2, y2 = zip(*res)

plt.title("housing prices " + city)
plt.ylabel("Average sales prices")
plt.xlabel("years")
plt.plot(x2, y2, color="blue", label="pred")
plt.plot(x, y, color="red", label="data")


print("the r squared of the regression:" + str(reg.score(X_test[["borrowcapacity","GemiddeldeBevolking_2", "housepressure"]], y_test)))

