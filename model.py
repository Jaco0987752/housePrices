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
data["Jaar"] = pd.to_numeric(data["Jaar"])
data["gemiddelde verkoopprijs.GemiddeldeVerkoopprijs_1"] = pd.to_numeric(data["gemiddelde verkoopprijs.GemiddeldeVerkoopprijs_1"])
data["GemiddeldeBevolking_2"] = pd.to_numeric(data["GemiddeldeBevolking_2"])
data["Bevolkingsdichtheid_57"] = pd.to_numeric(data["Bevolkingsdichtheid_57"])
data["Woningdichtheid_93"] = pd.to_numeric(data["Woningdichtheid_93"])
data["Hypotheekrente"] = pd.to_numeric(data["Hypotheekrente"])
data["huishoudengrote"] = pd.to_numeric(data["GemiddeldeHuishoudensgrootte_89"])


data["loon"] = pd.to_numeric(data["loon"])


data["borrowcapacity"] = data["loon"] / data["Hypotheekrente"]
data["housepressure"] = data["Bevolkingsdichtheid_57"] / data["Woningdichtheid_93"]  

#data["borrowcapacity"] = data["loon"] / np.sqrt(data["Hypotheekrente"])


city = 'Rotterdam'

# Get the values of gemeente and sort them on year.
gemeente = data[data["RegioS_Title"] == city]
gemeente = gemeente.sort_values("Hypotheekrente", ascending=False)


#print(gemeente[["RegioS_Title", "Jaar"]])

gemeente.to_csv("test.csv", sep=';', decimal=",")


# Extract x and y values, so they can be plot.
x_values = gemeente[["borrowcapacity"]]  #"borrowcapacity", "GemiddeldeBevolking_2", "housepressure"
y_values = gemeente[["gemiddelde verkoopprijs.GemiddeldeVerkoopprijs_1"]]

X_test, X_train, y_test, y_train, = train_test_split(
    x_values, y_values, test_size=0.66, shuffle=False ,random_state=42)


# Do linear regression.
reg = ElasticNet(alpha=1.0,  l1_ratio=0.5)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
y_pred2 = reg.predict(X_train)


#y_pred = np.append(np.array(y_pred), np.array(y_train["gemiddelde verkoopprijs.GemiddeldeVerkoopprijs_1"])) #np.array(y_pred2)) #


x = np.array(gemeente["borrowcapacity"])
y = np.array(y_values["gemiddelde verkoopprijs.GemiddeldeVerkoopprijs_1"])

plt.title("housing prices " + city)
plt.ylabel("Average sales prices")
plt.xlabel("Hypotheekrente")
plt.plot(x, y_pred, color="blue", label="pred")
plt.plot(x, y, color="red", label="data")


print("the r squared of the regression:" + str(reg.score(X_test, y_test)))

