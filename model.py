# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 12:57:44 2022

@author: Jaco
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

data = pd.read_csv("data.csv", sep=';')

# Make numeric.
data["Jaar"] = pd.to_numeric(data["Jaar"])
data["GemiddeldeVerkoopprijs"] = pd.to_numeric(data["gemiddelde verkoopprijs.GemiddeldeVerkoopprijs_1"])
data["GemiddeldeBevolking_2"] = pd.to_numeric(data["GemiddeldeBevolking_2"])
data["Bevolkingsdichtheid_57"] = pd.to_numeric(data["Bevolkingsdichtheid_57"])
data["Woningdichtheid_93"] = pd.to_numeric(data["Woningdichtheid_93"])
data["Hypotheekrente"] = pd.to_numeric(data["Hypotheekrente"])
data["huishoudengrote"] = pd.to_numeric(data["GemiddeldeHuishoudensgrootte_89"])


data["inkomen"] = pd.to_numeric(data["inkomen"])
#print(data[["inkomen", "Jaar", "Hypotheekrente"]])

data["borrowcapacity"] = data["inkomen"] / data["Hypotheekrente"]
data["housepressure"] = data["Bevolkingsdichtheid_57"] / data["Woningdichtheid_93"]  

#data["borrowcapacity"] = data["loon"] / np.sqrt(data["Hypotheekrente"])


city = 'Rotterdam' #'Rotterdam'#'Amsterdam'#

# Get the values of gemeente and sort them on year.
gemeente = data[data["RegioS_Title"] == city]
gemeente = gemeente.sort_values("Jaar", ascending=False)


#print(gemeente[["RegioS_Title", "Jaar"]])

gemeente.to_csv("test.csv", sep=';', decimal=",")


# Extract x and y values, so they can be plot.
x_values = gemeente[["borrowcapacity"]]  #"borrowcapacity", "GemiddeldeBevolking_2", "housepressure"
y_values = gemeente[["GemiddeldeVerkoopprijs"]]

x_test, x_train, y_test, y_train, = train_test_split(
    x_values, y_values, test_size=0.70, shuffle=True ,random_state=42)


# Do linear regression.
reg = Ridge(alpha=15.0)
reg.fit(x_train, y_train)
y_pred = reg.predict(x_values)
#y_pred2 = reg.predict(X_train)

x_train = np.array(x_train)
y_train = np.array(y_train)

#print(y_train)

x_test_ = np.array(x_test)
y_test_ = np.array(y_test)

#print(x_test)

plt.title("housing prices " + city)
plt.ylabel("Average sales prices")
plt.xlabel("borrow capacity")
plt.plot(np.array(x_values), y_pred, color="blue", label="regressionLine")
plt.scatter(x_train, y_train, color="red", label="training_data")
plt.scatter(x_test_, y_test_, color="yellow", label="test_data")

plt.legend()


print("the r squared of the regression:" + str(reg.score(x_test, y_test)))

