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
data["Hypotheekrente"] = pd.to_numeric(data["Hypotheekrente"])
data["inkomen"] = pd.to_numeric(data["inkomen"])
data["GemiddeldeVerkoopprijs"] = pd.to_numeric(data["gemiddelde verkoopprijs.GemiddeldeVerkoopprijs_1"])
# metadata
data["borrowcapacity"] = data["inkomen"] / data["Hypotheekrente"]

city = input("Enter city name like Amsterdam or Rotterdam: ").capitalize()

# Get the values of gemeente and sort them on year.
gemeente = data[data["RegioS_Title"] == city]
if(gemeente.empty):
    print("data for " + city + " is not found")
    exit()
income  =  float(input("Enter average year income in k-euro: "))
if(income <= 0):
    print("income should be above zero.")
    exit()
    
mortgageRate = float(input("Enter mortgage rate: "))
if(mortgageRate <= 0):
    print("mortgageRate should be above zero.")
    exit()
    
borrowCapacity = income / mortgageRate

gemeente = gemeente.sort_values("Jaar", ascending=False)

# Extract x and y values so the model can be trained on these.
x_values = np.array(gemeente[["borrowcapacity"]])  
y_values = np.array(gemeente[["GemiddeldeVerkoopprijs"]])

# Create a test train split.
x_test, x_train, y_test, y_train, = train_test_split(
    x_values, y_values, test_size=0.70, shuffle=True ,random_state=42)

# Do linear regression.
reg = ElasticNet(alpha=1.0,  l1_ratio=0.5)
reg.fit(x_train, y_train)
prediction = reg.predict([[borrowCapacity]])

print("The predicted average housprice for " + city + " is: " + str(prediction[0]))

x_test_ = np.array(x_test)
y_test_ = np.array(y_test)

print("the r squared of the regression:" + str(reg.score(x_test, y_test)))

