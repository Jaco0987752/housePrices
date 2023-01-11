# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 16:29:23 2022

@author: Jaco
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("Complete_dataset.csv", sep=';')

data2 = pd.read_csv("ds_gemeente_compleet.csv", sep=";", encoding='latin-1', decimal=".")
mergedData = pd.merge(data, data2, how='left', left_on=["RegioS","Perioden"], right_on=["RegioS", "Perioden"])

mergedData = mergedData[[
    "Perioden_Title", 
    "gemiddelde verkoopprijs.GemiddeldeVerkoopprijs_1",
    "GemiddeldeBevolking_2", 
    "RegioS_Title", 
    "Bevolkingsdichtheid_57", 
    "Woningdichtheid_93", 
    "Woningen_97",
    "GemiddeldeHuishoudensgrootte_89"]]


morgateData = pd.read_csv("Hypotheekrente.csv", sep=";")
wageData = pd.read_csv("income.csv", sep=";", decimal=",")[["jaar", "inkomen"]]

withMorgateData = pd.merge(mergedData, morgateData, how='left', left_on=["Perioden_Title"], right_on=["Jaar"])
withWageData = pd.merge(withMorgateData,  wageData, how='left', left_on=["Jaar"], right_on=["jaar"])
withWageData = withWageData.interpolate() # get a value for income in 2021 based on extrapolation.

print(withWageData[["inkomen", "Jaar", "Hypotheekrente"]])

withWageData = withWageData.replace(np.NaN, "0.0")
withWageData = withWageData.replace(".", "0.0")

withWageData.to_csv("data.csv", sep=';')
#print(withWageData["GemiddeldeHuishoudensgrootte_89"])
