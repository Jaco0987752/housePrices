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

data2 = pd.read_csv("ds_gemeente_compleet.csv", sep=";", encoding='latin-1')
mergedData = pd.merge(data, data2, how='left', left_on=["RegioS","Perioden"], right_on=["RegioS", "Perioden"])

mergedData = mergedData[[
    "Perioden_Title", 
    "gemiddelde verkoopprijs.GemiddeldeVerkoopprijs_1",
    "GemiddeldeBevolking_2", 
    "RegioS_Title", 
    "Bevolkingsdichtheid_57", 
    "Woningdichtheid_93", 
    "Woningen_97"]]


morgateData = pd.read_csv("Hypotheekrente.csv", sep=";")
withMorgateData = pd.merge(mergedData, morgateData, how='left', left_on=["Perioden_Title"], right_on=["Jaar"])
withMorgateData.to_csv("data.csv", sep=';')
print(mergedData)