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

# make numeric.
#data["Perioden_Title"] = pd.to_numeric(data["Perioden_Title"])
#data["gemiddelde verkoopprijs.GemiddeldeVerkoopprijs_1"] = pd.to_numeric(data["gemiddelde verkoopprijs.GemiddeldeVerkoopprijs_1"])
#data["GemiddeldeBevolking_2"] = pd.to_numeric(data["GemiddeldeBevolking_2"])

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

mergedData.to_csv("data.csv", sep=';')

print(mergedData)