import pandas as pd
import numpy as np

from joblib import dump
import os

import matplotlib.pyplot as plt
import seaborn as sns
import klib

from sklearn.model_selection import train_test_split, KFold,cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()
labeller = LabelEncoder()

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

RF_Clas = RandomForestClassifier()
DT_Clas = DecisionTreeClassifier()
svc = SVC()

from sklearn.metrics import classification_report,accuracy_score