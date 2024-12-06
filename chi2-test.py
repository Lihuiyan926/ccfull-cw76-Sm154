import subprocess
import os
import numpy as np
import matplotlib
# matplotlib.use('Agg') #调用后端
import matplotlib.pyplot as plt
# import sys
import platform
from iminuit import Minuit
from iminuit.util import propagate
# from iminuit.cost import LeastSquares
from scipy import interpolate #插值操作
import pandas as pd
import uuid
import shutil
# import time
from joblib import Parallel, delayed
from scipy.optimize import differential_evolution

data = pd.read_csv('filterdata03.csv')
Eeff = data['W186_Eeff']
Ratio = data['W186_Ratio']
Error = data['W186_Err']

expx=Eeff
expy=Ratio
# expy=Dqel
expyerr=Error
expn=len(expx)


fresult = "ANGULAR.DAT"
# fresult = "db2.out"
global fcc_x, fcc_y
# AngDis 读取
fcc_x, fcc_y = np.loadtxt(fresult, usecols=(0, 5), unpack=True)
# Dqel 读取
# fcc_x, fcc_y = np.genfromtxt(fresult, usecols=(0, 1), skip_header=1,skip_footer=1, unpack=True)

finterpolate = interpolate.interp1d(fcc_x, fcc_y, kind='cubic')  # 三次样条插值
fcc_exp = finterpolate(expx)

fcn =np.sum( (fcc_exp-expy)**2 /expyerr**2)
print(fcn)