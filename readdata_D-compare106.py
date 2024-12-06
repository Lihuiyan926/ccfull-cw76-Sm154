import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from collections import OrderedDict
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, NullFormatter)
import matplotlib.ticker
import scienceplots
lst = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

plt.style.use('science')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
#plt.rc('font', family='serif')
plt.rc('xtick', labelsize='10')
plt.rc('ytick', labelsize='10')
# 读取CSV文件
data = pd.read_csv('derivation.csv')
# data1=r"Derivation_exp.dat"
#fcal=r"derivation.dat"
#fcal =r"Dqel_Pb208.dat"
fcal =r"derivation_8.dat"
# gcal =r"derivation_8.dat"
# hcal =r"db_test.out"
# ical =r"derivation_line6.dat"
# jcal =r"derivation_line6-Qgg.dat"

E_03,DB_03 = np.genfromtxt(fcal, usecols=(0,1), unpack=True)
# E_04,DB_04 = np.genfromtxt(gcal, usecols=(0,1), unpack=True)
# E_05,DB_05 = np.genfromtxt(hcal, usecols=(0,1), unpack=True)
# E_06,DB_06 = np.genfromtxt(ical, usecols=(0,1), unpack=True)
# E_07,DB_07 = np.genfromtxt(jcal, usecols=(0,1), unpack=True)

Eeff = data['Sm154_Eeff']
Dqel = data['Sm154_D']
Derror=data['Sm154_DErr']


with plt.style.context(['science']):
    fig = plt.figure(figsize=(4,3))

    ax = fig.add_subplot(1, 1, 1)

    # plt.yscale('log')

# 绘制散点图
    ax.errorbar(x=Eeff, y=Dqel, yerr=Derror, fmt='o', markersize=3,
                label='Expt.')
    # 绘制CCFULL
    # ax.plot(E_03, DB_03, ls=lst['densely dashdotdotted'], marker=None,label=r'Without $3^-$($\beta_2=0.287, \beta_4 = 0.087$)')
    ax.plot(E_03, DB_03, ls=lst['densely dashdotdotted'], marker=None,
            label=r'Fit($\beta_2=0.272,\beta_3=0.037, \beta_4 = 0.106$)')
    # ax.plot(E_03, DB_03, ls=lst['densely dashdotdotted'], marker=None,
    #         label=r'W-Fit')
    # ax.plot(E_04, DB_04, '-.', label='8+')
    # ax.plot(E_05, DB_05, ':', label='Ccfull-sc2')
    # ax.plot(E_06, DB_06, '-', label='line6')
    # ax.plot(E_07, DB_07, '--', label='lin6-$Q_{gg}$')

# 添加标题和标签
    ax.set_xlabel('$E_{\mathrm{eff}}$ ')
    ax.set_ylabel('${\mathrm D}_{\mathrm{qel}}({\mathrm{MeV}}^{-1})$')

    # plt.xlim((0.85, 1.1))
    # plt.ylim((0.,0.15))

    # ax.axvline(x=1, linestyle='--', color='gray',linewidth=0.75,zorder=-1)

    legend=ax.legend( loc = 'upper left', prop={'size': 6},frameon=False,markerscale=1.2, scatterpoints =1 )
# 设置图例边框宽度
    legend.get_frame().set_linewidth(0.5)

    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2,hspace=0.0,wspace=0.25)
# 设置坐标轴的取值范围
#     plt.title('$^{16}{\mathrm{O}}+^{154}{\mathrm{Sm}}-Without\ 3^-\ Derivation$')
    plt.title('$^{16}{\mathrm{O}}+^{154}{\mathrm{Sm}}-All\ Fit\ Derivation$')
    plt.savefig('Sm154_D-compareallfit.png', dpi=800,bbox_inches = 'tight')