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
data = pd.read_csv('filterdata.csv')
# data = pd.read_csv('expdata_jia_deal.csv')
#读取CCFULL文件
fexp = r"ANGULAR.DAT"
# gexp=r"AngDis106-04.dat"
# hexp=r"AngDis106-05.dat"
# iexp=r"AngDis106-06.dat"
# jexp=r"AngDis106-07.dat"

Ecm_03, R_03 = np.loadtxt(fexp, usecols=(0,5), skiprows =(1), unpack=True)
# Ecm_04, R_04 = np.loadtxt(gexp, usecols=(0,8), skiprows =(1), unpack=True)
# #
# Ecm_05, R_05=np.loadtxt(hexp,usecols=(0,8), skiprows =(1), unpack=True)
# Ecm_06, R_06=np.loadtxt(iexp,usecols=(0,8), skiprows =(1), unpack=True)
#
# Ecm_07, R_07=np.loadtxt(jexp,usecols=(0,8), skiprows =(1), unpack=True)
#Pt_196Ecm,Pt_196R=np.loadtxt(kexp,usecols=(0,8), skiprows =(1), unpack=True)

#Pb_208Ecm,Pb_208R=np.loadtxt(lexp,usecols=(0,8), skiprows =(1), unpack=True)

Eeff = data['Sm154_Eeff']
Ratio = data['Sm154_Ratio']
Err=data['Sm154_Err']



with plt.style.context(['science']):
    fig = plt.figure(figsize=(4,3))

    ax = fig.add_subplot(1, 1, 1)

    # plt.yscale('log')
# 绘制散点图
#     ax.scatter(Sm152_Eeff, Sm152_Ratio,marker='o',s=9,label='Expt.')
    ax.errorbar(x=Eeff, y=Ratio, yerr=Err, fmt='o', markersize=3,
                label='Expt.')
# 绘制CCFULL
#     ax.plot(Ecm_03, R_03,ls=lst['densely dashdotdotted'], marker=None,label=r'Without $3^-$($\beta_2=0.287, \beta_4 = 0.087$)')
    ax.plot(Ecm_03, R_03, ls=lst['densely dashdotdotted'], marker=None,
            label=r'Fit($\beta_2=0.272,\beta_3=0.037, \beta_4 = 0.106$)')
#     ax.plot(Ecm_03, R_03, ls=lst['densely dashdotdotted'], marker=None,
#             label=r'W-fit')
    # ax.plot(Ecm_04, R_04,'-.', label='Ratio$>0.4$')
    # ax.plot(Ecm_05, R_05,':', label='Ratio$>0.5$')
    # ax.plot(Ecm_06, R_06, '-',label='Ratio$>0.6$')
    # ax.plot(Ecm_07, R_07,'--', label='Ratio$>0.7$')

# 添加标题和标签
    ax.set_xlabel('$E_{\mathrm{eff}} $ ')
    ax.set_ylabel('$\sigma_{\mathrm{qel}}/\sigma_{\mathrm{Ru}}$')

    # plt.xlim((30, 80.))
    # plt.ylim((0.07, 1.2))
    #plt.ylim((0.4,1.2))

    ax.axhline(y=0.5, linestyle='--', color='gray', linewidth=0.75)

    legend=ax.legend( loc = (0.04,0.03), prop={'size': 8},frameon=False,markerscale=1.2, scatterpoints =1 )
# 设置图例边框宽度
    legend.get_frame().set_linewidth(0.5)

    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2,hspace=0.0,wspace=0.25)
# 设置坐标轴的取值范围
#     plt.title('$^{16}{\mathrm{O}}+^{154}{\mathrm{Sm}}-Without\ 3^-\ Ratio\ compare$')
    plt.title('$^{16}{\mathrm{O}}+^{154}{\mathrm{Sm}}-All Fit\ Ratio\ compare$')
#     plt.title('$^{16}{\mathrm{O}}+^{154}{\mathrm{Sm}}-R_T=1.16\ Ratio\ compare$')
    plt.savefig('Sm154_sigma-compareallfit.png', dpi=800, bbox_inches='tight')
    #plt.title('$^{16}{\mathrm{O}}+^{152,154}{\mathrm{Sm}},^{184,186}{\mathrm{W}},^{194,196}{\mathrm{Pt}},^{208}{\mathrm{Pb}}$')
    #plt.savefig('16Osigma.png', dpi=800,bbox_inches = 'tight')