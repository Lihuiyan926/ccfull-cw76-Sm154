import csv
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import platform
from iminuit import Minuit
from iminuit.util import propagate
from matplotlib import gridspec
# from iminuit.cost import LeastSquares
from scipy import interpolate #插值操作
import pandas as pd
import shutil
import uuid
import time
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.stats import norm

# data = pd.read_csv('expdata_jia.csv')
data = pd.read_csv('filterdata07.csv')

Eeff = data['Sm154_Eeff']
Ratio = data['Sm154_Ratio']
Error = data['Sm154_Err']

expx=Eeff
expy=Ratio
# expy=Dqel
expyerr=Error
expn=len(expx)

fchain = 'chi2-106-37.csv'
if os.path.exists(fchain):  # if existed
       os.remove(fchain)

def model(expx,par):
    beta_200,beta_400 = par[0], par[1]

    uuid_str = uuid.uuid4().hex
    tmp_file_name = 'tmpfile_%s' % uuid_str
    fn = tmp_file_name
    # print(fn)
    if not os.path.exists(fn):
        os.makedirs(fn)
    # os.mkdir('Test')
    # 切换到文件夹下
    os.chdir(fn)

    shutil.copyfile('../ccfull-sc.inp', './ccfull-sc.inp')
    shutil.copyfile('../TEST2.INP', './TEST2.INP')
    shutil.copyfile('../a', './a')

    with open("ccfull-sc.inp", "r") as file:
        lines = file.readlines()

        # 修改参数
    linereplace1 = 3
    beta_20 = beta_200
    beta_40 = beta_400

    # 修改特定行的内容
    lines[linereplace1 - 1] = '0.082,' + str(beta_20) + ',' + str(beta_40) + ',3\n'

    # 将修改后的内容写回文件
    with open("ccfull-sc.inp", "w") as file:
        file.writelines(lines)
    # print(theta)

    sysstr = platform.system()
    if (sysstr == "Windows"):
        main = "ccfull-sc2.exe"
        # main2 = "derivation.exe"
    elif (sysstr == "Linux"):
        main = "./a"
        # main2 = "./derivation"
        os.system('chmod 777 a')
        # os.system('chmod 777 derivation')


    # if os.path.exists(main) and os.path.exists(main2):
    #    os.system(main)
    #    os.system(main2)
    if os.path.exists(main):
       os.system(main)

    fresult = "ANGULAR.DAT"
    if os.path.exists(fresult):
        print('ANGULAR finish')
    # fresult = "db2.out"
    global fcc_x, fcc_y
    # AngDis 读取
    fcc_x, fcc_y = np.loadtxt(fresult, usecols=(0, 5), unpack=True)
    # Dqel 读取
    # fcc_x, fcc_y = np.genfromtxt(fresult, usecols=(0, 1), skip_header=1,skip_footer=1, unpack=True)

    finterpolate = interpolate.interp1d(fcc_x, fcc_y, kind='cubic')  # 三次样条插值
    fcc_exp = finterpolate(expx)
    
    os.chdir('..')
    shutil.rmtree(fn)


    return fcc_exp

# total range
# beta_2_range = np.arange(0, 0.41, 0.01)
# beta_4_range = np.arange(-0.2, 0.21, 0.01)


# # cut range
# beta_2_range = np.arange(0.275, 0.315, 0.001)
# beta_4_range = np.arange(0.01, 0.06, 0.001)
# cut range
beta_2_range = np.arange(0.23, 0.28, 0.001)
beta_4_range = np.arange(0.02, 0.08, 0.001)
def calculate_chisq(beta_2, beta_4):
    par = [beta_2, beta_4]
    ftheo = model(expx, par)
    chisq = np.sum((ftheo - expy) ** 2 / expyerr ** 2)

    with open(fchain, 'a', newline='') as csvfile:
        fieldnames = ['beta_2', 'beta_4', 'chi-square']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:  # 如果文件为空，写入表头
            writer.writeheader()
        writer.writerow({'beta_2': beta_2, 'beta_4': beta_4, 'chi-square': chisq})

    return par, chisq


# 并行计算 chisq_values
num_cores = -1  # 使用所有可用的 CPU 核心
results = Parallel(n_jobs=num_cores)(delayed(calculate_chisq)(beta_2, beta_4) for beta_2 in beta_2_range for beta_4 in beta_4_range)

# 将结果整理成数组形式
chisq_values_parallel = np.array([result[1] for result in results]).reshape(len(beta_2_range), len(beta_4_range))

# # 绘制等高线图
# beta_4_grid ,beta_2_grid, = np.meshgrid( beta_4_range,beta_2_range)
# plt.contourf(beta_2_grid, beta_4_grid, chisq_values_parallel, cmap='RdYlBu', levels=np.linspace(0,400,100),extend='neither')
# plt.colorbar(label='Chi-square Value')
# C = plt.contour(beta_2_grid, beta_4_grid, chisq_values_parallel,alpha=0.8, colors='black', levels=np.linspace(50,200,4),extend='both')
# plt.clabel(C, inline=True, fontsize=8)  # 添加标签，使用内置显示和设置字体大小
#
# C1 = plt.contour(beta_2_grid, beta_4_grid,chisq_values_parallel,alpha=0.8, colors='black', levels=np.linspace(10,25,2),extend='both')
# plt.clabel(C1, inline=True, fontsize=6)  # 添加标签，使用内置显示和设置字体大小
# plt.xlabel('Beta_2')
# plt.ylabel('Beta_4')
# plt.title('$^{16}{\mathrm{O}}+^{154}{\mathrm{Sm}}$ Excitation function Chi-square Contour Plot (Parallel)')
# plt.savefig('Sm154_106_07.png')
# # plt.savefig('Sm154_cut_07.png')
# plt.show()

# 找到最小卡方值及其对应参数
min_chi2 = np.min(chisq_values_parallel)
best_params = np.unravel_index(np.argmin(chisq_values_parallel), chisq_values_parallel.shape)
best_beta_2 = beta_2_range[best_params[0]]
best_beta_4 = beta_4_range[best_params[1]]

print(f"最小卡方值: {min_chi2} 在参数 beta_2: {best_beta_2}, beta_4: {best_beta_4}")

# 计算单参数的概率密度
# prob_beta_2 = np.exp(-chisq_values_parallel / np.max(chisq_values_parallel))
# prob_beta_4 = np.exp(-chisq_values_parallel / np.max(chisq_values_parallel))

# 构建后验分布
posterior_values = np.exp(-0.5 * (chisq_values_parallel - min_chi2))  # 使用高斯近似
posterior_values /= np.sum(posterior_values)  # 归一化

# 提取后验分布
posterior_beta_2 = np.sum(posterior_values, axis=1)  # 对 beta_4 进行求和
posterior_beta_4 = np.sum(posterior_values, axis=0)  # 对 beta_2 进行求和


def calculate_uncertainty_and_ci(posterior, parameter_range, confidence_level=0.95):
    mean = np.sum(posterior * parameter_range)  # 计算后验均值
    variance = np.sum(posterior * (parameter_range - mean) ** 2)  # 计算方差
    std_dev = np.sqrt(variance)  # 计算标准差

    # 计算置信区间
    z_score = norm.ppf((1 + confidence_level) / 2)  # 获取z-score
    ci_lower = mean - z_score * std_dev  # 下限
    ci_upper = mean + z_score * std_dev  # 上限

    return mean, std_dev, z_score, z_score * std_dev


# 计算 beta_2 的均值、标准差和置信区间
mean_beta_2, uncertainty_beta_2, z_score_beta_2, z_dev_beta_2 = calculate_uncertainty_and_ci(posterior_beta_2, beta_2_range)
ci_lower_beta_2 = mean_beta_2 - z_dev_beta_2  # 下限
ci_upper_beta_2 = mean_beta_2 + z_dev_beta_2  # 上限
beta_2_condition = (beta_2_range >= ci_lower_beta_2) & (beta_2_range <= ci_upper_beta_2)
# 计算 beta_4 的均值、标准差和置信区间
mean_beta_4, uncertainty_beta_4, z_score_beta_4, z_dev_beta_4 = calculate_uncertainty_and_ci(posterior_beta_4, beta_4_range)
ci_lower_beta_4 = mean_beta_4 - z_dev_beta_4  # 下限
ci_upper_beta_4 = mean_beta_4 + z_dev_beta_4  # 上限
beta_4_condition = (beta_4_range >= ci_lower_beta_4) & (beta_4_range <= ci_upper_beta_4)

# 创建画布和网格
fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2, 2)

# 绘制左下角的卡方分布图
ax_main = plt.subplot(gs[1, 0])
beta_4_grid ,beta_2_grid, = np.meshgrid( beta_4_range,beta_2_range)
# ax_main.contourf(beta_2_grid, beta_4_grid, chisq_values_parallel, cmap='RdYlBu', levels=np.linspace(0,400,100),extend='neither')
# # ax_main.colorbar(label='Chi-square Value')
# C = ax_main.contour(beta_2_grid, beta_4_grid, chisq_values_parallel,alpha=0.8, colors='black', levels=np.linspace(50,200,4),extend='both')
# ax_main.clabel(C, inline=True, fontsize=8)  # 添加标签，使用内置显示和设置字体大小
#
# C1 = ax_main.contour(beta_2_grid, beta_4_grid,chisq_values_parallel,alpha=0.8, colors='black', levels=np.linspace(7,30,2),extend='both')

# cut
ax_main.contourf(beta_2_grid, beta_4_grid, chisq_values_parallel, cmap='RdYlBu', levels=np.linspace(0,40,100),extend='neither')
# ax_main.colorbar(label='Chi-square Value')
C = ax_main.contour(beta_2_grid, beta_4_grid, chisq_values_parallel,alpha=0.8, colors='black', levels=np.linspace(8,20,2),extend='both')
ax_main.clabel(C, inline=True, fontsize=8)  # 添加标签，使用内置显示和设置字体大小

C1 = ax_main.contour(beta_2_grid, beta_4_grid,chisq_values_parallel,alpha=0.8, colors='black', levels=np.linspace(0,5,2),extend='both')
ax_main.clabel(C1, inline=True, fontsize=6)  # 添加标签，使用内置显示和设置字体大小
ax_main.set_xlabel(r'$\beta_2$')
ax_main.set_ylabel(r'$\beta_4$')
# ax_main.set_title('$^{16}{\mathrm{O}}+^{186}{\mathrm{W}}$ Excitation function Chi-square Contour Plot (Parallel)')
# # plt.savefig('W186_106_37.png')

# 绘制右上角的beta_2概率密度
ax_beta_2 = plt.subplot(gs[0, 0])
plt.plot(beta_2_range, posterior_beta_2, color='blue')
# 更新 y 轴范围
ax_beta_2.set_ylim(bottom=0)  # 确保 y 轴从 0 开始
# 获取当前的 y 轴范围
y_min, y_max = ax_beta_2.get_ylim()
ax_beta_2.fill_betweenx([y_min, y_max], ci_lower_beta_2, ci_upper_beta_2,
                         color='lightgreen', alpha=0.5)
ax_beta_2.set_ylabel('Probability Density')
ax_beta_2.set_xlim(np.min(beta_2_range), np.max(beta_2_range))  # 设置 x 轴范围
# 隐藏 x 轴的刻度标签，但保留刻度线
ax_beta_2.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

# 在 beta_2 概率密度最高处画竖线
max_index_beta_2 = np.argmax(posterior_beta_2)  # 找到 beta_2 概率密度最大值的索引
max_value_beta_2 = beta_2_range[max_index_beta_2]  # 取对应的 beta_2 值
ax_beta_2.axvline(max_value_beta_2, color='red', linestyle='--')  # 画竖线
ax_beta_2.axvline(ci_lower_beta_2, color='grey', linestyle='--')
ax_beta_2.axvline(ci_upper_beta_2, color='grey', linestyle='--')
# 打印 beta_2 最大概率密度对应的坐标
print(f'Max Beta_2: {max_value_beta_2}')
ax_beta_2.set_title(r'$\beta_2 = {:.3f} \pm {:.3f}$'.format(max_value_beta_2, z_dev_beta_2))

# 绘制右下角的beta_4概率密度
ax_beta_4 = plt.subplot(gs[1, 1])
plt.plot(beta_4_range, posterior_beta_4, color='green')

# 更新 y 轴范围
ax_beta_4.set_ylim(bottom=0)  # 确保 y 轴从 0 开始
# 获取当前的 y 轴范围
y_min, y_max = ax_beta_4.get_ylim()
ax_beta_4.fill_betweenx([y_min, y_max], ci_lower_beta_4, ci_upper_beta_4,
                         color='lightblue', alpha=0.5)
# ax_beta_4.set_title(r'$\beta_4$ Probability Density')
ax_beta_4.set_ylabel('Probability Density')
ax_beta_4.yaxis.set_label_position("right")  # 将 y 轴标签移动到右侧
ax_beta_4.yaxis.tick_right()
ax_beta_4.set_xlim(np.min(beta_4_range), np.max(beta_4_range))

# 在 beta_4 概率密度最高处画竖线
max_index_beta_4 = np.argmax(posterior_beta_4)  # 找到概率密度最大值的索引
max_value_beta_4 = beta_4_range[max_index_beta_4]  # 取对应的 beta_4 值
ax_beta_4.axvline(max_value_beta_4, color='red', linestyle='--')  # 画竖线
ax_beta_4.axvline(ci_lower_beta_4, color='grey', linestyle='--')
ax_beta_4.axvline(ci_upper_beta_4, color='grey', linestyle='--')
# 打印 beta_4 最大概率密度对应的坐标
print(f'Max Beta_4: {max_value_beta_4}')
ax_beta_4.set_title(r'$\beta_4 = {:.3f} \pm {:.3f}$'.format(max_value_beta_4, z_dev_beta_4))

# 右上角保留为空或添加其他内容
ax_empty = plt.subplot(gs[0, 1])
ax_empty.axis('off')  # 隐藏右上角的空白区域

plt.tight_layout()
plt.savefig('Sm154-test-37-95.png')
plt.savefig('Sm154-test-37-95.eps')
plt.show()