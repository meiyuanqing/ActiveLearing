#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Project : ActiveLearing
@File : lineplot.py
@Author : Yuanqing Mei
@Time : 2022/11/24 23:21
@Homepage: http://github.com/meiyuanqing
@Email: dg1533019@smail.nju.edu.cn

为画折线图做准备
"""
import matplotlib.pyplot as plt
import numpy as np

x_axis_data = [1, 2, 3, 4, 5, 6, 7]
y_axis_data1 = [68.72, 69.17, 69.26, 69.63, 69.35, 70.3, 66.8]
y_axis_data2 = [71, 73, 52, 66, 74, 82, 71]
y_axis_data3 = [82, 83, 82, 76, 84, 92, 81]
y_axis_data4 = [60, 89, 86, 70, 80, 99, 88]

# 这个plot后面可以加一个三位的字符串，分别是 “颜色”, “点型”, “线型”，
# 线性有：
# linestyle=['-','--','-.',':']
# 点型有：
# marker=['.',',','o','v','^','<','>','1','2','3','4','s','p','*','h','H','+','x','D','d','|','_','.',',']
# color=['b','g','r','c','m','y','k','w']
plt.plot(x_axis_data, y_axis_data1, 'b*--', alpha=0.5, linewidth=1, label='acc')
plt.plot(x_axis_data, y_axis_data2, 'rs--', alpha=0.5, linewidth=1, label='acc')
plt.plot(x_axis_data, y_axis_data3, 'go--', alpha=0.5, linewidth=1, label='acc')
plt.plot(x_axis_data, y_axis_data4, 'kv--', alpha=0.5, linewidth=1, label='acc')

# 显示label
plt.legend()
plt.xlabel('time')
plt.ylabel('number')

# 仅设置y轴坐标范围
# plt.ylim(-1, 1)
plt.show()