#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Project : ActiveLearing
@File : RQ1_T_median.py
@Author : Yuanqing Mei
@Time : 2022/11/15 21:15
@Homepage: http://github.com/meiyuanqing
@Email: dg1533019@smail.nju.edu.cn

无监督方法： 用当前版本上所有度量与其中位数阈值比较，得到评分后，再加上SLOC度量的倒数，得到当前版本上所有模块的综合评分，
           取50%的综合评分中位数阈值预测各模块缺陷倾向。
"""


# 分别应用pred_20_score的投票得分，用0.1,0.2,...,0.9 九种对应百分比分位数作为阈值分别计算分类性能
def do_predict(df_test):
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix
    f1_1, gm_1, bpp_1, f1_2, gm_2, bpp_2, f1_3, gm_3, bpp_3, f1_4, gm_4, bpp_4, f1_5, gm_5, bpp_5, \
    f1_6, gm_6, bpp_6, f1_7, gm_7, bpp_7, f1_8, gm_8, bpp_8, f1_9, gm_9, bpp_9 = \
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    for i in range(9):

        df_predict = df_test.copy()

        for j in range(len(df_predict)):
            if df_predict.loc[j, 'pred_20_score'] >= df_predict.pred_20_score.quantile(0.1 * (i + 1)):
                df_predict.loc[j, 'predictBinary'] = 1
            else:
                df_predict.loc[j, 'predictBinary'] = 0

        c_matrix = confusion_matrix(df_predict["bugBinary"], df_predict["predictBinary"], labels=[0, 1])
        tn, fp, fn, tp = c_matrix.ravel()
        # print(tn, fp, fn, tp)
        if (tn + fp) == 0:
            tnr_value = 0
        else:
            tnr_value = tn / (tn + fp)

        if (fp + tn) == 0:
            fpr = 0
        else:
            fpr = fp / (fp + tn)

        recall_value = recall_score(df_predict["bugBinary"], df_predict["predictBinary"], labels=[0, 1])
        f1_value = f1_score(df_predict["bugBinary"], df_predict["predictBinary"], labels=[0, 1])
        gm_value = (recall_value * tnr_value) ** 0.5
        pdr = recall_value
        pfr = fpr  # fp / (fp + tn)
        bpp_value = 1 - (((0 - pfr) ** 2 + (1 - pdr) ** 2) * 0.5) ** 0.5

        if i == 0:
            f1_1, gm_1, bpp_1 = f1_value, gm_value, bpp_value
        elif i == 1:
            f1_2, gm_2, bpp_2 = f1_value, gm_value, bpp_value
        elif i == 2:
            f1_3, gm_3, bpp_3 = f1_value, gm_value, bpp_value
        elif i == 3:
            f1_4, gm_4, bpp_4 = f1_value, gm_value, bpp_value
        elif i == 4:
            f1_5, gm_5, bpp_5 = f1_value, gm_value, bpp_value
        elif i == 5:
            f1_6, gm_6, bpp_6 = f1_value, gm_value, bpp_value
        elif i == 6:
            f1_7, gm_7, bpp_7 = f1_value, gm_value, bpp_value
        elif i == 7:
            f1_8, gm_8, bpp_8 = f1_value, gm_value, bpp_value
        elif i == 8:
            f1_9, gm_9, bpp_9 = f1_value, gm_value, bpp_value

    return f1_1, gm_1, bpp_1, f1_2, gm_2, bpp_2, f1_3, gm_3, bpp_3, f1_4, gm_4, bpp_4, f1_5, gm_5, bpp_5, \
           f1_6, gm_6, bpp_6, f1_7, gm_7, bpp_7, f1_8, gm_8, bpp_8, f1_9, gm_9, bpp_9


# 分别应用pred_20_score的投票得分，用0.1,0.2,...,0.9 九种对应百分比分位数作为阈值分别计算分类性能
def predict_score_performance(df_test):
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix
    f1_1, gm_1, bpp_1, mcc_1, f1_2, gm_2, bpp_2, mcc_2, f1_3, gm_3, bpp_3, mcc_3, f1_4, gm_4, bpp_4, mcc_4, f1_5, gm_5,\
    bpp_5, mcc_5, f1_6, gm_6, bpp_6, mcc_6, f1_7, gm_7, bpp_7, mcc_7, f1_8, gm_8, bpp_8, mcc_8, f1_9, gm_9, bpp_9,\
    mcc_9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    for i in range(9):

        df_predict = df_test.copy()

        for j in range(len(df_predict)):
            if df_predict.loc[j, 'pred_20_score'] >= df_predict.pred_20_score.quantile(0.1 * (i + 1)):
                df_predict.loc[j, 'predictBinary'] = 1
            else:
                df_predict.loc[j, 'predictBinary'] = 0

        c_matrix = confusion_matrix(df_predict["bugBinary"], df_predict["predictBinary"], labels=[0, 1])
        tn, fp, fn, tp = c_matrix.ravel()
        # print(tn, fp, fn, tp)
        if (tn + fp) == 0:
            tnr_value = 0
        else:
            tnr_value = tn / (tn + fp)

        if (fp + tn) == 0:
            fpr = 0
        else:
            fpr = fp / (fp + tn)

        recall_value = recall_score(df_predict["bugBinary"], df_predict["predictBinary"], labels=[0, 1])
        f1_value = f1_score(df_predict["bugBinary"], df_predict["predictBinary"], labels=[0, 1])
        gm_value = (recall_value * tnr_value) ** 0.5
        pdr = recall_value
        pfr = fpr  # fp / (fp + tn)
        bpp_value = 1 - (((0 - pfr) ** 2 + (1 - pdr) ** 2) * 0.5) ** 0.5
        mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

        if i == 0:
            f1_1, gm_1, bpp_1, mcc_1 = f1_value, gm_value, bpp_value, mcc
        elif i == 1:
            f1_2, gm_2, bpp_2, mcc_2 = f1_value, gm_value, bpp_value, mcc
        elif i == 2:
            f1_3, gm_3, bpp_3, mcc_3 = f1_value, gm_value, bpp_value, mcc
        elif i == 3:
            f1_4, gm_4, bpp_4, mcc_4 = f1_value, gm_value, bpp_value, mcc
        elif i == 4:
            f1_5, gm_5, bpp_5, mcc_5 = f1_value, gm_value, bpp_value, mcc
        elif i == 5:
            f1_6, gm_6, bpp_6, mcc_6 = f1_value, gm_value, bpp_value, mcc
        elif i == 6:
            f1_7, gm_7, bpp_7, mcc_7 = f1_value, gm_value, bpp_value, mcc
        elif i == 7:
            f1_8, gm_8, bpp_8, mcc_8 = f1_value, gm_value, bpp_value, mcc
        elif i == 8:
            f1_9, gm_9, bpp_9, mcc_9 = f1_value, gm_value, bpp_value, mcc

    return f1_1, gm_1, bpp_1, mcc_1, f1_2, gm_2, bpp_2, mcc_2, f1_3, gm_3, bpp_3, mcc_3, f1_4, gm_4, bpp_4, mcc_4, \
           f1_5, gm_5, bpp_5, mcc_5, f1_6, gm_6, bpp_6, mcc_6, f1_7, gm_7, bpp_7, mcc_7, f1_8, gm_8, bpp_8, mcc_8, \
           f1_9, gm_9, bpp_9, mcc_9


def median_threshold_on_current_version(working_dir, result_dir):
    # display all columns and rows, and set the item of row of dataframe
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 5000)

    # display all columns and rows, and set the item of row of dataframe
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 5000)

    # display all rows and columns of a matrix
    np.set_printoptions(threshold=np.sys.maxsize, linewidth=np.sys.maxsize)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # 20 object-oriented features and a dependent variable:  'bug'
    metrics_20 = ['loc', 'wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'dam', 'moa', 'mfa',
                  'cam', 'ic', 'cbm', 'amc', 'max_cc', 'avg_cc']

    # store predictive performance of all versions
    df_median = pd.DataFrame(
        columns=['project', 'current_version', 'Sample_size', 'f1_0.1', 'gm_0.1', 'bpp_0.1', 'mcc_0.1',
                 'f1_0.2', 'gm_0.2', 'bpp_0.2', 'mcc_0.2', 'f1_0.3', 'gm_0.3', 'bpp_0.3', 'mcc_0.3',
                 'f1_0.4', 'gm_0.4', 'bpp_0.4', 'mcc_0.4', 'f1_0.5', 'gm_0.5', 'bpp_0.5', 'mcc_0.5',
                 'f1_0.6', 'gm_0.6', 'bpp_0.6', 'mcc_0.6', 'f1_0.7', 'gm_0.7', 'bpp_0.7', 'mcc_0.7',
                 'f1_0.8', 'gm_0.8', 'bpp_0.8', 'mcc_0.8', 'f1_0.9', 'gm_0.9', 'bpp_0.9', 'mcc_0.9'], dtype=object)

    with open(working_dir + "List_versions.txt") as l_all:
        lines_all = l_all.readlines()

    for line in lines_all:

        name = line.replace("\n", "")
        project = name.split('-')[0]

        # df_name for spv
        df_name = pd.read_csv(working_dir + project + '/' + name)
        print("The current release ", name, project)
        print(df_name.head())
        # bugBinary表示bug的二进制形式
        df_name["bugBinary"] = df_name.bug.apply(lambda x: 1 if x > 0 else 0)

        # pred_20 存储20个度量应用中位数阈值比较之后的得分
        df_name['pred_20'] = 0

        for metric in metrics_20:
            print("the current file is ", name, "the current metric is ", metric)
            df_name = df_name[~df_name[metric].isin(['undef', 'undefined'])].reset_index(drop=True)
            df_name = df_name[~df_name[metric].isnull()].reset_index(drop=True)
            df_name = df_name[~df_name['loc'].isin([0])].reset_index(drop=True)

            metric_t = df_name[metric].median()
            print("version,  metric, and its threshold value are ", name, metric, metric_t)

            # pred_20用于存储2个度量预测的得分，最大值为20，全预测为有缺陷，最小值为0，全预测为无缺陷。
            # 此处假设所有度量都与度量正相关。
            df_name['pred_20'] = df_name.apply(
                lambda x: x['pred_20'] + 1 if float(x[metric]) >= metric_t else x['pred_20'] + 0, axis=1)

        # pred_20_score用于存储20个度量预测的得分再加上小数部分，小数部分等于当前模块的loc的倒数。
        df_name['pred_20_score'] = df_name.apply(lambda x: x['pred_20'] + (1 / x['loc']), axis=1)

        # 应用pred_20_score的投票得分,用0.1,0.2,...,0.9九种阈值分别计算分类性能
        f1_1, gm_1, bpp_1, mcc_1, f1_2, gm_2, bpp_2, mcc_2, f1_3, gm_3, bpp_3, mcc_3, f1_4, gm_4, bpp_4, mcc_4, \
        f1_5, gm_5, bpp_5, mcc_5, f1_6, gm_6, bpp_6, mcc_6, f1_7, gm_7, bpp_7, mcc_7, f1_8, gm_8, bpp_8, mcc_8, \
        f1_9, gm_9, bpp_9, mcc_9 = predict_score_performance(df_name)

        df_median = df_median.append(
            {'project': project, 'current_version': name[:-4], 'Sample_size': len(df_name),
             'f1_0.1': f1_1, 'gm_0.1': gm_1, 'bpp_0.1': bpp_1, 'mcc_0.1': mcc_1, 'f1_0.2': f1_2, 'gm_0.2': gm_2,
             'bpp_0.2': bpp_2, 'mcc_0.2': mcc_2, 'f1_0.3': f1_3, 'gm_0.3': gm_3, 'bpp_0.3': bpp_3, 'mcc_0.3': mcc_3,
             'f1_0.4': f1_4, 'gm_0.4': gm_4, 'bpp_0.4': bpp_4, 'mcc_0.4': mcc_4, 'f1_0.5': f1_5, 'gm_0.5': gm_5,
             'bpp_0.5': bpp_5, 'mcc_0.5': mcc_5, 'f1_0.6': f1_6, 'gm_0.6': gm_6, 'bpp_0.6': bpp_6, 'mcc_0.6': mcc_6,
             'f1_0.7': f1_7, 'gm_0.7': gm_7, 'bpp_0.7': bpp_7, 'mcc_0.7': mcc_7, 'f1_0.8': f1_8, 'gm_0.8': gm_8,
             'bpp_0.8': bpp_8, 'mcc_0.8': mcc_8, 'f1_0.9': f1_9, 'gm_0.9': gm_9, 'bpp_0.9': bpp_9, 'mcc_0.9': mcc_9},
            ignore_index=True)

        df_median.to_csv(result_dir + 'median_threshold_on_current_versions.csv', index=False)


if __name__ == '__main__':
    import os
    import sys
    import csv
    import math
    import time
    import random
    import shutil
    from datetime import datetime
    import pandas as pd
    import numpy as np

    s_time = time.time()

    work_Directory = "/home/myq/talcvdp/Xudata/MORPH_projects/"
    result_Directory = "/home/myq/talcvdp/median_threshold_on_current_version/"

    print(os.getcwd())
    os.chdir(work_Directory)
    print(work_Directory)
    print(os.getcwd())

    median_threshold_on_current_version(work_Directory, result_Directory)

    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ".\nFrom ", time.asctime(time.localtime(s_time)), " to ",
          time.asctime(time.localtime(e_time)), ",\nThis", os.path.basename(sys.argv[0]), "ended within",
          execution_time, "(s), or ", (execution_time / 60), " (m).")
