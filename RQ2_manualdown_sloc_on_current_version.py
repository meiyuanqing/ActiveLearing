#!/usr/bin/env python
# encoding:utf-8
"""
Project: Object-oriented-Metric-Thresholds
File: RQ2_manualdown_`sloc`_on_current_version.py
Date : 2022/11/20 11:09
Author : njumy
HomePage : https://github.com/meiyuanqing
Email : dg1533019@smail.nju.edu.cn

在数据集上，应用SLOC的manualdown方法取中位数阈值，并与T-median方法上做比较。

"""

# compute the predictive performance of each threshold
# input: metric(including metric, sloc, labeled data(bug)), threshold  (, correlation);
# output: Recall, Precision, FPR, TNR, Accuracy, Error rate, F-measure, MCC, GM, and AUC
def predictive_performance(df_metric, threshold):

    from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix
    pearson, er, recall, precision, fpr, tnr, accuracy, error_rate, f1, mcc, gm, auc, bpp = \
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    # display all columns and rows, and set the item of row of dataframe
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 5000)

    metric_name = ''
    # If the name of metric is SLOC, the ER does not to compute.
    for col in df_metric.columns:
        if col not in ['bug', 'bugBinary', 'loc']:
            metric_name = col
        else:
            metric_name = 'loc'

    # 判断每个度量与bug之间的关系,用于阈值判断正反例
    Corr_metric_bug = df_metric.loc[:, [metric_name, 'bug']].corr('spearman')
    Spearman_value = Corr_metric_bug[metric_name][1]
    pearson = 2 * np.sin(np.pi * Spearman_value / 6)

    if pearson < 0:
        df_metric['predictBinary'] = df_metric[metric_name].apply(lambda x: 1 if x <= threshold else 0)
    else:
        df_metric['predictBinary'] = df_metric[metric_name].apply(lambda x: 1 if x >= threshold else 0)

    # confusion_matrix()函数中需要给出label, 0和1，否则该函数算不出TP,因为不知道哪个标签是poistive.
    c_matrix = confusion_matrix(df_metric["bugBinary"], df_metric['predictBinary'], labels=[0, 1])
    tn, fp, fn, tp = c_matrix.ravel()

    if (tn + fp) == 0:
        tnr = 0
    else:
        tnr = tn / (tn + fp)

    if (fp + tn) == 0:
        fpr = 0
    else:
        fpr = fp / (fp + tn)

    s_p, s, f_p, f = 0, 0, 0, 0

    if pearson < 0:
        for j in range(len(df_metric)):
            if float(df_metric.loc[j, metric_name]) <= threshold:
                df_metric.loc[j, 'predictBinary'] = 1
                s += df_metric.loc[j, 'loc']
                s_p += df_metric.loc[j, 'loc'] * 1
                f += df_metric.loc[j, 'bug']
                f_p += df_metric.loc[j, 'bug'] * 1
            else:
                df_metric.loc[j, 'predictBinary'] = 0
                s += df_metric.loc[j, 'loc']
                s_p += df_metric.loc[j, 'loc'] * 0
                f += df_metric.loc[j, 'bug']
                f_p += df_metric.loc[j, 'bug'] * 0
    else:
        for j in range(len(df_metric)):
            if float(df_metric.loc[j, metric_name]) >= threshold:
                df_metric.loc[j, 'predictBinary'] = 1
                s += df_metric.loc[j, 'loc']
                s_p += df_metric.loc[j, 'loc'] * 1
                f += df_metric.loc[j, 'bug']
                f_p += df_metric.loc[j, 'bug'] * 1
            else:
                df_metric.loc[j, 'predictBinary'] = 0
                s += df_metric.loc[j, 'loc']
                s_p += df_metric.loc[j, 'loc'] * 0
                f += df_metric.loc[j, 'bug']
                f_p += df_metric.loc[j, 'bug'] * 0

    if f != 0:
        effort_random = f_p / f
    else:
        effort_random = 0

    if s != 0:
        effort_m = s_p / s
    else:
        effort_m = 0

    if effort_random != 0:
        er = (effort_random - effort_m) / effort_random
    else:
        er = 0

    if metric_name == 'loc':
        er = 0

    auc = roc_auc_score(df_metric['bugBinary'], df_metric['predictBinary'], labels=[0, 1])
    recall = recall_score(df_metric['bugBinary'], df_metric['predictBinary'], labels=[0, 1])
    precision = precision_score(df_metric['bugBinary'], df_metric['predictBinary'], labels=[0, 1])
    f1 = f1_score(df_metric['bugBinary'], df_metric['predictBinary'], labels=[0, 1])
    gm = (recall * tnr) ** 0.5
    pdr = recall
    pfr = fpr  # fp / (fp + tn)
    bpp = 1 - (((0 - pfr) ** 2 + (1 - pdr) ** 2) * 0.5) ** 0.5
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    error_rate = (fp + fn) / (tp + fp + fn + tn)
    mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    # 以下代码用于计算AUC方差，后期若对AUC做元分析，则可以输出方差来计算
    # if auc_value > 1 or auc_value < 0:
    #     auc_value = 0.5
    # elif auc_value < 0.5:
    #     auc_value = 1 - auc_value
    # Q1 = auc_value / (2 - auc_value)
    # Q2 = 2 * auc_value * auc_value / (1 + auc_value)
    # auc_value_variance = auc_value * (1 - auc_value) + (value_1 - 1) * (Q1 - auc_value * auc_value) \
    #                      + (value_0 - 1) * (Q2 - auc_value * auc_value)
    # auc_value_variance = auc_value_variance / (value_0 * value_1)

    return pearson, er, recall, precision, fpr, tnr, accuracy, error_rate, f1, mcc, gm, auc, bpp


def manual_down(work_dir, result_dir):
    # display all columns and rows, and set the item of row of dataframe
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 5000)

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # 20 object-oriented features and a dependent variable:  'bug'
    metrics_20 = ['loc', 'wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'dam', 'moa', 'mfa',
                  'cam', 'ic', 'cbm', 'amc', 'max_cc', 'avg_cc']

    # 'sloc_median', 'sloc_max', 'sloc_min', 'sloc_mean', 'sloc_var',
    # 'precision', 'recall','auc', 'auc_var', 'gm', 'f1', 'bpp', 'fpr', 'tnr', 'er', 'accuracy', 'error_rate', 'mcc'
    manual_down_performance = pd.DataFrame(
        columns=['fileName', 'metric', 'metric_median', 'metric_max', 'metric_min', 'metric_mean', 'metric_var',
                 'pearson', 'precision', 'recall', 'auc', 'gm', 'f1', 'bpp', 'fpr', 'tnr', 'er',
                 'accuracy', 'error_rate', 'mcc'])

    with open(work_dir + "List_versions.txt") as l_all:
        lines_all = l_all.readlines()

    for line in lines_all:

        name = line.replace("\n", "")
        project = name.split('-')[0]

        print("The project is ", project)

        if os.path.exists(result_dir + project + '_manualdown_sloc_performance.csv'):
            print("The ", project, '_manualdown_sloc_performance.csv', " is created in last execution, do not be created this time.")
            continue

        df_name = pd.read_csv(work_dir + project + '/' + name)

        i = 0
        for metric in metrics_20:

            if metric != 'loc':
                continue

            print("The current file is ", name, ", the ", i, "th metric is ", metric)

            # 由于bug中存储的是缺陷个数,转化为二进制存储,若x>2,则可预测bug为3个以上的阈值,其他类推
            df_name['bugBinary'] = df_name.bug.apply(lambda x: 1 if x > 0 else 0)

            # 删除度量中空值和undef, undefined值 ['bug', 'bugBinary', metric, 'SLOC']
            df_metric = df_name[~df_name[metric].isin(['undef', 'undefined'])].loc[:, :]

            df_metric = df_metric.dropna(subset=[metric]).reset_index(drop=True)

            # exclude data sets (each corresponding to a class) where m has fewer than six non-zero data points
            if len(df_metric) - len(df_metric[df_metric[metric].isin([0, '0'])]) < 6:
                continue

            # Only one class present in y_true. ROC AUC score is not defined in that case. eg., mrm-1.1.csv
            if len(df_metric['bugBinary'].value_counts()) == 1:
                continue

            metric_min = df_metric[metric].astype(float).min()
            metric_max = df_metric[metric].astype(float).max()
            metric_median = df_metric[metric].astype(float).median()
            metric_mean = df_metric[metric].astype(float).mean()
            metric_var = df_metric[metric].astype(float).var()
            print(metric_min, metric_median, metric_max, metric_mean, metric_var)
            # .astype(float)
            if metric == 'loc':
                pearson, er, recall, precision, fpr, tnr, accuracy, error_rate, f1, mcc, gm, auc, bpp = \
                    predictive_performance(df_metric.loc[:, ['bug', 'bugBinary', metric]], metric_median)
            else:
                pearson, er, recall, precision, fpr, tnr, accuracy, error_rate, f1, mcc, gm, auc, bpp = \
                    predictive_performance(df_metric.loc[:, ['bug', 'bugBinary', metric, 'loc']], metric_median)

            manual_down_performance = manual_down_performance.append(
                {'fileName': name[:-4], 'metric': metric, 'metric_median': metric_median,
                 'metric_max': metric_max, 'metric_min': metric_min, 'metric_mean': metric_mean,
                 'metric_var': metric_var, 'pearson': pearson, 'precision': precision, 'recall': recall,
                 'auc': auc, 'gm': gm, 'f1': f1, 'bpp': bpp, 'fpr': fpr, 'tnr': tnr, 'er': er,
                 'accuracy': accuracy, 'error_rate': error_rate, 'mcc': mcc}, ignore_index=True)

            i += 1

    manual_down_performance.to_csv(result_dir + 'manualdown_performance_on_current_versions.csv', index=False)


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

    work_directory = "F:/talcvdp/Xudata/MORPH_projects/"
    result_directory = "F:/talcvdp/manual_down_on_current_version/"
    os.chdir(work_directory)

    manual_down(work_directory, result_directory)

    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ".\nFrom ", time.asctime(time.localtime(s_time)), " to ",
          time.asctime(time.localtime(e_time)), ",\nThis", os.path.basename(sys.argv[0]), "ended within",
          execution_time, "(s), or ", (execution_time / 60), " (m).")