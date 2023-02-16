#!/usr/bin/env python
# encoding:utf-8
"""
Project: Object-oriented-Metric-Thresholds
File: RQ2_NB_DT_SVM_on_al_test_spv.py
Date : 2023/2/15 20:38
Author : njumy
HomePage : https://github.com/meiyuanqing
Email : dg1533019@smail.nju.edu.cn

应用NB_DT方法用前一版本的真实标签数据训练模型，然后在当前版本与TAL相同测试集的数据上预测性能，以方便比较
"""


def prediction_performance(bugBinary, predictBinary):
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix

    # confusion_matrix()函数中需要给出label, 0和1，否则该函数算不出TP,因为不知道哪个标签是poistive.
    c_matrix = confusion_matrix(bugBinary, predictBinary, labels=[0, 1])
    tn, fp, fn, tp = c_matrix.ravel()

    if (tn + fp) == 0:
        tnr_value = 0
    else:
        tnr_value = tn / (tn + fp)

    if (fp + tn) == 0:
        fpr = 0
    else:
        fpr = fp / (fp + tn)

    recall_value = recall_score(bugBinary, predictBinary, labels=[0, 1])
    precision_value = precision_score(bugBinary, predictBinary, labels=[0, 1])
    f1_value = f1_score(bugBinary, predictBinary, labels=[0, 1])
    gm_value = (recall_value * tnr_value) ** 0.5
    pdr = recall_value
    pfr = fpr  # fp / (fp + tn)
    bpp_value = 1 - (((0 - pfr) ** 2 + (1 - pdr) ** 2) * 0.5) ** 0.5

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    error_rate = (fp + fn) / (tp + fp + fn + tn)
    mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    return recall_value, precision_value, f1_value, gm_value, bpp_value, accuracy, error_rate, mcc


def random_forest_on_tal_test(working_dir, result_dir):

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn import tree
    from sklearn import svm

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

    dir_spv_nb = result_dir + 'NB/spv_20_percent/'
    dir_spv_dt = result_dir + 'DT/spv_20_percent/'
    dir_spv_svm = result_dir + 'SVM/spv_20_percent/'
    dir_test = 'F:/talcvdp/active_learning_middle_prior_pseudo_test/spv_20_percent/'
    # dir_test = 'F:/talcvdp/active_learning_middle_prior_pseudo_test/spv_15_percent/'
    # dir_test = 'F:/talcvdp/active_learning_middle_prior_pseudo_test/spv_10_percent/'
    # dir_test = 'F:/talcvdp/active_learning_middle_prior_pseudo_test/spv_5_percent/'

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    if not os.path.exists(result_dir + 'NB/'):
        os.mkdir(result_dir + 'NB/')

    if not os.path.exists(dir_spv_nb):
        os.mkdir(dir_spv_nb)

    if not os.path.exists(result_dir + 'DT/'):
        os.mkdir(result_dir + 'DT/')

    if not os.path.exists(dir_spv_dt):
        os.mkdir(dir_spv_dt)

    if not os.path.exists(result_dir + 'SVM/'):
        os.mkdir(result_dir + 'SVM/')

    if not os.path.exists(dir_spv_svm):
        os.mkdir(dir_spv_svm)


    # 20 object-oriented features and a dependent variable:  'bug'
    metrics_20 = ['loc', 'wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'dam', 'moa', 'mfa',
                  'cam', 'ic', 'cbm', 'amc', 'max_cc', 'avg_cc']

    with open(working_dir + 'List.txt') as l_projects:
        projects_t = l_projects.readlines()

    for project_t in projects_t:

        project = project_t.replace("\n", "")

        # read each version of a project in order
        versions_sorted = []
        with open(working_dir + 'List_versions.txt') as l_versions:
            versions_t = l_versions.readlines()

        for version_t in versions_t:
            if version_t.split('-')[0] == project:
                versions_sorted.append(version_t.replace('\n', ''))
        print(versions_sorted)

        # df_name_previous: previous version of current version. df_previous_all: all previous versions of current one.
        df_name_previous = pd.DataFrame()
        name_previous = ''

        for name in versions_sorted:
            # store the results of active_learning_semisupervised：'f1_value', 'gm_value', 'bpp_value'
            df_nb = pd.DataFrame(
                columns=['Sample_size', 'project', 'last_version', 'current_version', 'recall_value', 'precision_value',
                         'f1_value', 'gm_value', 'bpp_value', 'accuracy', 'error_rate', 'mcc'], dtype=object)

            df_dt = pd.DataFrame(
                columns=['Sample_size', 'project', 'last_version', 'current_version', 'recall_value', 'precision_value',
                         'f1_value', 'gm_value', 'bpp_value', 'accuracy', 'error_rate', 'mcc'], dtype=object)

            df_svm = pd.DataFrame(
                columns=['Sample_size', 'project', 'last_version', 'current_version', 'recall_value', 'precision_value',
                         'f1_value', 'gm_value', 'bpp_value', 'accuracy', 'error_rate', 'mcc'], dtype=object)

            # df_name for spv
            df_name = pd.read_csv(working_dir + project + '/' + name)

            # 与for循环最后df_name_previous = df_name互应，若第一个赋值给上一个，进入下一循环，否则在for循环最后一句把当前赋值给上一个
            if name == versions_sorted[0]:
                df_name_previous = df_name.copy()
                name_previous = name
                continue

            # bugBinary表示bug的二进制形式, 'iter':表示该行(模块)是否被选中，初始为零，表示都没选中,反之选中
            df_name_previous["bugBinary"] = df_name_previous.bug.apply(lambda x: 1 if x > 0 else 0)
            df_name_previous['iter'] = 1

            # 读入与TAL方法相同的测试集数据
            df_name_test = pd.read_csv(dir_test + 'testing_data_' + name)

            # Gaussian Naive Bayes
            gnb = GaussianNB()
            df_name_test['predictBinary'] = gnb.fit(df_name_previous.loc[:, metrics_20],
                df_name_previous.loc[:, 'bugBinary']).predict(df_name_test.loc[:, metrics_20])

            print(df_name_test.columns.values)
            recall_value, precision_value, f1_value, gm_value, bpp_value, accuracy, error_rate, mcc = \
                prediction_performance(df_name_test['bugBinary'], df_name_test['predictBinary'])
            print(recall_value, precision_value, f1_value, gm_value, bpp_value, accuracy, error_rate, mcc)
            df_nb = df_nb.append(
                {'Sample_size': len(df_name_test), 'project': name.split('-')[0], 'last_version': name_previous[:-4],
                 'current_version': name[:-4], 'recall_value': recall_value, 'precision_value': precision_value,
                 'f1_value': f1_value, 'gm_value': gm_value, 'bpp_value': bpp_value, 'accuracy': accuracy,
                 'error_rate': error_rate, 'mcc': mcc}, ignore_index=True)
            df_nb.to_csv(dir_spv_nb + 'NaiveBayes_' + name, index=False)

            # # DecisionTreeClassifier
            clf_dt = tree.DecisionTreeClassifier()
            clf_dt = clf_dt.fit(df_name_previous.loc[:, metrics_20], df_name_previous.loc[:, 'bugBinary'])
            df_name_test['predictBinary'] = clf_dt.predict(df_name_test.loc[:, metrics_20])

            print(df_name_test.columns.values)
            recall_value, precision_value, f1_value, gm_value, bpp_value, accuracy, error_rate, mcc = \
                prediction_performance(df_name_test['bugBinary'], df_name_test['predictBinary'])
            print(recall_value, precision_value, f1_value, gm_value, bpp_value, accuracy, error_rate, mcc)
            df_dt = df_dt.append(
                {'Sample_size': len(df_name_test), 'project': name.split('-')[0], 'last_version': name_previous[:-4],
                 'current_version': name[:-4], 'recall_value': recall_value, 'precision_value': precision_value,
                 'f1_value': f1_value, 'gm_value': gm_value, 'bpp_value': bpp_value, 'accuracy': accuracy,
                 'error_rate': error_rate, 'mcc': mcc}, ignore_index=True)
            df_dt.to_csv(dir_spv_dt + 'DecisionTree_' + name, index=False)

            # svm
            clf_svm = svm.SVC()
            clf_svm.fit(df_name_previous.loc[:, metrics_20], df_name_previous.loc[:, 'bugBinary'])
            # clf_svm = clf_svm.fit(df_training.loc[:, metrics], df_training.loc[:, 'bug'])
            df_name_test['predictBinary'] = clf_svm.predict(df_name_test.loc[:, metrics_20])

            print(df_name_test.columns.values)
            recall_value, precision_value, f1_value, gm_value, bpp_value, accuracy, error_rate, mcc = \
                prediction_performance(df_name_test['bugBinary'], df_name_test['predictBinary'])
            print(recall_value, precision_value, f1_value, gm_value, bpp_value, accuracy, error_rate, mcc)
            df_svm = df_svm.append(
                {'Sample_size': len(df_name_test), 'project': name.split('-')[0], 'last_version': name_previous[:-4],
                 'current_version': name[:-4], 'recall_value': recall_value, 'precision_value': precision_value,
                 'f1_value': f1_value, 'gm_value': gm_value, 'bpp_value': bpp_value, 'accuracy': accuracy,
                 'error_rate': error_rate, 'mcc': mcc}, ignore_index=True)
            df_svm.to_csv(dir_spv_svm + 'svm_' + name, index=False)

            # 当前版本作为下一版本的前一版本
            df_name_previous = df_name.copy()
            name_previous = name
            print('********************this is an end of a version ' + name + '*******************************')


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

    work_Directory = "F:/talcvdp/Xudata/MORPH_projects/"
    result_Directory = "F:/talcvdp/NB_DT_SVM_on_TAL_test/"
    print(result_Directory[:-26])

    print(os.getcwd())
    os.chdir(work_Directory)
    print(work_Directory)
    print(os.getcwd())

    random_forest_on_tal_test(work_Directory, result_Directory)

    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ".\nFrom ", time.asctime(time.localtime(s_time)), " to ",
          time.asctime(time.localtime(e_time)), ",\nThis", os.path.basename(sys.argv[0]), "ended within",
          execution_time, "(s), or ", (execution_time / 60), " (m).")