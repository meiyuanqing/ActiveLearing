#!/usr/bin/env python
# encoding:utf-8
"""
Project: Object-oriented-Metric-Thresholds
File: RQ2_random_forest_logit_on_al_test_spv.py
Date : 2022/11/25 16:42
Author : njumy
HomePage : https://github.com/meiyuanqing
Email : dg1533019@smail.nju.edu.cn

应用随机森林的方法用前一版本的真实标签数据训练模型，然后在当前版本与TAL相同测试集的数据上预测性能，以方便比较
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

    dir_spv_rf = result_dir + 'randomForest/spv_20_percent/'
    dir_spv_logit = result_dir + 'logit/spv_20_percent/'
    dir_test = 'F:/talcvdp/active_learning_middle_prior_pseudo_test/spv_20_percent/'
    # dir_test = 'F:/talcvdp/active_learning_middle_prior_pseudo_test/spv_15_percent/'
    # dir_test = 'F:/talcvdp/active_learning_middle_prior_pseudo_test/spv_10_percent/'
    # dir_test = 'F:/talcvdp/active_learning_middle_prior_pseudo_test/spv_5_percent/'

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    if not os.path.exists(result_dir + 'randomForest/'):
        os.mkdir(result_dir + 'randomForest/')

    if not os.path.exists(dir_spv_rf):
        os.mkdir(dir_spv_rf)

    if not os.path.exists(result_dir + 'logit/'):
        os.mkdir(result_dir + 'logit/')

    if not os.path.exists(dir_spv_logit):
        os.mkdir(dir_spv_logit)

    # 20 object-oriented features and a dependent variable:  'bug'
    metrics_20 = ['loc', 'wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'dam', 'moa', 'mfa',
                  'cam', 'ic', 'cbm', 'amc', 'max_cc', 'avg_cc']

    with open(working_dir + 'List.txt') as l_projects:
        projects_t = l_projects.readlines()

    for project_t in projects_t:

        project = project_t.replace("\n", "")

        if project == 'ivy':
            continue

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
            df_random_forest = pd.DataFrame(
                columns=['Sample_size', 'project', 'last_version', 'current_version', 'recall_value', 'precision_value',
                         'f1_value', 'gm_value', 'bpp_value', 'accuracy', 'error_rate', 'mcc'], dtype=object)

            df_logit = pd.DataFrame(
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

            # 随机森林分类模型（RandomForestClassifier）与随机森林回归模型（RandomForestRegressor）
            df_rf_x = df_name_previous.loc[:, metrics_20]
            df_rf_y = df_name_previous.loc[:, 'bugBinary']
            # regr = RandomForestRegressor(random_state=0)
            # regr = RandomForestRegressor(max_depth=2, random_state=0)
            regr = RandomForestClassifier(random_state=0)
            # regr = RandomForestClassifier(n_estimators=100, criterion='entropy', max_features='sqrt', max_depth=100,
            #                               random_state=None)
            regr.fit(df_rf_x, df_rf_y)

            # 随机森林分类预测
            for row in range(len(df_name_test)):
                # print(regr.predict(df_name_test[df_name_test.index == row].loc[:, metrics_20]))
                # print(regr.predict_proba(df_name_test[df_name_test.index == row].loc[:, metrics_20]),
                #       regr.predict(df_name_test[df_name_test.index == row].loc[:, metrics_20]))

                # if regr.predict(df_name_test[df_name_test.index == row].loc[:, metrics_20]) >= 0.2:
                #     df_name_test.loc[row, 'predictBinary'] = 1
                # else:
                #     df_name_test.loc[row, 'predictBinary'] = 0
                df_name_test.loc[row, 'predictBinary'] = \
                    regr.predict(df_name_test[df_name_test.index == row].loc[:, metrics_20])

            print(df_name_test.columns.values)
            recall_value, precision_value, f1_value, gm_value, bpp_value, accuracy, error_rate, mcc = \
                prediction_performance(df_name_test['bugBinary'], df_name_test['predictBinary'])
            print(recall_value, precision_value, f1_value, gm_value, bpp_value, accuracy, error_rate, mcc)
            df_random_forest = df_random_forest.append(
                {'Sample_size': len(df_name_test), 'project': name.split('-')[0], 'last_version': name_previous[:-4],
                 'current_version': name[:-4], 'recall_value': recall_value, 'precision_value': precision_value,
                 'f1_value': f1_value, 'gm_value': gm_value, 'bpp_value': bpp_value, 'accuracy': accuracy,
                 'error_rate': error_rate, 'mcc': mcc}, ignore_index=True)
            df_random_forest.to_csv(dir_spv_rf + 'randomForestPerformance_' + name, index=False)

            # logit 模型
            # scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logit
            clf = LogisticRegression(random_state=0, max_iter=10000).fit(df_rf_x, df_rf_y)
            # logit分类预测
            for row in range(len(df_name_test)):
                df_name_test.loc[row, 'logit_predictBinary'] = \
                    clf.predict(df_name_test[df_name_test.index == row].loc[:, metrics_20])
            # print(df_name_test)

            recall_value_logit, precision_value_logit, f1_value_logit, gm_value_logit, bpp_value_logit, accuracy_logit,\
            error_rate_logit, mcc_logit = \
                prediction_performance(df_name_test['bugBinary'], df_name_test['logit_predictBinary'])
            print(recall_value_logit, precision_value_logit, f1_value_logit, gm_value_logit, bpp_value_logit,
                  accuracy_logit, error_rate_logit, mcc_logit)
            df_logit = df_logit.append(
                {'Sample_size': len(df_name_test), 'project': name.split('-')[0], 'last_version': name_previous[:-4],
                 'current_version': name[:-4], 'recall_value': recall_value_logit,
                 'precision_value': precision_value_logit, 'f1_value': f1_value_logit, 'gm_value': gm_value_logit,
                 'bpp_value': bpp_value_logit, 'accuracy': accuracy_logit, 'error_rate': error_rate_logit,
                 'mcc': mcc_logit}, ignore_index=True)
            df_logit.to_csv(dir_spv_logit + 'logitPerformance_' + name, index=False)

            # 当前版本作为下一版本的前一版本
            df_name_previous = df_name.copy()
            name_previous = name
            print('********************this is an end of a version ' + name + '*******************************')
            # break

        # break


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
    result_Directory = "F:/talcvdp/random_forest_logit_on_TAL_test/"
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
