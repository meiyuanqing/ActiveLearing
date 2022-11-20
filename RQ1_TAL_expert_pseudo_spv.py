#!/usr/bin/env python
# encoding:utf-8
"""
Project: Object-oriented-Metric-Thresholds
File: RQ1_TAL_expert_pseudo_spv.py
Date : 2022/11/16 13:08
Author : njumy
HomePage : https://github.com/meiyuanqing
Email : dg1533019@smail.nju.edu.cn

TAL-expert-pseudo: the method applies the TAL method only by using the selected modules labeled by domain experts and
                   pseudo labeled modules of the current version to train the threshold value of each metric.
"""


# output: the threshold derived from MGM
# note that the dataframe should input astype(float), i.e., MGM_threshold(df.astype(float))
def max_gm_threshold(df):
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix

    metric_name = ''
    for col in df.columns:
        if col not in ['bug', 'iter_bug']:
            metric_name = col

    # 2. 依次用该度量每一个值作为阈值计算出各自预测性能值,然后选择预测性能值最大的作为阈值,分别定义存入list,取最大值和最大值的下标值
    # 同时输出gm最大值阈值的F1和BPP值。
    GMs = []
    gm_max_value = 0
    f1_with_gm_max = 0
    bpp_with_gm_max = 0
    i_gm_max = 0

    # 判断每个度量与bug之间的关系,用于阈值判断正反例
    Corr_metric_bug = df.loc[:, [metric_name, 'iter_bug']].corr('spearman')

    Spearman_value = Corr_metric_bug[metric_name][1]
    Pearson_value = 2 * np.sin(np.pi * Spearman_value / 6)

    # the i value in this loop, is the subscript value in the list of AUCs, GMs etc.
    for i in range(len(df)):

        t = df.loc[i, metric_name]

        if Pearson_value < 0:
            df['predictBinary'] = df[metric_name].apply(lambda x: 1 if x <= t else 0)
        else:
            df['predictBinary'] = df[metric_name].apply(lambda x: 1 if x >= t else 0)

        # confusion_matrix()函数中需要给出label, 0和1，否则该函数算不出TP,因为不知道哪个标签是poistive.
        c_matrix = confusion_matrix(df["iter_bug"], df['predictBinary'], labels=[0, 1])
        tn, fp, fn, tp = c_matrix.ravel()

        if (tn + fp) == 0:
            tnr_value = 0
        else:
            tnr_value = tn / (tn + fp)

        if (fp + tn) == 0:
            fpr = 0
        else:
            fpr = fp / (fp + tn)

        # auc_value = roc_auc_score(df['iter_bug'], df['predictBinary'])
        recall_value = recall_score(df['iter_bug'], df['predictBinary'], labels=[0, 1])
        # precision_value = precision_score(df['bugBinary'], df['predictBinary'], labels=[0, 1])
        f1_value = f1_score(df['iter_bug'], df['predictBinary'], labels=[0, 1])

        gm_value = (recall_value * tnr_value) ** 0.5
        pdr = recall_value
        pfr = fpr  # fp / (fp + tn)
        bpp_value = 1 - (((0 - pfr) ** 2 + (1 - pdr) ** 2) * 0.5) ** 0.5

        GMs.append(gm_value)

        # 求出上述list中最大值，及对应的i值，可能会有几个值相同，且为最大值，则取第一次找到那个值(i)为阈值
        if gm_value > gm_max_value:
            gm_max_value = gm_value
            f1_with_gm_max = f1_value
            bpp_with_gm_max = bpp_value
            i_gm_max = i

    # 计算阈值,包括其他四个类型阈值
    gm_t = df.loc[i_gm_max, metric_name]

    return Pearson_value, gm_t, gm_max_value, f1_with_gm_max, bpp_with_gm_max


# 用当前版本上的阈值，在主动学习选择的模块上得出F1,GM,BPP的性能指标值
def adaptive_threshold(threhshold, df):
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix

    metric_name = ''
    for col in df.columns:
        if col not in ['bug', 'iter_bug']:
            metric_name = col

    # f1_adaptive, gm_adaptive, bpp_adaptive = 0, 0, 0

    # 判断每个度量与bug之间的关系,用于阈值判断正反例
    Corr_metric_bug = df.loc[:, [metric_name, 'iter_bug']].corr('spearman')

    Spearman_value = Corr_metric_bug[metric_name][1]
    Pearson_value = 2 * np.sin(np.pi * Spearman_value / 6)

    if Pearson_value < 0:
        df['predictBinary'] = df[metric_name].apply(lambda x: 1 if x <= threhshold else 0)
    else:
        df['predictBinary'] = df[metric_name].apply(lambda x: 1 if x >= threhshold else 0)

    # confusion_matrix()函数中需要给出label, 0和1，否则该函数算不出TP,因为不知道哪个标签是poistive.
    c_matrix = confusion_matrix(df["iter_bug"], df['predictBinary'], labels=[0, 1])
    tn, fp, fn, tp = c_matrix.ravel()

    if (tn + fp) == 0:
        tnr_value = 0
    else:
        tnr_value = tn / (tn + fp)

    if (fp + tn) == 0:
        fpr = 0
    else:
        fpr = fp / (fp + tn)

    # auc_value = roc_auc_score(df['iter_bug'], df['predictBinary'])
    recall_value = recall_score(df['iter_bug'], df['predictBinary'], labels=[0, 1])
    # precision_value = precision_score(df['bugBinary'], df['predictBinary'], labels=[0, 1])
    f1_value = f1_score(df['iter_bug'], df['predictBinary'], labels=[0, 1])

    gm_value = (recall_value * tnr_value) ** 0.5
    pdr = recall_value
    pfr = fpr  # fp / (fp + tn)
    bpp_value = 1 - (((0 - pfr) ** 2 + (1 - pdr) ** 2) * 0.5) ** 0.5

    # return f1_adaptive, gm_adaptive, bpp_adaptive
    return f1_value, gm_value, bpp_value


# 用于存储dataframe中一条记录
def store_data(iteration, last_version, current_version, Sample_size, iter_t, df_als):
    # 存储本次迭代的调整阈值及计算阈值中gm最大化的gm值
    df_als = df_als.append(
        {'iteration': iteration, 'last_version': last_version, 'current_version': current_version,
         'Sample_size': Sample_size,
         'loc': iter_t['loc'], 'wmc': iter_t['wmc'], 'dit': iter_t['dit'], 'noc': iter_t['noc'],
         'cbo': iter_t['cbo'], 'rfc': iter_t['rfc'], 'lcom': iter_t['lcom'], 'ca': iter_t['ca'],
         'ce': iter_t['ce'], 'npm': iter_t['npm'], 'lcom3': iter_t['lcom3'], 'dam': iter_t['dam'],
         'moa': iter_t['moa'], 'mfa': iter_t['mfa'], "cam": iter_t["cam"], "ic": iter_t["ic"],
         "cbm": iter_t["cbm"], "amc": iter_t["amc"], "max_cc": iter_t["max_cc"],
         "avg_cc": iter_t["avg_cc"]}, ignore_index=True)
    return df_als


# er指标中将SLOC换成loc
def predictive_performance(metric, metric_p, metric_t, df_test):
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix

    if metric_p > 0:
        df_test['predictBinary'] = df_test[metric].apply(lambda x: 1 if float(x) >= metric_t else 0)
    else:
        df_test['predictBinary'] = df_test[metric].apply(lambda x: 1 if float(x) <= metric_t else 0)

    # confusion_matrix()函数中需要给出label, 0和1，否则该函数算不出TP,因为不知道哪个标签是poistive.
    c_matrix = confusion_matrix(df_test["bugBinary"], df_test['predictBinary'], labels=[0, 1])
    tn, fp, fn, tp = c_matrix.ravel()

    if (tn + fp) == 0:
        tnr_value = 0
    else:
        tnr_value = tn / (tn + fp)

    if (fp + tn) == 0:
        fpr = 0
    else:
        fpr = fp / (fp + tn)

    s_p, s, f_p, f = 0, 0, 0, 0

    if metric_p < 0:
        for j in range(len(df_test)):
            if float(df_test.loc[j, metric]) <= metric_t:
                df_test.loc[j, 'predictBinary'] = 1
                s += df_test.loc[j, 'loc']
                s_p += df_test.loc[j, 'loc'] * 1
                f += df_test.loc[j, 'bug']
                f_p += df_test.loc[j, 'bug'] * 1
            else:
                df_test.loc[j, 'predictBinary'] = 0
                s += df_test.loc[j, 'loc']
                s_p += df_test.loc[j, 'loc'] * 0
                f += df_test.loc[j, 'bug']
                f_p += df_test.loc[j, 'bug'] * 0
    else:
        for j in range(len(df_test)):
            if float(df_test.loc[j, metric]) >= metric_t:
                df_test.loc[j, 'predictBinary'] = 1
                s += df_test.loc[j, 'loc']
                s_p += df_test.loc[j, 'loc'] * 1
                f += df_test.loc[j, 'bug']
                f_p += df_test.loc[j, 'bug'] * 1
            else:
                df_test.loc[j, 'predictBinary'] = 0
                s += df_test.loc[j, 'loc']
                s_p += df_test.loc[j, 'loc'] * 0
                f += df_test.loc[j, 'bug']
                f_p += df_test.loc[j, 'bug'] * 0

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

    if metric == 'loc':
        er = 0
    try:
        auc_value = roc_auc_score(df_test['bugBinary'], df_test['predictBinary'], labels=[0, 1])
    except Exception as err1:
        print(err1)
        auc_value = 0.5
    recall_value = recall_score(df_test['bugBinary'], df_test['predictBinary'], labels=[0, 1])
    precision_value = precision_score(df_test['bugBinary'], df_test['predictBinary'], labels=[0, 1])
    f1_value = f1_score(df_test['bugBinary'], df_test['predictBinary'], labels=[0, 1])
    gm_value = (recall_value * tnr_value) ** 0.5
    pdr = recall_value
    pfr = fpr  # fp / (fp + tn)
    bpp_value = 1 - (((0 - pfr) ** 2 + (1 - pdr) ** 2) * 0.5) ** 0.5

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    error_rate = (fp + fn) / (tp + fp + fn + tn)
    mcc = (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    valueOfbugBinary = df_test["predictBinary"].value_counts()  # 0 和 1 的各自的个数

    if len(valueOfbugBinary) <= 1:
        if valueOfbugBinary.keys()[0] == 0:
            value_0 = valueOfbugBinary[0]
            value_1 = 0
        else:
            value_0 = 0
            value_1 = valueOfbugBinary[1]
    else:
        value_0 = valueOfbugBinary[0]
        value_1 = valueOfbugBinary[1]

    if auc_value > 1 or auc_value < 0:
        auc_value = 0.5
    elif auc_value < 0.5:
        auc_value = 1 - auc_value
    Q1 = auc_value / (2 - auc_value)
    Q2 = 2 * auc_value * auc_value / (1 + auc_value)
    auc_value_variance = auc_value * (1 - auc_value) + (value_1 - 1) * (Q1 - auc_value * auc_value) \
                         + (value_0 - 1) * (Q2 - auc_value * auc_value)
    auc_value_variance = auc_value_variance / (value_0 * value_1)

    return precision_value, recall_value, auc_value, auc_value_variance, gm_value, f1_value, bpp_value, fpr, \
           tnr_value, er, accuracy, error_rate, mcc


def do_predction(i, name_previous, name, metrics_positvie, iter_pearson, iter_t, df_test, df_alt):
    iter_t_precision = {}
    iter_t_recall = {}
    iter_t_auc = {}
    iter_t_auc_var = {}
    iter_t_gm = {}
    iter_t_f1 = {}
    iter_t_bpp = {}
    iter_t_fpr = {}
    iter_t_tnr = {}
    iter_t_er = {}
    iter_t_accuracy = {}
    iter_t_error_rate = {}
    iter_t_mcc = {}

    for metric in metrics_positvie:
        # 在迭代中被删去的度量不计算其预测性能
        if iter_t[metric] == '/':
            iter_t_precision[metric] = '/'
            iter_t_recall[metric] = '/'
            iter_t_auc[metric] = '/'
            iter_t_auc_var[metric] = '/'
            iter_t_gm[metric] = '/'
            iter_t_f1[metric] = '/'
            iter_t_bpp[metric] = '/'
            iter_t_fpr[metric] = '/'
            iter_t_tnr[metric] = '/'
            iter_t_er[metric] = '/'
            iter_t_accuracy[metric] = '/'
            iter_t_error_rate[metric] = '/'
            iter_t_mcc[metric] = '/'
            continue
        precision_m, recall_m, auc_m, auc_var_m, gm_m, f1_m, bpp_m, fpr_m, tnr_m, er_m, accuracy_m, error_rate_m, mcc_m \
            = predictive_performance(metric, iter_pearson[metric], iter_t[metric], df_test)
        iter_t_precision[metric] = precision_m
        iter_t_recall[metric] = recall_m
        iter_t_auc[metric] = auc_m
        iter_t_auc_var[metric] = auc_var_m
        iter_t_gm[metric] = gm_m
        iter_t_f1[metric] = f1_m
        iter_t_bpp[metric] = bpp_m
        iter_t_fpr[metric] = fpr_m
        iter_t_tnr[metric] = tnr_m
        iter_t_er[metric] = er_m
        iter_t_accuracy[metric] = accuracy_m
        iter_t_error_rate[metric] = error_rate_m
        iter_t_mcc[metric] = mcc_m

    # store precision
    # 'iteration', 'last_version', 'current_version', 'Sample_size', 'loc', 'wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce',
    # 'npm', 'lcom3', 'dam', 'moa', 'mfa', 'cam', 'ic', 'cbm', 'amc', 'max_cc', 'avg_cc'
    df_alt = df_alt.append(
        {'iteration': str(i) + '_t_precision', 'last_version': name_previous, 'current_version': name[:-4],
         'Sample_size': len(df_test),
         'loc': iter_t_precision['loc'], 'wmc': iter_t_precision['wmc'], 'dit': iter_t_precision['dit'],
         'noc': iter_t_precision['noc'], 'cbo': iter_t_precision['cbo'], 'rfc': iter_t_precision['rfc'],
         'lcom': iter_t_precision['lcom'], 'ca': iter_t_precision['ca'], 'ce': iter_t_precision['ce'],
         'npm': iter_t_precision['npm'], 'lcom3': iter_t_precision['lcom3'], 'dam': iter_t_precision['dam'],
         'moa': iter_t_precision['moa'], 'mfa': iter_t_precision['mfa'], "cam": iter_t_precision["cam"],
         "ic": iter_t_precision["ic"], "cbm": iter_t_precision["cbm"], "amc": iter_t_precision["amc"],
         "max_cc": iter_t_precision["max_cc"], "avg_cc": iter_t_precision["avg_cc"]}, ignore_index=True)
    # store recall
    df_alt = df_alt.append(
        {'iteration': str(i) + '_t_recall', 'last_version': name_previous, 'current_version': name[:-4],
         'Sample_size': len(df_test),
         'loc': iter_t_recall['loc'], 'wmc': iter_t_recall['wmc'], 'dit': iter_t_recall['dit'],
         'noc': iter_t_recall['noc'], 'cbo': iter_t_recall['cbo'], 'rfc': iter_t_recall['rfc'],
         'lcom': iter_t_recall['lcom'], 'ca': iter_t_recall['ca'], 'ce': iter_t_recall['ce'],
         'npm': iter_t_recall['npm'], 'lcom3': iter_t_recall['lcom3'], 'dam': iter_t_recall['dam'],
         'moa': iter_t_recall['moa'], 'mfa': iter_t_recall['mfa'], "cam": iter_t_recall["cam"],
         "ic": iter_t_recall["ic"], "cbm": iter_t_recall["cbm"], "amc": iter_t_recall["amc"],
         "max_cc": iter_t_recall["max_cc"], "avg_cc": iter_t_recall["avg_cc"]}, ignore_index=True)
    # store auc
    df_alt = df_alt.append(
        {'iteration': str(i) + '_t_auc', 'last_version': name_previous, 'current_version': name[:-4],
         'Sample_size': len(df_test),
         'loc': iter_t_auc['loc'], 'wmc': iter_t_auc['wmc'], 'dit': iter_t_auc['dit'],
         'noc': iter_t_auc['noc'], 'cbo': iter_t_auc['cbo'], 'rfc': iter_t_auc['rfc'],
         'lcom': iter_t_auc['lcom'], 'ca': iter_t_auc['ca'], 'ce': iter_t_auc['ce'],
         'npm': iter_t_auc['npm'], 'lcom3': iter_t_auc['lcom3'], 'dam': iter_t_auc['dam'],
         'moa': iter_t_auc['moa'], 'mfa': iter_t_auc['mfa'], "cam": iter_t_auc["cam"],
         "ic": iter_t_auc["ic"], "cbm": iter_t_auc["cbm"], "amc": iter_t_auc["amc"],
         "max_cc": iter_t_auc["max_cc"], "avg_cc": iter_t_auc["avg_cc"]}, ignore_index=True)
    # store auc_var
    df_alt = df_alt.append(
        {'iteration': str(i) + '_t_auc_var', 'last_version': name_previous, 'current_version': name[:-4],
         'Sample_size': len(df_test),
         'loc': iter_t_auc_var['loc'], 'wmc': iter_t_auc_var['wmc'], 'dit': iter_t_auc_var['dit'],
         'noc': iter_t_auc_var['noc'], 'cbo': iter_t_auc_var['cbo'], 'rfc': iter_t_auc_var['rfc'],
         'lcom': iter_t_auc_var['lcom'], 'ca': iter_t_auc_var['ca'], 'ce': iter_t_auc_var['ce'],
         'npm': iter_t_auc_var['npm'], 'lcom3': iter_t_auc_var['lcom3'], 'dam': iter_t_auc_var['dam'],
         'moa': iter_t_auc_var['moa'], 'mfa': iter_t_auc_var['mfa'], "cam": iter_t_auc_var["cam"],
         "ic": iter_t_auc_var["ic"], "cbm": iter_t_auc_var["cbm"], "amc": iter_t_auc_var["amc"],
         "max_cc": iter_t_auc_var["max_cc"], "avg_cc": iter_t_auc_var["avg_cc"]}, ignore_index=True)
    # store gm
    df_alt = df_alt.append(
        {'iteration': str(i) + '_t_gm', 'last_version': name_previous, 'current_version': name[:-4],
         'Sample_size': len(df_test),
         'loc': iter_t_gm['loc'], 'wmc': iter_t_gm['wmc'], 'dit': iter_t_gm['dit'],
         'noc': iter_t_gm['noc'], 'cbo': iter_t_gm['cbo'], 'rfc': iter_t_gm['rfc'],
         'lcom': iter_t_gm['lcom'], 'ca': iter_t_gm['ca'], 'ce': iter_t_gm['ce'],
         'npm': iter_t_gm['npm'], 'lcom3': iter_t_gm['lcom3'], 'dam': iter_t_gm['dam'],
         'moa': iter_t_gm['moa'], 'mfa': iter_t_gm['mfa'], "cam": iter_t_gm["cam"],
         "ic": iter_t_gm["ic"], "cbm": iter_t_gm["cbm"], "amc": iter_t_gm["amc"],
         "max_cc": iter_t_gm["max_cc"], "avg_cc": iter_t_gm["avg_cc"]}, ignore_index=True)
    # store f1
    df_alt = df_alt.append(
        {'iteration': str(i) + '_t_f1', 'last_version': name_previous, 'current_version': name[:-4],
         'Sample_size': len(df_test),
         'loc': iter_t_f1['loc'], 'wmc': iter_t_f1['wmc'], 'dit': iter_t_f1['dit'],
         'noc': iter_t_f1['noc'], 'cbo': iter_t_f1['cbo'], 'rfc': iter_t_f1['rfc'],
         'lcom': iter_t_f1['lcom'], 'ca': iter_t_f1['ca'], 'ce': iter_t_f1['ce'],
         'npm': iter_t_f1['npm'], 'lcom3': iter_t_f1['lcom3'], 'dam': iter_t_f1['dam'],
         'moa': iter_t_f1['moa'], 'mfa': iter_t_f1['mfa'], "cam": iter_t_f1["cam"],
         "ic": iter_t_f1["ic"], "cbm": iter_t_f1["cbm"], "amc": iter_t_f1["amc"],
         "max_cc": iter_t_f1["max_cc"], "avg_cc": iter_t_f1["avg_cc"]}, ignore_index=True)
    # store bpp
    df_alt = df_alt.append(
        {'iteration': str(i) + '_t_bpp', 'last_version': name_previous, 'current_version': name[:-4],
         'Sample_size': len(df_test),
         'loc': iter_t_bpp['loc'], 'wmc': iter_t_bpp['wmc'], 'dit': iter_t_bpp['dit'],
         'noc': iter_t_bpp['noc'], 'cbo': iter_t_bpp['cbo'], 'rfc': iter_t_bpp['rfc'],
         'lcom': iter_t_bpp['lcom'], 'ca': iter_t_bpp['ca'], 'ce': iter_t_bpp['ce'],
         'npm': iter_t_bpp['npm'], 'lcom3': iter_t_bpp['lcom3'], 'dam': iter_t_bpp['dam'],
         'moa': iter_t_bpp['moa'], 'mfa': iter_t_bpp['mfa'], "cam": iter_t_bpp["cam"],
         "ic": iter_t_bpp["ic"], "cbm": iter_t_bpp["cbm"], "amc": iter_t_bpp["amc"],
         "max_cc": iter_t_bpp["max_cc"], "avg_cc": iter_t_bpp["avg_cc"]}, ignore_index=True)
    # store fpr
    df_alt = df_alt.append(
        {'iteration': str(i) + '_t_fpr', 'last_version': name_previous, 'current_version': name[:-4],
         'Sample_size': len(df_test),
         'loc': iter_t_fpr['loc'], 'wmc': iter_t_fpr['wmc'], 'dit': iter_t_fpr['dit'],
         'noc': iter_t_fpr['noc'], 'cbo': iter_t_fpr['cbo'], 'rfc': iter_t_fpr['rfc'],
         'lcom': iter_t_fpr['lcom'], 'ca': iter_t_fpr['ca'], 'ce': iter_t_fpr['ce'],
         'npm': iter_t_fpr['npm'], 'lcom3': iter_t_fpr['lcom3'], 'dam': iter_t_fpr['dam'],
         'moa': iter_t_fpr['moa'], 'mfa': iter_t_fpr['mfa'], "cam": iter_t_fpr["cam"],
         "ic": iter_t_fpr["ic"], "cbm": iter_t_fpr["cbm"], "amc": iter_t_fpr["amc"],
         "max_cc": iter_t_fpr["max_cc"], "avg_cc": iter_t_fpr["avg_cc"]}, ignore_index=True)
    # store tnr
    df_alt = df_alt.append(
        {'iteration': str(i) + '_t_tnr', 'last_version': name_previous, 'current_version': name[:-4],
         'Sample_size': len(df_test),
         'loc': iter_t_tnr['loc'], 'wmc': iter_t_tnr['wmc'], 'dit': iter_t_tnr['dit'],
         'noc': iter_t_tnr['noc'], 'cbo': iter_t_tnr['cbo'], 'rfc': iter_t_tnr['rfc'],
         'lcom': iter_t_tnr['lcom'], 'ca': iter_t_tnr['ca'], 'ce': iter_t_tnr['ce'],
         'npm': iter_t_tnr['npm'], 'lcom3': iter_t_tnr['lcom3'], 'dam': iter_t_tnr['dam'],
         'moa': iter_t_tnr['moa'], 'mfa': iter_t_tnr['mfa'], "cam": iter_t_tnr["cam"],
         "ic": iter_t_tnr["ic"], "cbm": iter_t_tnr["cbm"], "amc": iter_t_tnr["amc"],
         "max_cc": iter_t_tnr["max_cc"], "avg_cc": iter_t_tnr["avg_cc"]}, ignore_index=True)
    # store er
    df_alt = df_alt.append(
        {'iteration': str(i) + '_t_er', 'last_version': name_previous, 'current_version': name[:-4],
         'Sample_size': len(df_test),
         'loc': iter_t_er['loc'], 'wmc': iter_t_er['wmc'], 'dit': iter_t_er['dit'],
         'noc': iter_t_er['noc'], 'cbo': iter_t_er['cbo'], 'rfc': iter_t_er['rfc'],
         'lcom': iter_t_er['lcom'], 'ca': iter_t_er['ca'], 'ce': iter_t_er['ce'],
         'npm': iter_t_er['npm'], 'lcom3': iter_t_er['lcom3'], 'dam': iter_t_er['dam'],
         'moa': iter_t_er['moa'], 'mfa': iter_t_er['mfa'], "cam": iter_t_er["cam"],
         "ic": iter_t_er["ic"], "cbm": iter_t_er["cbm"], "amc": iter_t_er["amc"],
         "max_cc": iter_t_er["max_cc"], "avg_cc": iter_t_er["avg_cc"]}, ignore_index=True)
    # store accuracy
    df_alt = df_alt.append(
        {'iteration': str(i) + '_t_accuracy', 'last_version': name_previous, 'current_version': name[:-4],
         'Sample_size': len(df_test),
         'loc': iter_t_accuracy['loc'], 'wmc': iter_t_accuracy['wmc'], 'dit': iter_t_accuracy['dit'],
         'noc': iter_t_accuracy['noc'], 'cbo': iter_t_accuracy['cbo'], 'rfc': iter_t_accuracy['rfc'],
         'lcom': iter_t_accuracy['lcom'], 'ca': iter_t_accuracy['ca'], 'ce': iter_t_accuracy['ce'],
         'npm': iter_t_accuracy['npm'], 'lcom3': iter_t_accuracy['lcom3'], 'dam': iter_t_accuracy['dam'],
         'moa': iter_t_accuracy['moa'], 'mfa': iter_t_accuracy['mfa'], "cam": iter_t_accuracy["cam"],
         "ic": iter_t_accuracy["ic"], "cbm": iter_t_accuracy["cbm"], "amc": iter_t_accuracy["amc"],
         "max_cc": iter_t_accuracy["max_cc"], "avg_cc": iter_t_accuracy["avg_cc"]}, ignore_index=True)
    # store error_rate
    df_alt = df_alt.append(
        {'iteration': str(i) + '_t_error_rate', 'last_version': name_previous, 'current_version': name[:-4],
         'Sample_size': len(df_test),
         'loc': iter_t_error_rate['loc'], 'wmc': iter_t_error_rate['wmc'], 'dit': iter_t_error_rate['dit'],
         'noc': iter_t_error_rate['noc'], 'cbo': iter_t_error_rate['cbo'], 'rfc': iter_t_error_rate['rfc'],
         'lcom': iter_t_error_rate['lcom'], 'ca': iter_t_error_rate['ca'], 'ce': iter_t_error_rate['ce'],
         'npm': iter_t_error_rate['npm'], 'lcom3': iter_t_error_rate['lcom3'], 'dam': iter_t_error_rate['dam'],
         'moa': iter_t_error_rate['moa'], 'mfa': iter_t_error_rate['mfa'], "cam": iter_t_error_rate["cam"],
         "ic": iter_t_error_rate["ic"], "cbm": iter_t_error_rate["cbm"], "amc": iter_t_error_rate["amc"],
         "max_cc": iter_t_error_rate["max_cc"], "avg_cc": iter_t_error_rate["avg_cc"]}, ignore_index=True)
    # store mcc
    df_alt = df_alt.append(
        {'iteration': str(i) + '_t_mcc', 'last_version': name_previous, 'current_version': name[:-4],
         'Sample_size': len(df_test),
         'loc': iter_t_mcc['loc'], 'wmc': iter_t_mcc['wmc'], 'dit': iter_t_mcc['dit'],
         'noc': iter_t_mcc['noc'], 'cbo': iter_t_mcc['cbo'], 'rfc': iter_t_mcc['rfc'],
         'lcom': iter_t_mcc['lcom'], 'ca': iter_t_mcc['ca'], 'ce': iter_t_mcc['ce'],
         'npm': iter_t_mcc['npm'], 'lcom3': iter_t_mcc['lcom3'], 'dam': iter_t_mcc['dam'],
         'moa': iter_t_mcc['moa'], 'mfa': iter_t_mcc['mfa'], "cam": iter_t_mcc["cam"],
         "ic": iter_t_mcc["ic"], "cbm": iter_t_mcc["cbm"], "amc": iter_t_mcc["amc"],
         "max_cc": iter_t_mcc["max_cc"], "avg_cc": iter_t_mcc["avg_cc"]}, ignore_index=True)

    # print(df_alt)
    return df_alt


# 分别应用pred_20_score的投票得分，用0.1,0.2,...,0.9 九种对应百分比分位数作为阈值分别计算分类性能
def predict_score_performance(df_test):
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix
    f1_1, gm_1, bpp_1, mcc_1, f1_2, gm_2, bpp_2, mcc_2, f1_3, gm_3, bpp_3, mcc_3, f1_4, gm_4, bpp_4, mcc_4, f1_5, gm_5, \
    bpp_5, mcc_5, f1_6, gm_6, bpp_6, mcc_6, f1_7, gm_7, bpp_7, mcc_7, f1_8, gm_8, bpp_8, mcc_8, f1_9, gm_9, bpp_9, \
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


def active_learning_mvs_middle(working_dir, result_dir):
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

    dir_spv = result_dir + 'spv_5_percent/'

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    if not os.path.exists(dir_spv):
        os.mkdir(dir_spv)

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
            df_als = pd.DataFrame(columns=['iteration', 'last_version', 'current_version', 'Sample_size', 'loc', 'wmc',
                                           'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'dam', 'moa',
                                           'mfa', 'cam', 'ic', 'cbm', 'amc', 'max_cc', 'avg_cc'], dtype=object)

            df_performance = pd.DataFrame(
                columns=['iteration', 'Sample_size', 'project', 'last_version', 'current_version',
                         'f1_0.1', 'gm_0.1', 'bpp_0.1', 'mcc_0.1', 'f1_0.2', 'gm_0.2', 'bpp_0.2', 'mcc_0.2',
                         'f1_0.3', 'gm_0.3', 'bpp_0.3', 'mcc_0.3', 'f1_0.4', 'gm_0.4', 'bpp_0.4', 'mcc_0.4',
                         'f1_0.5', 'gm_0.5', 'bpp_0.5', 'mcc_0.5', 'f1_0.6', 'gm_0.6', 'bpp_0.6', 'mcc_0.6',
                         'f1_0.7', 'gm_0.7', 'bpp_0.7', 'mcc_0.7', 'f1_0.8', 'gm_0.8', 'bpp_0.8', 'mcc_0.8',
                         'f1_0.9', 'gm_0.9', 'bpp_0.9', 'mcc_0.9'], dtype=object)

            # df_name for spv
            df_name = pd.read_csv(working_dir + project + '/' + name)

            # 与for循环最后df_name_previous = df_name互应，若第一个赋值给上一个，进入下一循环，否则在for循环最后一句把当前赋值给上一个
            if name == versions_sorted[0]:
                df_name_previous = df_name
                name_previous = name
                continue

            # to_be_deleted_metrics用每次迭代过程中GM最大值求阈值过程的GM最大值的最小值,该度量不参与下一次迭代，最多减少10个（一半）
            to_be_deleted_metrics = []

            print("The last and current releases respectively are ", name_previous, name)

            # bugBinary表示bug的二进制形式, 'iter':表示该行(模块)是否被选中，初始为零，表示都没选中,反之选中
            df_name_previous["bugBinary"] = df_name_previous.bug.apply(lambda x: 1 if x > 0 else 0)
            df_name["bugBinary"] = df_name.bug.apply(lambda x: 1 if x > 0 else 0)
            df_name_previous['iter'] = 1
            df_name['iter'] = 0

            # pred_20 存储20个度量应用中位数阈值比较之后的得分
            df_name['pred_20'] = 0

            # percent_1用于存储当前版本上5%,10%,15%,20%的模块数,用于后面迭代的次数,其中percent_05取0.5的模块数
            percent_5 = math.ceil(len(df_name) * 0.05)
            percent_10 = math.ceil(len(df_name) * 0.1)
            percent_15 = math.ceil(len(df_name) * 0.15)
            percent_20 = math.ceil(len(df_name) * 0.20)
            print(percent_5, percent_10, percent_15, percent_20)

            for i in range(percent_5):
                print("This is ", i, "th iteration.")
                for metric in metrics_20:

                    df_name = df_name[~df_name[metric].isin(['undef', 'undefined'])].reset_index(drop=True)
                    df_name = df_name[~df_name[metric].isnull()].reset_index(drop=True)
                    df_name = df_name[~df_name['loc'].isin([0])].reset_index(drop=True)

                    df_name_previous = df_name_previous[
                        ~df_name_previous[metric].isin(['undef', 'undefined'])].reset_index(drop=True)
                    df_name_previous = df_name_previous[~df_name_previous[metric].isnull()].reset_index(drop=True)
                    df_name_previous = df_name_previous[~df_name_previous['loc'].isin([0])].reset_index(drop=True)

                    # i=0 第一次迭代时，用于取每个度量的中位数作为初始阈值
                    if i == 0 or len(df_als) == 0:
                        # metric_t为每次迭代的阈值，初始为中位数阈值，迭代后，是上一次迭代的调整阈值
                        metric_t = df_name[metric].median()
                        print("iteration,  metric, and its threshold value are ", name, i, metric, metric_t)
                    else:
                        # read threshold derived from last iteration
                        metric_t = df_als[df_als['iteration'] == str(i - 1) + '_threshold'][metric].values[0]
                        print("iteration,  metric, and its threshold value are ", name, i, metric, metric_t)

                    # 若该度量在上一轮迭代中被删，则不参与投票
                    if metric_t == '/':
                        continue

                    # pred_20存储20个度量预测得分，最大值20，全预测有缺陷，最小值为0，全预测为无缺陷。此处假设所有度量都与缺陷正相关
                    df_name['pred_20'] = df_name.apply(
                        lambda x: x['pred_20'] + 1 if float(x[metric]) >= float(metric_t) else x['pred_20'] + 0, axis=1)

                # pred_20_score用于存储20个通用度量预测的得分再加上小数部分，小数部分等于当前模块的SLOC的倒数。
                df_name['pred_20_score'] = df_name.apply(lambda x: x['pred_20'] + (1 / x['loc']), axis=1)

                score_list = sorted(df_name[df_name['iter'] == 0]['pred_20_score'].tolist(), reverse=True)

                # 给本次迭代中主动学习选出的模块，iter度量赋值为1，若为0则表示未被选中，先将本次（iter==0）的模块的pred_20_score按
                # 从高到低排序，然后取中位数位置上那个模块，并将取到的模块iter赋值为1，若得分相同，则取第一个得分与中位数相同的模块
                for row in range(len(df_name)):
                    if df_name.loc[row, 'pred_20_score'] == score_list[math.ceil(len(score_list) * 0.5)]:
                        df_name.loc[row, 'iter'] = 1
                        break

                print('the number of selected modules is ', len(df_name[df_name['iter'] == 1]))

                # iter_bug用于存储伪标签(iter==0)，和真实标签(iter==1)
                df_name['iter_bug'] = df_name['bug']
                df_name_temp = df_name.loc[:, metrics_20 + ['bug', "bugBinary", 'pred_20', 'pred_20_score',
                                                            'iter', 'iter_bug']].copy()

                # df_iter应用主动学习的真实标签和前一版本所有标签训练每个度量阈值，然后在未主动学习的模块上预测性能
                df_iter = df_name_temp[df_name_temp['iter'] == 1].copy().reset_index(drop=True)
                print(len(df_iter))

                # 在未被选中的模块上，测试预测性能，同时保存最后一次迭代的数据，用与基线比较
                df_manual_labels = df_name_temp[df_name_temp['iter'] == 0].copy().reset_index(drop=True)
                # df_active = df_name_temp[df_name_temp['iter'] == 1].copy()

                label_threshold = df_manual_labels.pred_20_score.quantile(0.5)
                df_manual_labels['iter_bug'] = df_manual_labels.apply(
                    lambda x: 1 if x['pred_20_score'] >= label_threshold else 0, axis=1)
                # print(df_manual_labels)

                df_iter['iter_bug'] = df_iter.iter_bug.apply(lambda x: 1 if x > 0 else 0)
                # 用前一版本真实标签代入当前版本训练阈值
                df_name_previous['iter_bug'] = df_name_previous.bug.apply(lambda x: 1 if x > 0 else 0)

                # df_training用于合并当前主动学习的模块和前一版本的模块的真实标签，以及当前版本未选中的模块的伪标签
                df_training = df_iter.loc[:, metrics_20 + ['iter_bug']]
                df_training = df_training.append(df_manual_labels.loc[:, metrics_20 + ['iter_bug']]).reset_index(
                    drop=True)
                # df_training = df_training.append(df_name_previous.loc[:, metrics_20 + ['iter_bug']]).reset_index(
                #     drop=True)
                df_test = df_name_temp[df_name_temp['iter'] == 0].copy().reset_index(drop=True)

                # 应用pred_20_score的投票得分，计算CE值，再用0.1,0.2,...,0.9九种阈值分别计算分类性能
                f1_1, gm_1, bpp_1, mcc_1, f1_2, gm_2, bpp_2, mcc_2, f1_3, gm_3, bpp_3, mcc_3, f1_4, gm_4, bpp_4, mcc_4,\
                f1_5, gm_5, bpp_5, mcc_5, f1_6, gm_6, bpp_6, mcc_6, f1_7, gm_7, bpp_7, mcc_7, f1_8, gm_8, bpp_8, mcc_8,\
                f1_9, gm_9, bpp_9, mcc_9 = predict_score_performance(df_test)

                # 用主动学习模块，前一版本真实标签和当前版本伪标签模块，用MGM方法训练得到本次迭代的调整阈值，iter_t用于存储每次迭代
                # 各度量的阈值,iter_gm_max用于存储计算该阈值的GM最大值，在每次迭代时，从20个候选度量中删去GM最大值最小的那一个度量
                iter_t, iter_pearson, iter_gm_max, iter_f1_with_gm_max, iter_bpp_with_gm_max = {}, {}, {}, {}, {}

                for metric in metrics_20:

                    # 若度量在to_be_deleted_metrics列表中，则放弃统计得分
                    if metric in to_be_deleted_metrics:
                        iter_t[metric] = '/'
                        iter_pearson[metric] = '/'
                        iter_gm_max[metric] = '/'
                        iter_f1_with_gm_max[metric] = '/'
                        iter_bpp_with_gm_max[metric] = '/'
                        continue

                    pearson_t0, gm_t0, gm_max, f1_with_gm_max, bpp_with_gm_max = \
                        max_gm_threshold(df_training.loc[:, ['iter_bug', metric]].astype(float))
                    print(name, i, metric, pearson_t0, gm_t0, gm_max)
                    iter_t[metric] = gm_t0
                    iter_pearson[metric] = pearson_t0
                    iter_gm_max[metric] = gm_max
                    iter_f1_with_gm_max[metric] = f1_with_gm_max
                    iter_bpp_with_gm_max[metric] = bpp_with_gm_max

                # 存储本次迭代的调整阈值及计算阈值中gm最大化的gm值
                df_als = store_data(str(i) + '_threshold', name_previous, name[:-4], len(df_training), iter_t, df_als)

                # 存储本次迭代的计算阈值中gm最大化的gm值
                df_als = store_data(str(i) + '_gm_value', name_previous, name[:-4], len(df_training), iter_gm_max,
                                    df_als)

                # 存储本次迭代的计算阈值中gm最大化时F1值
                df_als = store_data(str(i) + '_f1_value', name_previous, name[:-4], len(df_training),
                                    iter_f1_with_gm_max, df_als)

                # 存储本次迭代的计算阈值中gm最大化时BPP值
                df_als = store_data(str(i) + '_bpp_value', name_previous, name[:-4], len(df_manual_labels),
                                    iter_bpp_with_gm_max, df_als)

                # 若通过GM最大化求出的阈值过程中,GM最大值比较小，说明该度量在当前版本上并不能很好的度量，至少在GM指标度量值来看，
                # 不适宜用当前度量用阈值来预测缺陷倾向。gm_max_list用于存储当前迭代过程中的gm的最大值
                # to_be_deleted_metrics中最多保留10个度量不再用于下一次迭代。
                gm_max_list = {}
                # print(iter_gm_max.values())
                for key, value in iter_gm_max.items():
                    # print(key, value)
                    if value != '/':
                        gm_max_list[key] = value
                # print(gm_max_list)
                for key, value in gm_max_list.items():
                    if (value == min(gm_max_list.values())) and len(to_be_deleted_metrics) <= 9:
                        print("the metric and it gm value which is the minimum value ", key, value)
                        to_be_deleted_metrics.append(key)

                # 计算本次迭代后的主动学习阈值在剩下未被专家检测的模块上的预测性能
                df_als = do_predction(i, name_previous, name, metrics_20, iter_pearson, iter_t,
                                      df_name_temp[df_name_temp['iter'] == 0].reset_index(drop=True), df_als)

                df_als.to_csv(dir_spv + 'activeLearningThreshold_' + name, index=False)

                df_performance = df_performance.append(
                    {'iteration': str(i) + '_threshold', 'Sample_size': len(df_test), 'project': project,
                     'last_version': name_previous[:-4], 'current_version': name[:-4], 'f1_0.1': f1_1, 'gm_0.1': gm_1,
                     'bpp_0.1': bpp_1, 'mcc_0.1': mcc_1, 'f1_0.2': f1_2, 'gm_0.2': gm_2, 'bpp_0.2': bpp_2,
                     'mcc_0.2': mcc_2, 'f1_0.3': f1_3, 'gm_0.3': gm_3, 'bpp_0.3': bpp_3, 'mcc_0.3': mcc_3,
                     'f1_0.4': f1_4, 'gm_0.4': gm_4, 'bpp_0.4': bpp_4, 'mcc_0.4': mcc_4, 'f1_0.5': f1_5, 'gm_0.5': gm_5,
                     'bpp_0.5': bpp_5, 'mcc_0.5': mcc_5, 'f1_0.6': f1_6, 'gm_0.6': gm_6, 'bpp_0.6': bpp_6,
                     'mcc_0.6': mcc_6, 'f1_0.7': f1_7, 'gm_0.7': gm_7, 'bpp_0.7': bpp_7, 'mcc_0.7': mcc_7,
                     'f1_0.8': f1_8, 'gm_0.8': gm_8, 'bpp_0.8': bpp_8, 'mcc_0.8': mcc_8, 'f1_0.9': f1_9, 'gm_0.9': gm_9,
                     'bpp_0.9': bpp_9, 'mcc_0.9': mcc_9}, ignore_index=True)

                df_performance.to_csv(dir_spv + 'activeLearningPerformance_' + name, index=False)

                # 每次迭代时，对上次未能被选中的模块V-score得分清零，
                df_name['pred_20'] = df_name.apply(lambda x: 0 if x['iter'] == 0 else x['pred_20'], axis=1)

            # 当前版本作为下一版本的前一版本
            df_name_previous = df_name.copy()
            name_previous = name

        break


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
    result_Directory = "F:/talcvdp/active_learning_expert_pseudo/"

    print(os.getcwd())
    os.chdir(work_Directory)
    print(work_Directory)
    print(os.getcwd())

    active_learning_mvs_middle(work_Directory, result_Directory)

    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ".\nFrom ", time.asctime(time.localtime(s_time)), " to ",
          time.asctime(time.localtime(e_time)), ",\nThis", os.path.basename(sys.argv[0]), "ended within",
          execution_time, "(s), or ", (execution_time / 60), " (m).")