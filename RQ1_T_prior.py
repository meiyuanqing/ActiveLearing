#!/usr/bin/env python
# encoding:utf-8
"""
Project: Object-oriented-Metric-Thresholds
File: RQ1_T_prior.py
Date : 2022/11/16 9:26
Author : njumy
HomePage : https://github.com/meiyuanqing
Email : dg1533019@smail.nju.edu.cn

本方法没有主动学习，仅用前一版本的训练各度量的阈值，然后在当前版本上用投票方法得出预测性能。

"""


# output: the threshold derived from MGM
# note that the dataframe should input astype(float), i.e., MGM_threshold(df.astype(float))
def max_gm_threshold(df):
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix

    metric_name = ''
    for col in df.columns:
        if col not in ['bug', 'bugBinary']:
            metric_name = col

    # 2. 依次用该度量每一个值作为阈值计算出各自预测性能值,然后选择预测性能值最大的作为阈值,分别定义存入list,取最大值和最大值的下标值
    # 同时输出gm最大值阈值的F1和BPP值。
    GMs = []
    gm_max_value = 0
    f1_with_gm_max = 0
    bpp_with_gm_max = 0
    i_gm_max = 0

    # 判断每个度量与bug之间的关系,用于阈值判断正反例
    Corr_metric_bug = df.loc[:, [metric_name, 'bugBinary']].corr('spearman')

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
        c_matrix = confusion_matrix(df["bugBinary"], df['predictBinary'], labels=[0, 1])
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
        recall_value = recall_score(df['bugBinary'], df['predictBinary'], labels=[0, 1])
        # precision_value = precision_score(df['bugBinary'], df['predictBinary'], labels=[0, 1])
        f1_value = f1_score(df['bugBinary'], df['predictBinary'], labels=[0, 1])

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
    # df_als = df_als.append(
    #     {'iteration': iteration, 'last_version': last_version, 'current_version': current_version,
    #      'Sample_size': Sample_size,
    #      'LCOM1': iter_t['LCOM1'], 'LCOM2': iter_t['LCOM2'], 'LCOM3': iter_t['LCOM3'], 'ICH': iter_t['ICH'],
    #      'NHD': iter_t['NHD'], 'OCAIC': iter_t['OCAIC'], 'OCMIC': iter_t['OCMIC'], 'OMMIC': iter_t['OMMIC'],
    #      'CBO': iter_t['CBO'], 'DAC': iter_t['DAC'], 'ICP': iter_t['ICP'], 'MPC': iter_t['MPC'],
    #      'NIHICP': iter_t['NIHICP'], 'RFC': iter_t['RFC'], "NMA": iter_t["NMA"], "NA": iter_t["NA"],
    #      "NAIMP": iter_t["NAIMP"], "NM": iter_t["NM"], "NMIMP": iter_t["NMIMP"], "NumPara": iter_t["NumPara"],
    #      "SLOC": iter_t["SLOC"], "stms": iter_t["stms"]}, ignore_index=True)

    df_als = df_als.append(
        {'iteration': iteration, 'last_version': last_version, 'current_version': current_version,
         'Sample_size': Sample_size, 'loc': iter_t['loc'], 'wmc': iter_t['wmc'], 'dit': iter_t['dit'],
         'noc': iter_t['noc'], 'cbo': iter_t['cbo'], 'rfc': iter_t['rfc'], 'lcom': iter_t['lcom'], 'ca': iter_t['ca'],
         'ce': iter_t['ce'], 'npm': iter_t['npm'], 'lcom3': iter_t['lcom3'], 'dam': iter_t['dam'], 'moa': iter_t['moa'],
         'mfa': iter_t['mfa'], "cam": iter_t["cam"], "ic": iter_t["ic"], "cbm": iter_t["cbm"], "amc": iter_t["amc"],
         "max_cc": iter_t["avg_cc"], "avg_cc": iter_t["avg_cc"]}, ignore_index=True)

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
                s += df_test.loc[j, 'SLOC']
                s_p += df_test.loc[j, 'SLOC'] * 1
                f += df_test.loc[j, 'bug']
                f_p += df_test.loc[j, 'bug'] * 1
            else:
                df_test.loc[j, 'predictBinary'] = 0
                s += df_test.loc[j, 'SLOC']
                s_p += df_test.loc[j, 'SLOC'] * 0
                f += df_test.loc[j, 'bug']
                f_p += df_test.loc[j, 'bug'] * 0
    else:
        for j in range(len(df_test)):
            if float(df_test.loc[j, metric]) >= metric_t:
                df_test.loc[j, 'predictBinary'] = 1
                s += df_test.loc[j, 'SLOC']
                s_p += df_test.loc[j, 'SLOC'] * 1
                f += df_test.loc[j, 'bug']
                f_p += df_test.loc[j, 'bug'] * 1
            else:
                df_test.loc[j, 'predictBinary'] = 0
                s += df_test.loc[j, 'SLOC']
                s_p += df_test.loc[j, 'SLOC'] * 0
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

    if metric == 'SLOC':
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
    df_alt = store_data(str(i) + '_t_precision', name_previous, name[:-4], len(df_test), iter_t_precision, df_alt)

    # store recall
    df_alt = store_data(str(i) + '_t_recall', name_previous, name[:-4], len(df_test), iter_t_recall, df_alt)

    # store auc
    df_alt = store_data(str(i) + '_t_auc', name_previous, name[:-4], len(df_test), iter_t_auc, df_alt)

    # store auc_var
    df_alt = store_data(str(i) + '_t_auc_var', name_previous, name[:-4], len(df_test), iter_t_auc_var, df_alt)

    # store gm
    df_alt = store_data(str(i) + '_t_gm', name_previous, name[:-4], len(df_test), iter_t_gm, df_alt)

    # store f1
    df_alt = store_data(str(i) + '_t_f1', name_previous, name[:-4], len(df_test), iter_t_f1, df_alt)

    # store bpp
    df_alt = store_data(str(i) + '_t_bpp', name_previous, name[:-4], len(df_test), iter_t_bpp, df_alt)

    # store fpr
    df_alt = store_data(str(i) + '_t_fpr', name_previous, name[:-4], len(df_test), iter_t_fpr, df_alt)

    # store tnr
    df_alt = store_data(str(i) + '_t_tnr', name_previous, name[:-4], len(df_test), iter_t_tnr, df_alt)

    # store er
    df_alt = store_data(str(i) + '_t_er', name_previous, name[:-4], len(df_test), iter_t_er, df_alt)

    # store accuracy
    df_alt = store_data(str(i) + '_t_accuracy', name_previous, name[:-4], len(df_test), iter_t_accuracy, df_alt)

    # store error_rate
    df_alt = store_data(str(i) + '_t_error_rate', name_previous, name[:-4], len(df_test), iter_t_error_rate, df_alt)

    # store mcc
    df_alt = store_data(str(i) + '_t_mcc', name_previous, name[:-4], len(df_test), iter_t_mcc, df_alt)

    # print(df_alt)
    return df_alt


# 分别应用pred_22_score的投票得分，用0.1,0.2,...,0.9 九种对应百分比分位数作为阈值分别计算分类性能
def predict_score_performance(df_test):
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix
    f1_1, gm_1, bpp_1, mcc_1, f1_2, gm_2, bpp_2, mcc_2, f1_3, gm_3, bpp_3, mcc_3, f1_4, gm_4, bpp_4, mcc_4, f1_5, gm_5, \
    bpp_5, mcc_5, f1_6, gm_6, bpp_6, mcc_6, f1_7, gm_7, bpp_7, mcc_7, f1_8, gm_8, bpp_8, mcc_8, f1_9, gm_9, bpp_9, \
    mcc_9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    for i in range(9):

        df_predict = df_test.copy()

        for j in range(len(df_predict)):
            if df_predict.loc[j, 'pred_22_score'] >= df_predict.pred_22_score.quantile(0.1 * (i + 1)):
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


def only_prior_threshold_spv(working_dir, result_dir):
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

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # 20 object-oriented features and a dependent variable:  'bug'
    metrics_20 = ['loc', 'wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'dam', 'moa', 'mfa',
                  'cam', 'ic', 'cbm', 'amc', 'max_cc', 'avg_cc']

    # store the results of active_learning_semisupervised：'f1_value', 'gm_value', 'bpp_value'
    df_als = pd.DataFrame(columns=['iteration', 'last_version', 'current_version', 'Sample_size', 'loc', 'wmc',
                                   'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'dam', 'moa',
                                   'mfa', 'cam', 'ic', 'cbm', 'amc', 'max_cc', 'avg_cc'], dtype=object)

    df_performance = pd.DataFrame(
        columns=['project', 'last_version', 'current_version', 'Sample_size',
                 'f1_0.1', 'gm_0.1', 'bpp_0.1', 'mcc_0.1', 'f1_0.2', 'gm_0.2', 'bpp_0.2', 'mcc_0.2',
                 'f1_0.3', 'gm_0.3', 'bpp_0.3', 'mcc_0.3', 'f1_0.4', 'gm_0.4', 'bpp_0.4', 'mcc_0.4',
                 'f1_0.5', 'gm_0.5', 'bpp_0.5', 'mcc_0.5', 'f1_0.6', 'gm_0.6', 'bpp_0.6', 'mcc_0.6',
                 'f1_0.7', 'gm_0.7', 'bpp_0.7', 'mcc_0.7', 'f1_0.8', 'gm_0.8', 'bpp_0.8', 'mcc_0.8',
                 'f1_0.9', 'gm_0.9', 'bpp_0.9', 'mcc_0.9'], dtype=object)

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

            # df_name for spv
            df_name = pd.read_csv(working_dir + project + '/' + name)

            # 与for循环最后df_name_previous = df_name互应，若第一个赋值给上一个，进入下一循环，否则在for循环最后一句把当前赋值给上一个
            if name == versions_sorted[0]:
                df_name_previous = df_name
                name_previous = name
                continue

            print("The last and current releases respectively are ", name_previous, name)

            # bugBinary表示bug的二进制形式, 'iter':表示该行(模块)是否被选中，初始为零，表示都没选中,反之选中
            df_name_previous["bugBinary"] = df_name_previous.bug.apply(lambda x: 1 if x > 0 else 0)
            df_name["bugBinary"] = df_name.bug.apply(lambda x: 1 if x > 0 else 0)

            # pred_22 存储22个度量应用中位数阈值比较之后的得分
            df_name['pred_22'] = 0

            # store thresholds of prior version
            iter_t, iter_pearson, iter_gm_max, iter_f1_with_gm_max, iter_bpp_with_gm_max = {}, {}, {}, {}, {}

            for metric in metrics_20:
                # print("the current file is ", name, "the current metric is ", metric)
                df_name = df_name[~df_name[metric].isin(['undef', 'undefined'])].reset_index(drop=True)
                df_name = df_name[~df_name[metric].isnull()].reset_index(drop=True)
                df_name = df_name[~df_name['loc'].isin([0])].reset_index(drop=True)

                df_name_previous = df_name_previous[~df_name_previous[metric].isin(['undef', 'undefined'])].reset_index(
                    drop=True)
                df_name_previous = df_name_previous[~df_name_previous[metric].isnull()].reset_index(drop=True)
                df_name_previous = df_name_previous[~df_name_previous['loc'].isin([0])].reset_index(drop=True)

                # metric_t为前一版本的同一度量，应用mgm方法计算的阈值
                pearson_t0, gm_t0, gm_max, f1_with_gm_max, bpp_with_gm_max = \
                    max_gm_threshold(df_name_previous.loc[:, ['bugBinary', metric]].astype(float))

                iter_t[metric] = gm_t0
                iter_pearson[metric] = pearson_t0
                iter_gm_max[metric] = gm_max
                iter_f1_with_gm_max[metric] = f1_with_gm_max
                iter_bpp_with_gm_max[metric] = bpp_with_gm_max

                metric_t = gm_t0
                # metric_t = df_name[metric].median()
                print("version, metric, and its threshold value are ", name_previous, name, metric, metric_t)

                # pred_22用于存储22个度量预测得分，最大值为22，全预测有缺陷，最小值为0，全预测为无缺陷。此处度量都与缺陷正相关
                df_name['pred_22'] = df_name.apply(lambda x: x['pred_22'] + 1
                if float(x[metric]) >= float(metric_t) else x['pred_22'] + 0, axis=1)

            # pred_22_score用于存储22个度量预测的得分再加上小数部分，小数部分等于当前模块的SLOC的倒数。
            df_name['pred_22_score'] = df_name.apply(lambda x: x['pred_22'] + (1 / x['loc']), axis=1)

            # 应用pred_22_score的投票得分，计算CE值，再用0.1,0.2,...,0.9九种阈值分别计算分类性能
            f1_1, gm_1, bpp_1, mcc_1, f1_2, gm_2, bpp_2, mcc_2, f1_3, gm_3, bpp_3, mcc_3, f1_4, gm_4, bpp_4, mcc_4, \
            f1_5, gm_5, bpp_5, mcc_5, f1_6, gm_6, bpp_6, mcc_6, f1_7, gm_7, bpp_7, mcc_7, f1_8, gm_8, bpp_8, mcc_8, \
            f1_9, gm_9, bpp_9, mcc_9 = predict_score_performance(df_name)

            df_performance = df_performance.append(
                {'project': project, 'last_version': name_previous, 'current_version': name[:-4],
                 'Sample_size': len(df_name),
                 'f1_0.1': f1_1, 'gm_0.1': gm_1, 'bpp_0.1': bpp_1, 'mcc_0.1': mcc_1, 'f1_0.2': f1_2, 'gm_0.2': gm_2,
                 'bpp_0.2': bpp_2, 'mcc_0.2': mcc_2, 'f1_0.3': f1_3, 'gm_0.3': gm_3, 'bpp_0.3': bpp_3, 'mcc_0.3': mcc_3,
                 'f1_0.4': f1_4, 'gm_0.4': gm_4, 'bpp_0.4': bpp_4, 'mcc_0.4': mcc_4, 'f1_0.5': f1_5, 'gm_0.5': gm_5,
                 'bpp_0.5': bpp_5, 'mcc_0.5': mcc_5, 'f1_0.6': f1_6, 'gm_0.6': gm_6, 'bpp_0.6': bpp_6, 'mcc_0.6': mcc_6,
                 'f1_0.7': f1_7, 'gm_0.7': gm_7, 'bpp_0.7': bpp_7, 'mcc_0.7': mcc_7, 'f1_0.8': f1_8, 'gm_0.8': gm_8,
                 'bpp_0.8': bpp_8, 'mcc_0.8': mcc_8, 'f1_0.9': f1_9, 'gm_0.9': gm_9, 'bpp_0.9': bpp_9,
                 'mcc_0.9': mcc_9}, ignore_index=True)

            df_performance.to_csv(result_dir + 'prior_threshold_voting_prediction_on_current_versions.csv', index=False)

            # 存储本次迭代的调整阈值及计算阈值中gm最大化的gm值
            df_als = store_data('prior_threshold', name_previous, name[:-4], len(df_name_previous), iter_t, df_als)

            # 存储本次迭代的计算阈值中gm最大化的gm值
            df_als = store_data('prior_gm_value', name_previous, name[:-4], len(df_name_previous), iter_gm_max, df_als)

            # 存储本次迭代的计算阈值中gm最大化时F1值
            df_als = store_data('prior_f1_value', name_previous, name[:-4], len(df_name_previous), iter_f1_with_gm_max,
                                df_als)

            # 存储本次迭代的计算阈值中gm最大化时BPP值
            df_als = store_data('prior_bpp_value', name_previous, name[:-4], len(df_name_previous),
                                iter_bpp_with_gm_max, df_als)

            df_als.to_csv(result_dir + 'prior_threshold_on_current_versions.csv', index=False)

            # 当前版本作为下一版本的前一版本
            df_name_previous = df_name.copy()
            name_previous = name


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
    result_Directory = "F:/talcvdp/only_prior_threshold_on_current_release/"

    print(os.getcwd())
    os.chdir(work_Directory)
    print(work_Directory)
    print(os.getcwd())

    only_prior_threshold_spv(work_Directory, result_Directory)

    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ".\nFrom ", time.asctime(time.localtime(s_time)), " to ",
          time.asctime(time.localtime(e_time)), ",\nThis", os.path.basename(sys.argv[0]), "ended within",
          execution_time, "(s), or ", (execution_time / 60), " (m).")
