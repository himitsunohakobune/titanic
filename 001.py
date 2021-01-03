
import numpy as np
import pandas as pd
from tqdm import tqdm
#import matplotlib.pyplot as plt
#import seaborn as sns
from pandas import Series, DataFrame
# sns.set_style('whitegrid')
import math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, auc
from logging import getLogger

#from load_data import load_train_data,load_test_data


TRAIN_DATA = '../Porto-Seguro_check/train.csv'
TEST_DATA = '../Porto-Seguro_check/test.csv'

logger = getLogger(__name__)


def read_csv(path):
    logger.debug('enter')
    df = pd.read_csv('path')
    logger.debug('exit')
    return df


def load_train_data():
    logger.debug('enter')
    df = pd.read_csv(TRAIN_DATA)
    logger.debug('exit')
    return df


def load_test_data():
    logger.debug('enter')
    df = pd.read_csv(TEST_DATA)
    logger.debug('exit')
    return df


logger = getLogger(__name__)

DIR = 'result_tmp/'
SAMPLE_SUBMIT_FILE = '../Porto-Seguro_check/sample_submission.csv'


def gini(y, pred):
    fpr, tpr, thr = roc_curve(y, pred, pos_label=1)
    g = 2*auc(fpr, tpr)-1
    return g


if __name__ == '__main__':
    print(load_train_data().head())
    print(load_test_data().head())
    from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

    log_fmt = Formatter(
        '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(DIR+'train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    df = load_train_data()

    x_train = df.drop('target', axis=1)
    y_train = df['target'].values

    use_cols = x_train.columns.values

    logger.info('train columns:{} {}'.format(use_cols.shape, use_cols))

    logger.info('data preparation end {}'.format(x_train.shape))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    # Cは正則化項、fit_interceptは切片を使うか、
    all_params = {'C': [10**i for i in range(-1, 2)],
                  'fit_intercept': [True, False],
                  'solver': ['liblinear'],
                  'penalty': ['l2', 'l1'],
                  'random_state': [0]}

    min_score = 100
    min_params = None

    for params in tqdm(list(ParameterGrid(all_params))):
        logger.info('params:{}'.format(params))

        list_logloss_score = []
        list_auc_score = []
        list_gini_score=[]

        for train_idx, valid_idx in cv.split(x_train, y_train):
            trn_x = x_train.iloc[train_idx, :]
            val_x = x_train.iloc[valid_idx, :]

            trn_y = y_train[train_idx]
            val_y = y_train[valid_idx]

            # **でディクショナリをキーワードargsとして渡せる
            clf = LogisticRegression(**params)
            clf.fit(trn_x, trn_y)
            pred = clf.predict_proba(val_x)[:, 1]
            sc_logloss = log_loss(val_y, pred)
            sc_auc = roc_auc_score(val_y, pred)
            sc_gini=gini(val_y,pred)

            list_logloss_score.append(sc_logloss)
            list_auc_score.append(sc_auc)
            list_gini_score.append(sc_gini)


            logger.debug('   松logloss:{},auc:{},gini:{}'.format(sc_logloss, sc_auc,sc_gini))
            
            break

        sc_logloss = np.mean(list_logloss_score)
        sc_auc = np.mean(list_auc_score)
        sc_gini = np.mean(list_gini_score)
        logger.info('浜田min_score:{},gini:{}'.format(min_score,sc_gini))

        if min_score > sc_gini:
            min_score = sc_gini
            print("min_scoreがsc_miniより大きいのでif文に入っています")
            print("その１",sc_gini)
            print("その２",min_score)
            min_params = params
        print("上島",sc_gini)
        print("肥後",min_score)
        logger.info('竹logloss:{},auc:{},gini:{}'.format(sc_logloss, sc_auc,sc_gini))     
        logger.info('梅current min score:{},params:{}'.format(min_score, min_params))

    logger.info('minimum params:{}'.format(min_params))
    logger.info('minimum auc:{}'.format(min_score))
    logger.info('minimum gini:{}'.format(min_score))


    clf = LogisticRegression(random_state=0)
    clf.fit(x_train, y_train)

    logger.info('train end')
    df = load_test_data()

    x_test = df[use_cols].sort_values('id')
    logger.info('test data load end {}'.format(x_train.shape))
    pred_test = clf.predict_proba(x_test)[:, 1]

    df_submit = pd.read_csv(SAMPLE_SUBMIT_FILE)
    df_submit['target'] = pred_test

    df_submit.to_csv(DIR+'submit7.csv', index=False)
    logger.info('end')
