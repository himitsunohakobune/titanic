
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from pandas import Series,DataFrame
#sns.set_style('whitegrid')
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,roc_auc_score

from logging import getLogger

#from load_data import load_train_data,load_test_data


TRAIN_DATA='../Porto-Seguro_check/train.csv'
TEST_DATA='../Porto-Seguro_check/test.csv'

logger = getLogger(__name__)

def read_csv(path):
    logger.debug('enter')
    df=pd.read_csv('path')
    logger.debug('exit')
    return df

def load_train_data():
    logger.debug('enter')
    df=pd.read_csv(TRAIN_DATA)
    logger.debug('exit')
    return df

def load_test_data():
    logger.debug('enter')
    df=pd.read_csv(TEST_DATA)
    logger.debug('exit')
    return df


logger=getLogger(__name__)

DIR='result_tmp/'
SAMPLE_SUBMIT_FILE='../Porto-Seguro_check/sample_submission.csv'

if __name__=='__main__':
    print(load_train_data().head())
    print(load_test_data().head())
    from logging import StreamHandler,DEBUG,Formatter,FileHandler,getLogger

    


    log_fmt=Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s')
    handler=StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler=FileHandler(DIR+'train.py.log','a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.info('start')

    df=load_train_data()
    
    x_train=df.drop('target',axis=1)
    y_train=df['target'].values

    use_cols=x_train.columns.values

    logger.info('train columns:{} {}'.format(use_cols.shape,use_cols))

    logger.info('data preparation end {}'.format(x_train.shape))


    clf=LogisticRegression(random_state=0)
    clf.fit(x_train,y_train)

    logger.info('train end')
    df=load_test_data()

    x_test=df[use_cols].sort_values('id')
    logger.info('test data load end {}'.format(x_train.shape))
    pred_test=clf.predict_proba(x_test)

    df_submit=pd.read_csv(SAMPLE_SUBMIT_FILE)
    df_submit['target']=pred_test

    df_submit.to_csv(DIR+'submit2.csv',index=False)

    








