import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime
from MHDataLearn.preprocessing.calculate import calc_age_admit,\
                                                calc_readmit,\
                                                check_emergency,\
                                                emergency_readmit,\
                                                postcode_to_lsoa,\
                                                lsoa_to_imd,\
                                                los_train
from MHDataLearn.preprocessing.clean import data_types


def test_calc_age_admit():
    test_df = pd.read_csv('https://raw.githubusercontent.com/krisjb/'\
                          'MHDataLearn/main/Data/DummyData.csv')
    test_df = data_types(test_df)
    test_df = calc_age_admit(test_df)
    assert 'age_admit' in test_df.columns
    

def test_check_emergency1():
    test_df = pd.read_csv('https://raw.githubusercontent.com/krisjb/'\
                          'MHDataLearn/main/Data/DummyData.csv')
    test_df = data_types(test_df)
    test_df = check_emergency(test_df)
    assert 'Emergency' in test_df.columns


def test_check_emergency2():
    test_df = pd.read_csv('https://raw.githubusercontent.com/krisjb/'\
                          'MHDataLearn/main/Data/DummyData.csv')
    test_df = data_types(test_df)
    test_df = check_emergency(test_df)
    column = test_df['Emergency']
    max_value = column.max()
    assert max_value == 1
    

def test_calc_readmit1():
    test_df = pd.read_csv('https://raw.githubusercontent.com/krisjb/'\
                          'MHDataLearn/main/Data/DummyData.csv')
    test_df = data_types(test_df)
    test_df = calc_readmit(test_df)
    assert 'days_since_admit' in test_df.columns
    

def test_calc_readmit2():
    test_df = pd.read_csv('https://raw.githubusercontent.com/krisjb/'\
                          'MHDataLearn/main/Data/DummyData.csv')
    test_df = data_types(test_df)
    test_df = calc_readmit(test_df)
    dtype = test_df['days_since_admit'].dtype
    assert dtype == 'float64'
    

def test_emergency_readmit():
    test_df = pd.read_csv('https://raw.githubusercontent.com/krisjb/'\
                          'MHDataLearn/main/Data/DummyData.csv')
    test_df = data_types(test_df)
    test_df = calc_readmit(test_df)
    test_df = check_emergency(test_df)
    test_df = emergency_readmit(test_df)
    assert 'EmergencyReadmit' in test_df.columns


def test_los_train():
    test_df = pd.read_csv('https://raw.githubusercontent.com/krisjb/'\
                          'MHDataLearn/main/Data/DummyData.csv')
    test_df = data_types(test_df)
    test_df = los_train(test_df)
    dtype = test_df['len_stay'].dtype
    assert dtype == 'timedelta64[ns]'