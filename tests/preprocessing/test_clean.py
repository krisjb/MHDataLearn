import pandas as pd
# import numpy as np
# from datetime import timedelta
# from datetime import datetime
from MHDataLearn.preprocessing.clean import age_check, \
                                            gender_replace, \
                                            marital_replace, \
                                            accom_replace, \
                                            employ_replace, \
                                            mhclass_replace


def test_gender_replace():
    test_df = pd.read_csv('https://raw.githubusercontent.com/krisjb/'
                          'MHDataLearn/main/Data/DummyData.csv')
    test_df = gender_replace(test_df)
    gender_uniquevalues = test_df["Gender"].nunique(dropna=True)
    assert gender_uniquevalues == 3


def test_marital_replace():
    test_df = pd.read_csv('https://raw.githubusercontent.com/krisjb/'
                          'MHDataLearn/main/Data/DummyData.csv')
    test_df = marital_replace(test_df)
    marital_uniquevalues = test_df["MaritalStatus"].nunique(dropna=True)
    assert marital_uniquevalues == 2


def test_accom_replace():
    test_df = pd.read_csv('https://raw.githubusercontent.com/krisjb/'
                          'MHDataLearn/main/Data/DummyData.csv')
    test_df = accom_replace(test_df)
    accom_uniquevalues = test_df["SettledAccommodationInd"].nunique(dropna=True)
    assert accom_uniquevalues == 2


def test_employ_replace():
    test_df = pd.read_csv('https://raw.githubusercontent.com/krisjb/'
                          'MHDataLearn/main/Data/DummyData.csv')
    test_df = employ_replace(test_df)
    employ_uniquevalues = test_df["EmployStatus"].nunique(dropna=True)
    assert employ_uniquevalues == 2


def test_mhclass_replace():
    test_df = pd.read_csv('https://raw.githubusercontent.com/krisjb/'
                          'MHDataLearn/main/Data/DummyData.csv')
    test_df = mhclass_replace(test_df)
    mhclass_uniquevalues = test_df["MHCareClusterSuperClass"].nunique(dropna=True)
    assert mhclass_uniquevalues == 4

                                            









# class TestClean(unittest.TestCase):
#     """
#     This class provides test for all the functions inside the clean module.

#     """

#     def test_age_check1(self):
#         """
#         This is a test for the function age_check
#         """
#         unique_age = df["age_admit"].unique(dropna=False)
#         self.assertGreaterEqual(a in unique_age, 16)

#     def test_age_check2(self):
#         """
#         This is a test for the function age_check
#         """
#         unique_age = df["age_admit"].unique(dropna=False)
#         self.assertLessEqual(a in unique_age, 110)

#     def test_gender_replace(self):
#         """
#         This is a test for the function gender_replace
#         """
#         gender_uniquevalues = df["Gender"].nunique(dropna=True)
#         self.assertEqual(gender_uniquevalues, 3)

#     def test_marital_replace(self):
#         """
#         This is a test for the function marital_replace
#         """
#         marital_uniquevalues = df["MaritalStatus"].nunique(dropna=True)
#         self.assertEqual(marital_uniquevalues, 2)
    
#     def test_accom_replace(self):
#         """
#         This is a test for the function accom_replace
#         """
#         accom_uniquevalues = df["SettledAccommodationInd"].nunique(dropna=True)
#         self.assertEqual(accom_uniquevalues, 2)
    
#     def test_employ_replace(self):
#         """
#         This is a test for the function employ_replace
#         """
#         employ_uniquevalues = df["EmployStatus"].nunique(dropna=True)
#         self.assertEqual(employ_uniquevalues, 2)