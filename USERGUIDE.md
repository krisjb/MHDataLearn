# **MH Data Learn - User Guide**  

This is a set of simple functions to support identifying mental health patients who may be at risk of readmission. It includes tools to assist in standardising a dataset and then feeding in local data to a number of models which will determine variable selection and returns a best fit model based on local data.

Current data can then be fed to the model and patients identified to be at risk.

It is designed to be a predictor of current inpatients at risk of an unplanned readmission within 30 days of discharge.

This rate varies across trusts but is estimated to account for between 5% and 20% of admissions. This tool is deigned to help identify patients at risk so that appropriate packages of care may be put in place. As there is such variation across trusts, we have designed the package to work on your local data and build a local model, rather than a generic model.

> The package is designed to assist the clinical decision making process and help identify patients at risk so that appropriate care may be put in place.  Any model will come with a level of confidence and this should not override clinical knowledge, rather used as a tool to support it. 

## **Usage**

The package uses data based on standard **Mental Health Services Data Set (MHSDS)** schema.  In this way there are standardised data definitions and assumptions. 

The process runs in 4 steps

1. **Set up, Import and prepare historic training data**

2. **Feed data into model so that best model and features can be selected based on local dataset**

3. **Import and set up current inpatient data**

4. **Feed data into trained model and return with risk prediction to be developed in future**

## Test using dummy data

Below is an example usage of MHDataLearn. It uses a default MHSDS dataset (DummyData.csv) which is included in the package. The functions clean and preprocess the data, then train classification algorithms to predict patient emergency readmissions within 30 days of discharge from features within the dataset. Finally, model performance metrics are visualised.

```python
import MHDataLearn
from MHDataLearn.preprocessing import load, process
from MHDataLearn.modelselector.default_models import train_default_models

# load the default MHSDS dataset
df = load.load_data()

# preprocess the data in preparation for ml models
df = process.wrangle_data(df)

# split data in to test/train, train models and report performance metrics
models = train_default_models(df)
```



## Individual functions



## Sub-package preprocessing

## *load* 

### load_data(filepath)

This utilises pandas read csv function and file path can be set to a local drive or url where data is saved.  If left blank it will load in a dummy test set of data that has been created at random. Dummy data can be found at https://raw.githubusercontent.com/krisjb/MHDataLearn/main/Data/.

## *clean* 

### data_types(df)

Converts data in columns DOB, StartDateHospProvSpell and DischDateHospProvSpell to date format.

### gender_replace(df)

Replaces value 9 with 3 for ease of modelling.

### marital_replace(df) 

Converts marital status to numeric

| Field name    | Value         | Converts to |
| ------------- | ------------- | ----------- |
| MaritalStatus | S, D, W, P, N | 0           |
| MaritalStatus | M             | 1           |

### accom_replace(df)

Replaces values with numeric

| Field name               | Value | Converts to |
| ------------------------ | ----- | ----------- |
| SettledAccommodationInd  | Y     | 1           |
| .SettledAccommodationInd | N, Z  | 0           |

### employ_replace(df)

Replaces values with numeric

| Field name   | Value          | Converts to |
| ------------ | -------------- | ----------- |
| EmployStatus | 1,3            | 1           |
| EmployStatus | 2,4,5,6,7,8,ZZ | 0           |

### mhclass_replace(df)

Converts the mental health supercluster into numerical values

| Field name              | Value | Converts to |
| ----------------------- | ----- | ----------- |
| MHCareClusterSuperClass | A     | 1           |
| MHCareClusterSuperClass | B     | 2           |
| MHCareClusterSuperClass | C     | 3           |
| MHCareClusterSuperClass | Z     | 4           |

## *calculate*

### calc_age_admit(df)

Adds column to dataframe., calculates an age at admission - takes days between DOB and StartDateHospProvSpell divides by 365.2425 and rounds down to an integer.

### calc_readmit(df)

Checks on a patient by patient basis if there is a discharge date followed by another admission for that patient.  If there is an further admission with will return the date of that admission and the number of days between discharge and next admission.  It also provides a count of the number of admission spells.

### check_emergency(df)

Checks the *AdmMethCodeHospProvSpell* and if in 21,22,23,24,15,2A,2B,2D converts to a 1 else converts to a 0

| Field name                 | Value                   | Converts to |
| -------------------------- | ----------------------- | ----------- |
| *AdmMethCodeHospProvSpell* | 21,22,23,24,15,2A,2B,2D | 1           |
| *AdmMethCodeHospProvSpell* | Any other               | 0           |

### emergency_readmit(df)

Adds a marker where an unplanned readmission has occurred within 30 days of discharge.  Checks readmission from calc_readmit(df) and emergency from check_emergency(df) are both present.

### los_train(df)

Calculates number of days in spell from StartDateHospProvSpell to DischDateHospProvSpell

### los_current(df)

Calculates number of days in spell from StartDateHospProvSpell to current date

### age_check(df)

Checks age is present and within permitted bounds (16-110), otherwise replaces with a median age from the available data

### postcode_to_lsoa(df) 

Looks up Lower Layer Super Output Areas (LSOA)  from UK postcode and adds this column to dataframe   

*WARNING: CSV file used is 766MB so this may take time*

### lsoa_to_imd(df)

Looks up Index of Multiple Deprivation (IMD)  decile for Lower Layer Super Output Areas (LSOA)  and adds 'imd' column to dataframe.  Fills missing values with median imd.

### *process*

### wrangle_data(df, test= "training")

This function runs all the sub functions below in one go.  If the test is set to training it will calculate LOS based on the los_train function, it will also run the calc_readmit and emergency_readmit functions.  Otherwise it will run the los_current function and calculate LOS based on your current patients to today's date.

## *modelselector*

### default_models

### split_data(X, Y)

This function split the features and the target into training and test set.  

### reveal_best_classification_model(X_train, Y_train, X_test, Y_test)

This function build four classifcation models using four different Algorithms,
  -Logistic regression
  -Support Vector Machines
  -Decision Tree Classifier
  -K Nearest Neighbors
It runs a grid search cv through the models to determine the best hyper parameter and their best score, it compares the scores of all the four algorithms and creates a dictionary of the model with their respective best score and best parameter as per the gridsearch cv hyperparameter tuning during cross validations.

 ### create_prediction(model, x_test)

This creates prediction from the model

### plot_confusion_matrix(Y_test,y_predict, modelname)

This function plots the confusion matrix of the models.  This will identify true positives, true negatives, false positives and false negatives.  

### visualize_model_performance(model)

This function creates a bar plot with score of the individual models and their labels.

### train_default_models(df, imd_include=False)

Creates a list of default features and outcome variable (Emergency Readmission within 30 days of discharge).  Scales feature set using Standard Scaler    Splits data in to training and test sets (80/20).Trains classification models (Logistic Regression, Decision Tree and KNN models) and outputs confusion matrices plots and performance metrics in a data frame.

# Annex 1 Creating your own dataset



The recommended dataset is as follows. 

| Table             | Data Item                | Join on                                                     |
| ----------------- | ------------------------ | ----------------------------------------------------------- |
| MHS001MPI         | LocalPatientId           | LocalPatientId                                              |
| MHS001MPI         | PersonBirthDate          | LocalPatientId                                              |
| MHS001MPI         | Gender                   | LocalPatientId                                              |
| MHS001MPI         | EthnicCategory           | LocalPatientId                                              |
| MHS001MPI         | Postcode                 | LocalPatientId                                              |
| MHS001MPI         | MaritalStatus            | LocalPatientId                                              |
| MHS801            | MHCareClusterSuperClass  | LocalPatientId                                              |
| MHS501            | AdmMethCodeHospProvSpell |                                                             |
| MHS501            | PlannedDischDestCode     |                                                             |
| MHS501            | StartDateHospProvSpell   |                                                             |
| MHS501            | DischDateHospProvSpell   |                                                             |
| MHS502            | HospitalBedTypeMH        | EndDateWardStay = DischDateHospProvSpell & HospProvSpellNum |
| MHS003AccomStatus | SettledAccommodationInd  | LocalPatientId                                              |
| MHS004EmpStatus   | EmployStatus             | LocalPatientId                                              |

 **Suggested SQL** 

```
SELECT AdmMethCodeHospProvSpell,        
       PlannedDischDestCode,        
       StartDateHospProvSpell,              
       DischDateHospProvSpell,              
       HospitalBedTypeMH,       
       PA.LocalPatientId,              
       PersonBirthDate,              
       Gender,        
       EthnicCategory,        
       Postcode,          
       MaritalStatus,          
       HospitalBedTypeMH,           
       MHCareClusterSuperClass,              
       SettledAccommodationInd,             
       EmployStatus
FROM MHS801 AS PS         
LEFT JOIN MHS001MPI  AS PA   
	ON PA.LocalPatientId = PS.LocalPatientId               
LEFT JOIN MHS502 A`S WD      
	ON     WD.EndDateWardStay = PS.DischDateHospProvSpell            
      	   AND WD.HospProvSpellNum = PS.HospProvSpellNum           
LEFT JOIN MHS001MPI  AS PA 
	ON PA.LocalPatientId = PS.LocalPatientId               
LEFT JOIN MHS801 AS CL
	ON CL.LocalPatientId = PS.LocalPatientId               
LEFT JOIN MHS003AccomStatus AS AC
	ON AC.LocalPatientId = PS.LocalPatientId         
LEFT JOIN MHS004EmpStatus   AS EM
	ON EM.LocalPatientId = PS.LocalPatientId              
WHERE  DischDateHospProvSpell BETWEEN ‘2018-01-01' AND 2021-12-31'
---    use the below line to return current patients
---    DischDateHospProvSpell IS NULL
```

*Note*

You must include AdmMethCodeHospProvSpell as this is the indicator to show if an admission was planned or an emergency admission. You must also include provider spell  start  (StartDateHospProvSpell) and end dates (DischDateHospProvSpell).    It is recommended that you save the output as a csv file locally and read it in as follows:



# Annex 2 Defining your own feature list

 It is possible to add additional variables to the base dataset and these will be assessed.  It is recommended that these are kept to limited nominal data or continuous data items to maintain speed and reduce over complexity of the model. 

The basic data items pre chosen are based on this research

> [Osborn, D., Favarato, G., Lamb, D., Harper, T., Johnson, S.,  Lloyd-Evans, B., . . . Weich, S. (2021). Readmission after discharge  from acute mental healthcare among 231 988 people in England: Cohort  study exploring predictors of readmission including availability of  acute day units in local areas. *BJPsych Open,* *7*(4), E136. doi:10.1192/bjo.2021.961](https://www.cambridge.org/core/journals/bjpsych-open/article/readmission-after-discharge-from-acute-mental-healthcare-among-231-988-people-in-england-cohort-study-exploring-predictors-of-readmission-including-availability-of-acute-day-units-in-local-areas/A1EBFD2640641972BB865404C2EDE982)

If you wish to add your own features you can use the following

```
Y = df['columnname']
feature_list = ['columnname1', 'columnname2']
X = df[feature_list]
```

# Annex 3 Testing	

Navigate to MHDataLearn directory and run pytest in the command prompt. 
