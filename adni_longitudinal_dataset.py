
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KernelDensity
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, KFold
import plotly.graph_objects as go
import fnmatch
import seaborn as sns
import torch
import sys
import pickle 
from functions import select_baseline_data

data_dir = './data/'

def create_baseline_temporal_dataset(subjects, dataframe, dataframeunnorm, target, features, visualize=False):
    '''
    subjects: list of the subject ids
    dataframe: dataframe with all the data
    target: H_MUSE ROI features
    '''
    print('Target', target)
    cnt = 0
    num_samples = 0
    list_of_subjects, list_of_subject_ids = [], []
    data_x, data_y, data_xbase = [], [], []

    samples = {'PTID': [], 'X': [], 'Y': []}
    covariates = {'PTID': [], 'MRI_Scanner_Model':[], 'Age': [], 'BaselineDiagnosis': []}

    total_cognitive_scores = ['MMSE_nearest_2.0', 'ADAS_COG_11', 'ADAS_COG_13', 'ADNI_EF', 'ADNI_LAN', 'ADNI_MEM']


    longitudinal_covariates = {'PTID': [], 'Time': [], 'Age': [],  'Diagnosis': [], 'Hypertension': [],
                               'Diabetes': [], 'DLICV': [], 'Study': [], 'Education_Years': [], 'Race': [] , 'SPARE_BA': [], 'SPARE_AD': [], 
                               'Sex': [], 'MMSE_nearest_2.0': [], 'ADAS_COG_11': [], 'ADAS_COG_13': [], 'ADNI_EF': [], 'ADNI_LAN': [], 'ADNI_MEM': []}
    comorbidities = ['Hypertension', 'Diabetes']

    if visualize:
        vdata = {'target': [], 'class': [], 'time': [], 'id': []}
        cnt = 0

    # remove the PTID from the features!
    features.remove('PTID')
    features.remove('Delta_Baseline')
    features.remove('Time')
    # hmuse = [i for i in features if i.startswith('H_MUSE')]

    # print('Features', features)
    clinical_features = [f for f in features if not f.startswith('H_MUSE')]
    # print('Clinical Features', clinical_features)

    # target = [t for t in target if t.startswith('H_')]
    print('Target', len(target))
    print('Input Features', features)

    for i, subject_id in enumerate(subjects):

        subject = dataframe[dataframe['PTID']==subject_id]
        subject_unnorm = dataframeunnorm[dataframeunnorm['PTID']==subject_id]

        # print(subject)
        for k in range(0, subject.shape[0]):
            samples['PTID'].append(subject_id)
            covariates['PTID'].append(subject_id)

            print('Baseline Features',  features)

            x = subject[features].iloc[0].to_list()

            # print(x)

            delta = subject['Time'].iloc[k]
            man_device = subject['MRI_Scanner_Model'].iloc[k]
            diagnosis = subject['Diagnosis'].iloc[k]
            baseline_diagnosis = subject['Diagnosis'].iloc[0]
            baseline_hypertension = subject['Hypertension'].iloc[0]    
            age = subject_unnorm['Age'].iloc[k]
            dlicv = subject['DLICV'].iloc[k]
            study = subject['Study'].iloc[k]
            edu_years = subject['Education_Years'].iloc[k]
            race = subject['Race'].iloc[k]
            spare_ad = subject['SPARE_AD'].iloc[k]
            spare_ba = subject['SPARE_BA'].iloc[k]
            mmse = subject['MMSE_nearest_2.0'].iloc[k]
            adas11 = subject['ADAS_COG_11'].iloc[k]
            adas13 = subject['ADAS_COG_13'].iloc[k]
            ef = subject['ADNI_EF'].iloc[k]
            lan = subject['ADNI_LAN'].iloc[k]
            mem = subject['ADNI_MEM'].iloc[k]
            sex = subject['Sex'].iloc[k] # 0 M 1 F

            
            for com in comorbidities:
                print(com)
                longitudinal_covariates[com].append(subject[com].iloc[k])

            # print('Delta', delta)
            x.extend([delta])

            # print('Input', x)
            # print('Target', target)
            t = subject[target].iloc[k] #.to_list()

            print('Target', t)
            covariates['MRI_Scanner_Model'].append(man_device)
            covariates['Age'].append(age)
            covariates['BaselineDiagnosis'].append(baseline_diagnosis)
            longitudinal_covariates['PTID'].append(subject_id)
            longitudinal_covariates['Time'].append(delta)
            longitudinal_covariates['Age'].append(age)
            longitudinal_covariates['Sex'].append(sex)
            longitudinal_covariates['Diagnosis'].append(diagnosis)
            longitudinal_covariates['DLICV'].append(dlicv)
            longitudinal_covariates['Study'].append(study)
            longitudinal_covariates['Education_Years'].append(edu_years)
            longitudinal_covariates['Race'].append(race)
            longitudinal_covariates['SPARE_AD'].append(spare_ad)
            longitudinal_covariates['SPARE_BA'].append(spare_ba)
            longitudinal_covariates['MMSE_nearest_2.0'].append(mmse)
            longitudinal_covariates['ADAS_COG_11'].append(adas11)
            longitudinal_covariates['ADAS_COG_13'].append(adas13)
            longitudinal_covariates['ADNI_EF'].append(ef)
            longitudinal_covariates['ADNI_LAN'].append(lan)
            longitudinal_covariates['ADNI_MEM'].append(mem)
            
            samples['X'].append(x)
            samples['Y'].append(t.tolist())

            data_x.append(x)
            data_y.append(t)

        subject_data = list(zip(data_x, data_y))
        num_samples +=len(subject_data)
        list_of_subjects.append(subject_data)
        list_of_subject_ids.append(subject_id)

    assert len(samples['PTID']) == len(samples['X'])
    assert len(samples['X']) == len(samples['Y'])

    return samples, subject_data, num_samples, list_of_subjects, list_of_subject_ids, cnt, covariates, longitudinal_covariates

"""**Data Selection**
1. Read Data and remove all ADNI Screening and BLSA 1.5 T
2. Drop all NaN MUSE
3. Map the Diagnosis Column
"""
'''
This script runs  on the cluster
'''
# New Directory: /cbica/projects/ISTAGING/Pipelines/ISTAGING_Data_Consolidation_2020/v2.0
# Old Directory: /cbica/projects/ISTAGING/Pipelines/ISTAGING_Data_Consolidation_2020/Latest_Release/HARMONIZED_MUSE/istaging.pkl.gz
# data = pd.read_pickle('/cbica/projects/ISTAGING/Pipelines/ISTAGING_Data_Consolidation_2020/v2.0/istaging.pkl.gz')
# data = data.loc[data.Study.isin(['ADNI'])] # was: ADNI, BLSA

# data = pd.read_csv('adni_dataset.csv')


# cogn = pd.read_csv("ADNI_COGNITIVE_SCORES.csv")
# cogn = cogn[['PTID', 'Date', 'ADNI_EF', 'ADNI_LAN', 'ADNI_MEM', 'ADAS_COG_11', 'ADAS_COG_13']]
# cogn['Date'] = cogn['Date'].astype('datetime64[ns]')
# data['Date'] = data['Date'].astype('datetime64[ns]')

# print('Before Merging the Extra Cognitive Scores', data.shape)
# data = data.merge(right=cogn, how='inner', on=['PTID', 'Date'])
# print('After Merging the Extra Cognitive Scores', data.shape)

# # Impute missing cognitive scores within each subject by using the closest known values
# def impute_closest(group):
#     group = group.sort_values(by='Date')
#     for column in cognitive_columns:
#         group[column] = group[column].fillna(method='ffill').fillna(method='bfill')
#     return group

# # Identify cognitive score columns
# cognitive_columns = ['ADNI_EF', 'ADNI_LAN', 'ADNI_MEM', 'ADAS_COG_11', 'ADAS_COG_13']
# data = data.groupby('PTID').apply(impute_closest).reset_index(drop=True)

# print(data.shape)

# print(data['Date'].head(10))

# date_column = 'Date'  # Replace with the actual date column name

# # Convert date column to datetime
# data[date_column] = pd.to_datetime(data[date_column])
# pet_vars = ['AV45_SUVR', 'FDG', 'PIB_SUVR', 'PIB_SUVR_PAC', 'PIB_Status', 'PTau_CSF', 'Tau_CSF']

# # Function to impute PET variables based on closest date within 6 months
# def impute_pet_variables(group):
#     group = group.sort_values(by=date_column)
#     for pet in pet_vars:
#         for i in range(len(group)):
#             if pd.isna(group.iloc[i][pet]):
#                 closest = group.iloc[(group[date_column] - group.iloc[i][date_column]).abs().argsort()]
#                 for j in range(len(closest)):
#                     if not pd.isna(closest.iloc[j][pet]) and abs((closest.iloc[j][date_column] - group.iloc[i][date_column]).days) <= 180:
#                         group.at[group.index[i], pet] = closest.iloc[j][pet]
#                         break
#     return group

# # Apply the function to each subject
# data = data.groupby('PTID').apply(impute_pet_variables).reset_index(drop=True)

# # Save the imputed data to a new CSV file
# output_file_path ='ADNI_IMAGING_COGNITIVE_PET_imputed.csv'  # Replace with your desired output file path
# data.to_csv(output_file_path, index=False)

# print("Imputation completed and saved to adni_imaging_cognitive_pet_imputed.csv")
# print(data.shape)

# print('Remove the 1.5T BLSA Data')
# data = data[data['SITE']!='BLSA-1.5T']

# cognitive_scores = ['MMSE_nearest_2.0'] 
# total_cognitive_scores = ['MMSE_nearest_2.0', 'ADAS_COG_11', 'ADAS_COG_13', 'ADNI_EF', 'ADNI_LAN', 'ADNI_MEM']

# missing_counts = data[pet_vars].isna().sum()

# # Count the available values for each PET variable
# available_counts = data[pet_vars].notna().sum()

# # Total samples with any available PET variables
# total_samples_with_available = data[pet_vars].notna().any(axis=1).sum()

# # Assuming there is a unique identifier for each subject in the dataset
# subject_identifier = 'PTID'  # Replace with the actual subject identifier column name
# total_subjects_with_available = data.loc[data[pet_vars].notna().any(axis=1), subject_identifier].nunique()

# # Display the results
# print("Missing Counts:")
# print(missing_counts)
# print("\nAvailable Counts:")
# print(available_counts)
# print("\nTotal Samples with Available PET Variables:", total_samples_with_available)
# print("Total Subjects with Available PET Variables:", total_subjects_with_available)
# sys.exit(0)

data = pd.read_csv('ADNI_IMAGING_COGNITIVE_PET_imputed.csv')
hmuse_cols = [name for name in data.columns if (name.startswith('H_MUSE_Volume') and int(name[14:])<300)]
print('H_MUSE Columns', len(hmuse_cols))
# store the hmuse_cols list in a file
with open(data_dir + 'hmuse_cols.txt', 'w') as f:
    for item in hmuse_cols:
        f.write("%s\n" % item)

# print('H_MUSE Columns', hmuse_cols)
# read the file

# for c in data.columns:
#     print(c)
# sys.exit(0)
# revome duplicate visit codes per subject
data = data.drop_duplicates(subset=['PTID', 'Visit_Code'], keep='first')

print('Revome all the rows that have Visit_Code == ADNI Screening')
data = data[data['Visit_Code']!='ADNI Screening']
data = data[data['Visit_Code']!='ADNIGO Screening MRI']
print('After', data.shape)

# replace all the NaN diagnosis with the closest diagnosis
data['Diagnosis'] = data['Diagnosis'].fillna(method='ffill')
print('Diagnosis Before')

# Only for the subjects at AIBL Study, replace the PTID with AIBL+PTID
data.loc[data['Study']=='AIBL', 'PTID'] = 'aibl' + data.loc[data['Study']=='AIBL', 'PTID'].astype(str)
data.loc[data['Study']=='PENN', 'PTID'] = 'penn' + data.loc[data['Study']=='PENN', 'PTID'].astype(str)

data['Date'] = data['Date'].astype('datetime64[ns]')
print('SUBJECTS::', len(list(data['PTID'].unique())))

###### Filter out ROWS that have all H_MUSE NANs ######
print('1. Filter NAN MUSE...') # ok
hmuse = list(data.filter(regex='H_MUSE*'))
data = data.dropna(axis=0, subset=hmuse)

print('SUBJECTS::', len(list(data['PTID'].unique())))

unique_diagnosis = list(data['Diagnosis'].unique())
subject_list = list(data['PTID'].unique())

dx_mapping = pd.read_csv('DX_Mapping.csv')

print('Diagnosis Before')
print(data['Diagnosis'].unique())

# using the dx_mapping file, map the diagnosis to the new diagnosis
old_diagnosis, new_diagnosis = [], []

for i, u in enumerate(unique_diagnosis):
    old_diagnosis.append(u)
    indx = dx_mapping[dx_mapping['Diagnosis']==u].index.values
    if len(indx) == 0:
        new_diagnosis.append(u)
    else:
        new_diagnosis.append(dx_mapping['Class'].iloc[indx[0]])

print('Old Diagnosis', old_diagnosis)
print('New Diagnosis', new_diagnosis)

data['Diagnosis'].replace(old_diagnosis, new_diagnosis, inplace=True)
print('Diagnosis After')
print(data['Diagnosis'].unique())


### Only care about the AD and CN Subjects ###
## Remove the subjects that have Vascular Dementia, FTD, PD, Lewy Body Dementia, Hydrocephalus, PCA, TBI, 'MCI'
data = data[~data['Diagnosis'].isin(['Vascular Dementia', 'other', 'FTD', '', 'PD', 'Lewy Body Dementia', 'Hydrocephalus', 'PCA', 'TBI'])]

## Missing Diagnosis just place it as -1 ##
# replace the nan with the UNK
nan_diagnosis_count = data['Diagnosis'].isna().sum()
data.loc[data['Diagnosis'].isna(), 'Diagnosis'] = 'unk'

# delete missing diagnosis
print('Before Deleting Missing Diagnosis', data.shape)
# data = data[data['Diagnosis']!='unk']
print('After Deleting Missing Diagnosis', data.shape)
print(data['Diagnosis'].unique())
print('SUBJECTS::', len(list(data['PTID'].unique())))

data['Diagnosis'].replace(['CN', 'AD', 'dementia', 'MCI', 'early MCI'] ,
[0, 2, 2, 1, 1], inplace=True)

print(data['Diagnosis'].unique())
print('Initial SUBJECTS::', len(list(data['PTID'].unique())))

data['Hypertension'].replace(['Hypertension negative/absent', 'Hypertension positive/present'], [0,1], inplace=True)
data['Hyperlipidemia'].replace(['Hyperlipidemia absent', 'Hyperlipidemia recent/active'], [0,1], inplace=True)
data['Diabetes'].replace(['Diabetes negative/absent', 'Diabetes positive/present'], [0,1], inplace=True)

# prompt: remove all the subjects with only one acquisition.
data = data.groupby('PTID').filter(lambda x: x.shape[0] > 1)

print('Subjects after removing the single-sampled ones', len(list(data['PTID'].unique())))

for s in list(data['SITE'].unique()):
    print(s)

def delta_baseline_fix(data):
    for pt in list(data['PTID'].unique()):
        # print(pt)
        # Identifying indices where the current patient's data is located
        pt_indices = data[data['PTID'] == pt].index
        # Calculating the baseline to subtract
        base = data.loc[pt_indices[0], 'Delta_Baseline']

        # print('Delta Baseline Before', data.loc[pt_indices, 'Delta_Baseline'].tolist())

        if base != 0:  # Only adjust if the base is not already 0
           
            print(pt, data.loc[pt_indices[0], 'Study'])
            print('Delta Baseline Before', data.loc[pt_indices, 'Delta_Baseline'].tolist())

            # Subtracting the base from Delta_Baseline for all entries of the current patient
            data.loc[pt_indices, 'Delta_Baseline'] -= base

            print('After', data.loc[pt_indices, 'Delta_Baseline'].tolist())

    return data

data = delta_baseline_fix(data)

print('Subjects')
print(len(list(data['PTID'].unique())))
print('Acquisitions')
print(data.shape[0])
# prompt: verify that for every subject the Delta_Baseline on the first acquisition is zero
for pt in list(data['PTID'].unique()):
    # print('ID', pt)
    pt_data = data[data['PTID'] == pt]
    # print(pt_data.iloc[0]['Delta_Baseline'])
    
    # Remove any subject that has any value at Delta_Baseline to be negative 
    if pt_data.iloc[0]['Delta_Baseline'] != 0.0:
        print('Error')
    
# prompt: create a new column that is the Delta_Baseline divided by 30 and keep only the integer part and round to the greater integer
import numpy as np
data['Time'] = np.ceil((data['Delta_Baseline'] / 30)).astype(int)

# prompt: calculate the time intervals among consecutive aquisitions within a subject and then plot the distribution of time intervals. Also calculate the mean and the std

import matplotlib.pyplot as plt
import numpy as np
# Calculate the time intervals between consecutive acquisitions
time_intervals, acquisitions = [], [] 
for subject_id in data['PTID'].unique():
  subject_data = data[data['PTID'] == subject_id].sort_values(by='Time')
  acquisitions.append(subject_data.shape[0])
  for i in range(1, len(subject_data)):
    time_interval = subject_data['Time'].iloc[i] - subject_data['Time'].iloc[i-1]
    time_intervals.append(time_interval)

# Plot the distribution of time intervals
plt.hist(time_intervals, bins=20)
plt.xlabel('Time interval (months)')
plt.ylabel('Frequency')
plt.title('Distribution of time intervals between consecutive acquisitions')
plt.show()

# Calculate the mean and standard deviation of the time intervals
mean_interval = np.mean(time_intervals)
std_interval = np.std(time_intervals)

print(f"Mean time interval: {mean_interval}")
print(f"Standard deviation of time interval: {std_interval}")

# prompt: within a subject remove the duplicate entries in Time column
print(data.shape)
print(len(list(data['PTID'].unique())))
data = data.groupby(['PTID', 'Time']).agg(lambda x: x.iloc[0]).reset_index()
print(data.shape)
print(len(list(data['PTID'].unique())))

data_unnorm = data.copy()

"""**Z-scoring on MUSE**"""
def data_normalization_all(data):
    # print('Extract Statistics from', data.shape)
    mean_list, std_list = [],[]
    mean_list = data.mean(axis=0).tolist()
    std_list = data.std(axis=0).tolist()
    for m in mean_list:
        print(m)
    return mean_list, std_list

print('MUSE Data Normalization...')
subjects_df_hmuse = data.filter(regex='H_MUSE*')
mean, std = data_normalization_all(data=subjects_df_hmuse)
print('Unnormalized MUSE ROIS', subjects_df_hmuse.shape)

for i, c in enumerate(list(subjects_df_hmuse)):
    m,s= mean[i], std[i]
    subjects_df_hmuse[c] = (subjects_df_hmuse[c] - m)/s
print('Normalized MUSE ROIS', subjects_df_hmuse.shape)
for h in hmuse:
    # print(h)
    data[h] = subjects_df_hmuse[h]

"""**Verify the z-scoring**"""

print(data['H_MUSE_Volume_4'].head(10))

##### LMM Operations - Do not needed for staging #####
# # prompt: create an additional column named Baseline_Age that is the Age of the first acquisition in every subject
# # What if the min acquisition is not the first one?? ### LOOK AT IT !
# data['Baseline_Age'] = data.groupby('PTID')['Age'].transform('min')

# # prompt: for all the features that start from H_MUSE_* create an additional column named Baseline_H_MUSE_*  that contains the initial H_MUSE_ value for every subjet
# hmuse_cols = [col for col in data.columns if col.startswith('H_MUSE_')]

# for col in hmuse_cols:
#     data['Baseline_' + col] = data.groupby('PTID')[col].transform('first')

# for col in ['SPARE_AD', 'SPARE_BA']: 
#     data['Baseline_' + col] = data.groupby('PTID')[col].transform('first')


"""Processing of Clinical Covariates
1. Normalization of Baseline Age
2. Binarization of Education Years
3. Binarization of Sex
4. Missing Data Indication -1
"""
education_years_median = data['Education_Years'].median()
education_years_std = data['Education_Years'].std()
print('Edu Years Stats', education_years_median, education_years_std)

print('Normalize Age...')
mean_age, std_age = data['Age'].mean() ,data['Age'].std()
data['Age'] = data['Age'].apply(lambda x: (x-mean_age)/std_age)
print('Mean Age', mean_age)
print('STD Age', std_age)

print('Binarize Education Years...')
# mean_ey, std_ey = subjects_df['Education_Years'].mean(),subjects_df['Education_Years'].std()
data['Education_Years'] = data['Education_Years'].apply(lambda x: 1 if x>16 else 0)
data['Education_Years'] = pd.to_numeric(data['Education_Years'])

print('Binarize Sex')
data['Sex'].replace(['M', 'F'], [0,1], inplace=True)

### Augment Here with PET Variables and Clinical Scores 
pet_vars = ['AV45_SUVR', 'FDG', 'PIB_SUVR', 'PIB_SUVR_PAC', 'PIB_Status', 'PTau_CSF', 'Tau_CSF']
total_cognitive_scores = ['MMSE_nearest_2.0', 'ADAS_COG_11', 'ADAS_COG_13', 'ADNI_EF', 'ADNI_LAN', 'ADNI_MEM']

clinical_features = ['Diagnosis', 'Age', 'Sex', 'APOE4_Alleles', 'Education_Years', 'AV45_SUVR', 'FDG', 'PIB_SUVR', 'PIB_SUVR_PAC', 'PIB_Status', 'PTau_CSF', 'Tau_CSF'
, 'MMSE_nearest_2.0', 'ADAS_COG_11', 'ADAS_COG_13', 'ADNI_EF', 'ADNI_LAN', 'ADNI_MEM', 'PTID', 'Delta_Baseline', 'Time']

# mean_spareba, std_spareba = data['SPARE_BA'].mean() ,data['SPARE_BA'].std()
# data['SPARE_BA'] = data['SPARE_BA'].apply(lambda x: (x-mean_spareba)/std_spareba)

for cf in clinical_features:
    print(cf)
    data[cf] = data[cf].fillna(-1)

all_subjects = list(data['PTID'].unique())

"""**LMM Data**"""

print('Subjects', len(list(data['PTID'].unique())))

# cast all the PTID to string
data['PTID'] = data['PTID'].astype(str)

# data.to_csv('LMM_data_baseline_sparesadniblsa.csv')
print('Subjects', len(data['PTID'].unique().tolist()))


"""**Save the pickle files**"""
# import pickle
# clinical_features = ['Diagnosis', 'Age', 'Sex', 'APOE4_Alleles', 'Education_Years', 'PTID', 'Delta_Baseline',  'Time']
# features = [name for name in data.columns if (name.startswith('H_MUSE_Volume') and int(name[14:])<300)]
# features.extend(clinical_features)

# # Saving with pickle
# # with open(data_dir + "features.pkl", "wb") as file:
# #     pickle.dump(features, file)

target = [name for name in data.columns if (name.startswith('H_MUSE_Volume') and int(name[14:])<300)]
target.extend(['SPARE_AD', 'SPARE_BA'])
target.extend([ 'ADAS_COG_13', 'MMSE_nearest_2.0'])

features = clinical_features
# all_subjects = list(data['PTID'].unique())

# data.to_csv(data_dir + 'adni_imaging_clinical_cognitive_baseline.csv')


print('All Diagnosis', len(data['Diagnosis'].unique())) 

print('Total Number of Subjects::', len(all_subjects)) 
samples, subject_data, num_samples, list_of_subjects, list_of_subject_ids, cnt, covariates, longitudinal_covariates = create_baseline_temporal_dataset(subjects=all_subjects, dataframe=data, dataframeunnorm=data_unnorm,  target=target, features=features)

# samples, subject_data, num_samples, list_of_subjects, list_of_subject_ids,  = create_temporal_dataset(subjects=all_subjects, dataframe=data, dataframeunnorm=data_unnorm,  target=target, features=features)


samples_df = pd.DataFrame(data=samples)
covariates_df = pd.DataFrame(data=covariates)
longitudinal_covariates_df = pd.DataFrame(data=longitudinal_covariates)
print('Longitudinal Covariates are stored in ', data_dir + 'longitudinal_covariates_adni.csv')
longitudinal_covariates_df.to_csv(data_dir + 'longitudinal_covariates_cognitive_pet_adni.csv')
samples_df.to_csv(data_dir + 'subjectsamples_baseline_cognitive_pet_adni.csv')
# covariates_df.to_csv(data_dir + 'covariates_staging_adni.csv')    
# print('Total Number of Samples::', samples_df.shape) 

"""**5 Fold Cross Validation**"""
from sklearn.model_selection import KFold

#### Check of duplicates in  list_of_subject_ids #####
print('Check for Duplicates...')
assert len(list(set(list_of_subject_ids))) == len(list_of_subject_ids)

# save the list_of_subject_ids in a file
with open(data_dir + 'list_of_subject_ids_adni.pkl', 'wb') as handle:
    pickle.dump(list_of_subject_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)

# now read the pickle 
with open(data_dir + 'list_of_subject_ids_adni.pkl', 'rb') as handle:
    list_of_subject_ids = pickle.load(handle)

print('Data for K-FOLD Splitting...', len(list_of_subject_ids))
## CREATE 5 FOLDS
kf = KFold(n_splits=5, random_state=None, shuffle=True)
for i, (train_index, test_index) in enumerate(kf.split(list_of_subject_ids)):
    print('Fold::', i)

    train_subject_ids = []
    test_subject_ids = []
    print("TRAIN:", len(train_index), "TEST:", len(test_index))

    for tr in train_index:
        train_subject_ids.append(list_of_subject_ids[tr])

    for te in test_index:
        test_subject_ids.append(list_of_subject_ids[te])

    for t in test_subject_ids:
        if t in train_subject_ids:
            print('There is a leak!!!!')
            sys.exit(0)

    print('Train IDs', len(train_subject_ids))
    print('Test IDs', len(test_subject_ids))

    with open( data_dir  + 'train_subjects_adni'  + str(i) + '.pkl', 'wb') as handle:
        pickle.dump(train_subject_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open( data_dir +  'test_subjects_adni' +  str(i) + '.pkl', 'wb') as handle:
        pickle.dump(test_subject_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)

sys.exit(0)

# Calculating median and standard deviation for Age and Education Years
age_median = data['Baseline_Age'].median()
age_std = data['Baseline_Age'].std()
education_years_median = data['Education_Years'].median()
education_years_std = data['Education_Years'].std()

print(f"Age Median: {age_median}, Age Standard Deviation: {age_std:.2f}")
print(f"Education Years Median: {education_years_median}, Education Years Standard Deviation: {education_years_std:.2f}")

# Calculating percentages for Sex
sex_counts = data['Sex'].value_counts(normalize=True) * 100

# Calculating percentages for Clinical Status at Baseline
clinical_status_counts = data['Diagnosis'].value_counts(normalize=True) * 100

# Calculating percentages for Race
race_counts = data['Race'].value_counts(normalize=True) * 100

# Calculating percentages for APOE4 Alleles
apoe4_counts = data['APOE4_Alleles'].value_counts(normalize=True) * 100

print(f"Sex Percentages:\n{sex_counts}")
print(f"Clinical Status at Baseline Percentages:\n{clinical_status_counts}")
print(f"Race Percentages:\n{race_counts}")
print(f"APOE4 Alleles Percentages:\n{apoe4_counts}")

print('Total Samples', data.shape)
print('Total Subjects', len(list(data['PTID'].unique())))
print('Mean/Std of Acquisitions', np.mean(acquisitions), np.std(acquisitions))

# prompt: Visualize the Baseline Age Distribution for every study  in a plotly boxplot
import plotly.express as px
# Creating the boxplot with specified customizations
fig = px.box(data, x="Study", y="Baseline_Age", title="OASIS Baseline Age Distribution",color_discrete_sequence=['green'])  # This sets all boxplots to red

# Centering the title and setting layout properties
fig.update_layout(
    title={
        'text': "Baseline Age Distribution by Study",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
     width=500, # Fixed width
    height=500, # Fixed height
    yaxis_title='Baseline Age',
    xaxis_title='Study',
    plot_bgcolor='white', # White background
    font=dict(size=15) # Setting font size for all text in the plot
)

# Highlighting a specific boxplot with a different color
# Assuming you want to change the color of the first boxplot
# fig.update_traces(selector=dict(type='box', x=8), marker_color='#FFA07A')  # Adjust 'x' based on the position
fig.write_image(data_dir + 'baseline_age_distribution_of_long_studies.svg')



# prompt: Visualize the Time distribution for all studies

# fig = px.histogram(data, x="Time", nbins=len(data['Time'].unique()), title="Time Distribution")
fig = go.Figure(data=[go.Histogram(x=data['Time'], marker_color='orange')])

# Centering the title and setting layout properties
fig.update_layout(
    title={
        'text': "Time Distribution",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
     width=800, # Fixed width
    height=600, # Fixed height

    plot_bgcolor='white', # White background
    font=dict(size=17) # Setting font size for all text in the plot
)

# prompt: plot the distribution of H_MUSE_Volume_4

fig = px.histogram(data, x="H_MUSE_Volume_4", nbins=100, title="Distribution of H_MUSE_Volume_4")

# Centering the title and setting layout properties
fig.update_layout(
    title={
        'text': "Distribution of H_MUSE_Volume_4",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
     width=800, # Fixed width
    height=600, # Fixed height

    plot_bgcolor='white', # White background
    font=dict(size=17) # Setting font size for all text in the plot
)

# prompt: normalize all the H_MUSE_Volume*

for col in hmuse_cols:
  data['Baseline_' + col] = (data['Baseline_' + col] - data['Baseline_' + col].min()) / (data['Baseline_' + col].max() - data['Baseline_' + col].min())

# prompt: normalize the Age

data['Baseline_Age'] = (data['Baseline_Age'] - data['Baseline_Age'].min()) / (data['Baseline_Age'].max() - data['Baseline_Age'].min())

# prompt: plot the distribution of H_MUSE_Volume_4

fig = px.histogram(data, x="H_MUSE_Volume_4", nbins=100, title="Distribution of Normalized H_MUSE_Volume_4")

# Centering the title and setting layout properties
fig.update_layout(
    title={
        'text': "Distribution of Normalized H_MUSE_Volume_4",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
     width=800, # Fixed width
    height=600, # Fixed height

    plot_bgcolor='white', # White background
    font=dict(size=17) # Setting font size for all text in the plot
)
fig.show()