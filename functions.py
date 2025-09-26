import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
import torch
import pickle
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import seaborn as sns 
from torch.utils.data import Dataset, DataLoader



def process_temporal_singletask_data(train_x, train_y, test_x, test_y, test_ids): 
    assert train_x.shape[0] == train_y.shape[0]
    train_x_data = [] 
    assert len(train_x) > 0

    for i, t in enumerate(train_x): 
        a = t.strip('][').split(', ')
        # print(a)
        b = [float(i) for i in a]    
        # print(b)
        train_x_data.append(np.expand_dims(np.array(b), 0))

    test_x_data = []
    for i, t in enumerate(test_x): 
        a = t.strip('][').split(', ')
        # print(a)
        b = [float(i) for i in a]
        # print(b)
        test_x_data.append(np.expand_dims(np.array(b), 0))

    test_y_data = [] 
    # print(test_y)
    for i, t in enumerate(test_y): 
        # print(t)
        a = t.strip('][').split(', ')
        # print(a)
        b = [float(i) for i in a]
        # print(b)
        # print(len(b))
        test_y_data.append(np.expand_dims(np.array(b), 0))
    
    train_y_data = [] 
    for i, t in enumerate(train_y): 
        a = t.strip('][').split(', ')
        # print(a)
        b = [float(i) for i in a]
        # print(b)
        # print(len(b))
        train_y_data.append(np.expand_dims(np.array(b), 0))

    train_x_data = np.concatenate(train_x_data, axis=0)
    test_x_data = np.concatenate(test_x_data, axis=0)
    train_y_data = np.concatenate(train_y_data, axis=0)
    test_y_data = np.concatenate(test_y_data, axis=0)
    
    train_y, test_y = np.array(train_y), np.array(test_y)

    # print(train_x_data)

    data_train_x = torch.Tensor(train_x_data)
    data_train_y = torch.Tensor(train_y_data)
    data_test_x = torch.Tensor(test_x_data)
    data_test_y = torch.Tensor(test_y_data)  

    # data_train_x,  data_train_y, data_test_x, data_test_y  = torch.Tensor(train_x_data), torch.Tensor(train_y_data), torch.Tensor(test_x_data), torch.Tensor(test_y_data)

    return data_train_x, data_train_y, data_test_x, data_test_y

def process_temporal_multitask_data(train_x, train_y, test_x, test_y, test_ids): 

    print(type(train_x), type(train_y), type(test_x), type(test_y)) 

    train_x_data = [] 
    for i, t in enumerate(train_x): 
        a = t.strip('][').split(', ')
        # print(a) 
        b = [float(i) for i in a]
        # print(b)
        # print(np.expand_dims(np.array(b), 0).shape)
        train_x_data.append(np.expand_dims(np.array(b), 0))

    test_x_data = []
    for i, t in enumerate(test_x): 
        a = t.strip('][').split(', ')
        b = [float(i) for i in a]
        test_x_data.append(np.expand_dims(np.array(b), 0))

    train_y_data = [] 
    for i, t in enumerate(train_y): 
        a = t.strip('][').split(', ')
        # print(a)
        # for k in a: 
        #     print(k)
        b = [float(i) for i in a]
        # print(len(b))
        train_y_data.append(np.expand_dims(np.array(b), 0))

    test_y_data = [] 
    for i, t in enumerate(test_y): 
        a = t.strip('][').split(', ')
        # print(a)
        # sys.exit
        b = [float(i) for i in a]
        # print(len(b))
        test_y_data.append(np.expand_dims(np.array(b), 0))
    
    

    train_x_data = np.concatenate(train_x_data, axis=0)
    test_x_data = np.concatenate(test_x_data, axis=0)
    train_y_data = np.concatenate(train_y_data, axis=0)
    test_y_data = np.concatenate(test_y_data, axis=0)
    
    train_y, test_y = np.array(train_y), np.array(test_y)

    data_train_x,  data_train_y, data_test_x, data_test_y  = torch.Tensor(train_x_data), torch.Tensor(train_y_data), torch.Tensor(test_x_data), torch.Tensor(test_y_data)

    return data_train_x, data_train_y, data_test_x, data_test_y
    pass 

def process_population_multitask_data(train_x, train_y, test_x, test_y, test_ids): 

    print(type(train_x), type(train_y), type(test_x), type(test_y)) 

    train_x_data = [] 
    for i, t in enumerate(train_x): 
        a = t.strip('][').split(', ')
        b = [float(i) for i in a]
        train_x_data.append(np.expand_dims(np.array(b), 0))

    test_x_data = []
    for i, t in enumerate(test_x): 
        a = t.strip('][').split(', ')
        b = [float(i) for i in a]
        test_x_data.append(np.expand_dims(np.array(b), 0))

    test_y_data = [] 
    for i, t in enumerate(test_y): 
        a = t.strip('][').split(', ')
        b = [float(i) for i in a]
        # print(len(b))
        test_y_data.append(np.expand_dims(np.array(b), 0))
    
    train_y_data = [] 
    for i, t in enumerate(train_y): 
        a = t.strip('][').split(', ')
        b = [float(i) for i in a]
        # print(len(b))
        train_y_data.append(np.expand_dims(np.array(b), 0))

    train_x_data = np.concatenate(train_x_data, axis=0)
    test_x_data = np.concatenate(test_x_data, axis=0)
    train_y_data = np.concatenate(train_y_data, axis=0)
    test_y_data = np.concatenate(test_y_data, axis=0)
    
    train_y, test_y = np.array(train_y), np.array(test_y)

    data_train_x,  data_train_y, data_test_x, data_test_y  = torch.Tensor(train_x_data), torch.Tensor(train_y_data), torch.Tensor(test_x_data), torch.Tensor(test_y_data)

    return data_train_x, data_train_y, data_test_x, data_test_y

def create_hmuse_singletask_temporal_dataset(subjects, dataframe, target, features, genomic, follow_up=0, visualize=True): 
    '''
    subjects: list of the subject ids 
    dataframe: dataframe with all the data 
    target: a single H_MUSE ROI, it is also the input feature along with the delta time
    '''
    print('Target', target)
    cnt = 0 
    num_samples = 0 
    list_of_subjects, list_of_subject_ids = [], []  
    data_x, data_y = [], [] 

    hmuse = [i for i in features if i.startswith('H_MUSE')]

    samples = {'PTID': [], 'X': [], 'Y': []} 
    covariates = {'PTID': [], 'MRI_Scanner_Model':[]}

    if visualize: 
        vdata = {'target': [], 'class': [], 'time': [], 'id': []  } 
        cnt = 0 

    # remove the PTID from the features! 
    features.remove('PTID')
    features.remove('Delta_Baseline')
    features.remove('Time')
    # print('Features', features)
    clinical_features = [f for f in features if not f.startswith('H_MUSE') ]
    # print('Clinical Features', clinical_features)
    
    if genomic: 
        cols = list(dataframe.columns)
        print(cols)
        genomic_features = []
        for c in cols: 
            if c.startswith('rs'): 
                genomic_features.append(c)
        print('Genomic Features', genomic_features)

    # hmuse = target  # is all the brain image!! 

    for i, subject_id in enumerate(subjects): 

        subject = dataframe[dataframe['PTID']==subject_id]
        # print(subject.shape)
        # print(subject[hmuse])
        if visualize: 
            if subject.shape[0] > 8 and cnt < 10: 
                cnt += 1 
                vdata['class'].extend([hmuse for i in range(subject.shape[0])])
                vdata['target'].extend(subject[target].to_list())
                vdata['time'].extend(subject['Time'].to_list())
                vdata['id'].extend(subject['PTID'].to_list())

        # print(subject)
        for k in range(0, subject.shape[0]): 
            samples['PTID'].append(subject_id)
            covariates['PTID'].append(subject_id)

            x = subject[hmuse].iloc[0].to_list()           
            # print(x)
            # print('Clinical Features', clinical_features)         
            x.extend(subject[clinical_features].iloc[0].to_list())

            # print('Imaging+Clinical Features at Baseline', len(x))

            if genomic: 
                # print('Genomic Features')
                x.extend(subject[genomic_features].iloc[0].to_list())

            if follow_up: 
                fp, fp_delta, fp_scanner = [], [], []  
                # print(follow_up)
                for w in range(1,follow_up+1):
                    fp.extend(subject[hmuse].iloc[w].to_list()) 
                    fp_delta.append(subject['Time'].iloc[w])
                    #fp_scanner.append(subject['MRI_Scanner_Model'].iloc[i])

                assert len(fp_delta) == follow_up
                # assert len(fp_scanner) == follow_up
                # print(fp_delta)

                fp.extend(fp_delta)
                #fp.extend(fp_scanner)
            
                x.extend(fp)
                # print('Follow-Up Information', len(fp))
                # print('Inaging+Clinical+Follow Ups', len(x))

            delta = subject['Time'].iloc[k]
            man_device = subject['MRI_Scanner_Model'].iloc[k]

            # x.extend([delta, man_device])
            x.extend([delta])


            # print('Input to the temporal GP', len(x))
            # sys.exit(0)
            covariates['MRI_Scanner_Model'].append(man_device)
            samples['X'].append(x)
            samples['Y'].append([subject[target].iloc[k]])
            data_x.append(x)
            data_y.append([subject[target].iloc[k]])


        assert len(covariates['MRI_Scanner_Model']) == len(samples['X'])

        subject_data = list(zip(data_x, data_y))
        num_samples +=len(subject_data)
        list_of_subjects.append(subject_data)
        list_of_subject_ids.append(subject_id)

    
    assert len(samples['PTID']) == len(samples['X'])
    assert len(samples['X']) == len(samples['Y'])
    
    if visualize: 
        plt.figure(figsize=(10,7), dpi=150)
        ax = sns.lineplot(x='time', y='target', hue='id', data=vdata)
        plt.legend(fontsize=15) 
        plt.xlabel('Time (in months)', fontsize=30)
        plt.ylabel(hmuse, fontsize=30)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        plt.title('Temporal Function of ' + target ,fontsize=25)
        plt.savefig('temporal_function_roi' + target +'.png')

    return samples, subject_data, num_samples, list_of_subjects, list_of_subject_ids, cnt, covariates
 
def create_hmuse_temporal_dataset(subjects, dataframe, target, features, genomic, followup, visualize=False): 
    '''
    subjects: list of the subject ids 
    dataframe: dataframe with all the data 
    target: H_MUSE ROI features
    '''

    print('Target', target)

    cnt = 0 
    num_samples = 0 
    list_of_subjects, list_of_subject_ids = [], []  
    data_x, data_y = [], [] 

    samples = {'PTID': [], 'X': [], 'Y': []} 
    covariates = {'PTID': [], 'MRI_Scanner_Model':[]}


    if visualize: 
        vdata = {'target': [], 'class': [], 'time': [], 'id': []  } 
        cnt = 0 

    if genomic: 
        cols = list(dataframe.columns)
        print(cols)
        genomic_features = []
        for c in cols: 
            if c.startswith('rs'): 
                genomic_features.append(c)
        print('Genomic Features', genomic_features)

    # remove the PTID from the features! 
    features.remove('PTID')
    features.remove('Delta_Baseline')
    features.remove('Time')
    hmuse = [i for i in features if i.startswith('H_MUSE')]

    # print('Features', features)
    clinical_features = [f for f in features if not f.startswith('H_MUSE')]
    # print('Clinical Features', clinical_features)

    # print('HERE', hmuse)
    for i, subject_id in enumerate(subjects): 

        subject = dataframe[dataframe['PTID']==subject_id]

        # print(subject.shape)
        # print(subject[hmuse])
        if visualize: 
            if subject.shape[0] > 8 and cnt < 10: 
                cnt += 1 
                vdata['class'].extend([hmuse for i in range(subject.shape[0])])
                vdata['target'].extend(subject[hmuse].to_list())
                vdata['time'].extend(subject['Time'].to_list())
                vdata['id'].extend(subject['PTID'].to_list())

        # print(subject)
        for k in range(0, subject.shape[0]): 
            samples['PTID'].append(subject_id)
            covariates['PTID'].append(subject_id)

            x = subject[hmuse].iloc[0].to_list()           
            # print(x)
            # print('Clinical Features', clinical_features)         
            x.extend(subject[clinical_features].iloc[0].to_list())

            if genomic: 
                # print('Genomic Features')
                x.extend(subject[genomic_features].iloc[0].to_list())

            if followup: 
                fp = [] 
                for i in range(followup):
                    fp.extend(subject[hmuse].iloc[1+i].to_list()) 
 
                fp.extend(subject['Time'].iloc[1:followup+i].to_list())
                # fp.extend(subject['MRI_Scanner_Model'].iloc[1:followup+i].to_list())
                
                x.extend(fp)
                # print('Follow-Up Information', fp)

            delta = subject['Time'].iloc[k]
            man_device = subject['MRI_Scanner_Model'].iloc[k]

            # print('Delta', delta)
            x.extend([delta])
            t = subject[target].iloc[k].to_list()
            # print(t)
            # for k in t: 
            #     print(k)
            # sys.exit(0)
            covariates['MRI_Scanner_Model'].append(man_device)
            samples['X'].append(x)
            samples['Y'].append(t)
            data_x.append(x)
            data_y.append(t)

        subject_data = list(zip(data_x, data_y))
        num_samples +=len(subject_data)
        list_of_subjects.append(subject_data)
        list_of_subject_ids.append(subject_id)

    assert len(samples['PTID']) == len(samples['X'])
    assert len(samples['X']) == len(samples['Y'])
    
    if visualize: 
        plt.figure(figsize=(10,7), dpi=150)
        ax = sns.lineplot(x='time', y='target', hue='class', style='id', data=vdata)
        plt.legend(fontsize=15) 
        plt.xlabel('Time (in months)', fontsize=30)
        plt.ylabel('ROI', fontsize=30)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        plt.title('Temporal Function of ROIs' ,fontsize=25)
        plt.savefig('temporal_function_roi.png')

    return samples, subject_data, num_samples, list_of_subjects, list_of_subject_ids, cnt, covariates
    pass 

def select_baseline_data(df): 

    ptid=''
    row_index = [] 
    for index, row in df.iterrows(): 
        if row['PTID'] != ptid:
            ptid = row['PTID']
            row_index.append(index)
    return df.loc[row_index, :]


# Function for including certain variables into dataframe.
# The variable will be merged to the line with same PTID and nearest date
def nearest_date_merge(df, df_clin_var, var_list, on_var='Date', by_var='PTID', tolerance='60D'): 
    # df: dataframe to be merged into
    # df_clin_var: dataframe with variables to be included.
    # var_list: names of variables to be included
    # tolerance: largest tolerated date gap;
    df['Date']=pd.to_datetime(df['Date'])
    df_clin_var['Date']=pd.to_datetime(df_clin_var['Date'])
    df=df.sort_values(by=['Date'])
    drop_list=var_list+['Date']
    df_clin_var=df_clin_var.sort_values(by=['Date']).dropna(subset=drop_list)
    df=pd.merge_asof(df,df_clin_var[var_list],on=on_var,by=by_var,tolerance=pd.Timedelta(tolerance),direction='nearest') 
    df=df.sort_values(by=['Study','PTID','Date'])
    df.groupby(['Study','PTID'])
    return df


# Function for correcting age and sex effect via linear regression
def age_sex_correction(df,cov):
    # df; dataframe with ROI volumes with one extra column indicating diagnosis
    # cov: dataframe with three columns: sex, age, diagnosis
    cov['Sex'].replace({'F': 0, 'M': 1},inplace=True)
    max_age, min_age=np.max(cov['Age']), np.min(cov['Age'])
    cov['Age'] = (cov['Age'] - min_age) / (max_age - min_age)
    cn_df = np.array(df.loc[df.Diagnosis=='CN'].drop(columns=['Diagnosis']),dtype='float64') # all the hmuse rois 
    cn_cov = np.array(cov.loc[cov.Diagnosis=='CN'].drop(columns=['Diagnosis']), dtype='float64' )
    pt_df = np.array(df.loc[df.Diagnosis.isin(['MCI','Dementia'])].drop(columns=['Diagnosis']),dtype='float64')
    pt_cov = np.array(cov.loc[cov.Diagnosis.isin(['MCI', 'Dementia'])].drop(columns=['Diagnosis']), dtype='float64' )
    
    coeffs = [0] * cn_df.shape[1] 

    for i in range(cn_df.shape[1]):
        reg = LinearRegression().fit(cn_cov, cn_df[:,i])
        cn_df[:,i]=cn_df[:,i]-np.dot(cn_cov,reg.coef_)
        pt_df[:,i]=pt_df[:,i]-np.dot(pt_cov,reg.coef_)
        coeffs[i] = reg.coef_

    return coeffs


# normalize each ROI data with respect to CN data to ensure a mean of 0 and std of 1 among CN group in each ROI
def data_normalization_cn(cn_data):
    print('Extract Statistics from', cn_data.shape)
    mean_list, std_list = [],[] 
    mean_list = cn_data.mean(axis=0).tolist()
    std_list = cn_data.std(axis=0).tolist() 
    print(len(mean_list), len(std_list))
    for m in mean_list:
        print(m)
    return mean_list, std_list 

def data_normalization_all(data):
    # print('Extract Statistics from', data.shape)
    mean_list, std_list = [],[] 
    mean_list = data.mean(axis=0).tolist()
    std_list = data.std(axis=0).tolist() 
    # print(len(mean_list), len(std_list))
    # for m in mean_list:
        # print(m)
    return mean_list, std_list  

def data_normalization(df, keyword): 
    '''
    Normalizes a subset of pandas columns 
    '''
   #TODO
    pass 

# compute the MAE
### Utils 
# compute the MAE
def mae(y, y_hat):
    """
    PARAMETERS
    y: array of ground truth values
    y_hat: array of predicted values

    RETURN
    mae_result: float of mean absolute error 
    abs_diff: absolute error
    """
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)
    diff = np.subtract(y, y_hat)
    abs_diff = np.fabs(diff)
    mae_result = np.sum(abs_diff, axis=0)/len(y_hat)

    return mae_result, abs_diff


def mse(y, y_hat): 
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)
    diff = np.subtract(y, y_hat)
    diff_squared = np.power(diff, 2) 

    mse_result = np.sum(diff_squared, axis=0)/len(y_hat)

    rmse_result = np.sqrt(mse_result)

    return mse_result, rmse_result, diff_squared

def R2(y, y_hat): 
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)
    ybar = np.sum(y, axis=0)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((y_hat-ybar)**2, axis=0)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2, axis=0)

    r_sq = ssreg/sstot

    return r_sq


if __name__ == "__main__": 

    data = pd.read_pickle('/cbica/projects/ISTAGING/Pipelines/ISTAGING_Data_Consolidation_2020/Latest_Release/istaging.pkl.gz')
    ADNI_cognitive_data = pd.read_csv("/cbica/projects/ISTAGING/Pipelines/ISTAGING_Data_Consolidation_2020/Latest_Release/Extra/ADNI_COGNITIVE_SCORES.csv")


    # Select only ADNI and BLSA data
    ADNI_BLSA_data = data.loc[data.Study.isin(['ADNI'])] # was: ADNI, BLSA
    # Keep only selected clinical variables and ROI/WML 
    Selected_Var = ['PTID','Phase','Age','Sex','Study','Date','APOE4_Alleles','Education_Years','Diagnosis','Abeta_CSF','Tau_CSF','PTau_CSF','MRID']
    Selected_ROI = [ name for name in data.columns if ('H_MUSE_Volume' in name and int(name[14:])<300)] 
    # Selected_wml = [ name for name in data.columns if 'H_WMLS_Volume' in name]
    # ADNI_BLSA_data_selected = ADNI_BLSA_data[Selected_Var+Selected_ROI+Selected_wml]

    ADNI_BLSA_data_selected = ADNI_BLSA_data[Selected_Var + Selected_ROI]


    # Use nearest merge to match CSF/Cognition data with nearest available ROI volume data
    ADNI_BLSA_data_selected = nearest_date_merge(ADNI_BLSA_data_selected.drop(columns=(['Abeta_CSF','Tau_CSF','PTau_CSF'])), ADNI_BLSA_data_selected,['PTID','Date','Abeta_CSF','Tau_CSF','PTau_CSF'])
    ADNI_BLSA_data_selected = nearest_date_merge(ADNI_BLSA_data_selected,ADNI_cognitive_data,['PTID','Date','ADNI_EF','ADNI_MEM','ADNI_LAN'])
    # Select out baseline data only for construction of training set

    ADNI_BLSA_data_selected.to_csv('ADNI_Extra_Cogn_Scores.csv')

    ADNI_BLSA_baseline_data = select_baseline_data(ADNI_BLSA_data_selected)


    # Selected out ADNI2/GO data as training set
    Train_data = ADNI_BLSA_baseline_data.loc[ADNI_BLSA_baseline_data.Phase.isin(['ADNI2','ADNIGO'])]

    hmuse = list(Train_data.filter(regex='H_MUSE*'))
    Train_data = Train_data.dropna(axis=0, subset=hmuse)
    Train_data = Train_data.dropna(axis=0, subset=['Age', 'Sex', 'Diagnosis'])

    # Apply age/sex effect correction
    Corrected_CN, Corrected_PT = age_sex_correction(Train_data[Selected_ROI+['Diagnosis']],
                                                    Train_data[['Age','Sex','Diagnosis']])
    # Normalize data with respect to CN 
    Normalized_CN, Normalized_PT = data_normalization(Corrected_CN, Corrected_PT)



    # Selected out ADNI2/GO data as training set
    Train_data = ADNI_BLSA_baseline_data.loc[ADNI_BLSA_baseline_data.Phase.isin(['ADNI2','ADNIGO'])]
    # Apply age/sex effect correction
    Corrected_CN, Corrected_PT = age_sex_correction(Train_data[Selected_ROI+['Diagnosis']],
                                                    Train_data[['Age','Sex','Diagnosis']])
    # Normalize data with respect to CN 
    Normalized_CN, Normalized_PT = data_normalization(Corrected_CN, Corrected_PT)


    roi_std, roi_mean = np.std(Normalized_PT,axis=0), np.mean(Normalized_PT,axis=0)
    sorted_std, sorted_roi_std = zip(*sorted(zip(roi_std, Selected_ROI),reverse=True))
    sorted_mean, sorted_roi_mean = zip(*sorted(zip(roi_mean, Selected_ROI)))
    fig = plt.figure(figsize=(16,6))
    plt.title('Patient ROI Standard Deviation',fontsize=20)
    plt.bar([name[14:] for name in sorted_roi_std[0:20]],sorted_std[0:20])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()






def prep_fold_data(file_):
    datasamples = pd.read_csv(file_) 
    
    print(datasamples.shape)
    
    unique_ids = list(datasamples['PTID'].unique())

    datasamples['PTID'] = datasamples['PTID'].astype("category")    
    print(datasamples['PTID'].dtypes)    
    datasamples['PTID'] = datasamples['PTID'].cat.codes

    print(datasamples['PTID'].dtypes)    

    unique_int_ids = list(datasamples['PTID'].unique())


    print(len(unique_ids), len(unique_int_ids)) 

    
    IDS = np.expand_dims(np.array(datasamples['PTID'].to_list()), axis=1) 

    print('IDS', IDS.shape)

    X = datasamples['X']
    Y = datasamples['Y'] 

    X_list, Y_list = [],[] 

    for i, t in enumerate(X): 
        a = t.strip('][').split(', ')
        b = [float(i) for i in a]
        features = len(b)
        X_list.append(np.expand_dims(np.array(b), 0))
    for i, t in enumerate(Y): 
        a = t.strip('][').split(', ')
        b = [float(i) for i in a]
        Y_list.append(np.expand_dims(np.array(b), 0))

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0 )

    # print('Input', 'Target', X.shape, Y.shape)

    Inds = np.zeros((Y.shape[0], Y.shape[1]))

    # print('Inds', Inds.shape)

    data = np.concatenate((IDS, X, Y, Inds), axis=1)

    return data, unique_ids, unique_int_ids, features


##### LOAD AND SAVE MODEL FUNCTIONS #### 

# Save the model state dictionary and the optimizer state and the data 
def save_model(model, optimizer, likelihood, filename="model_state.pth"):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'likelihood_state_dict': likelihood.state_dict()
    }, filename)


# Load the model, optimizer, and likelihood from the saved file
def load_model(model, optimizer, likelihood, filename="deep_kernel_gp_model.pth"):
    print('Load Model')
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
    return model, optimizer, likelihood
