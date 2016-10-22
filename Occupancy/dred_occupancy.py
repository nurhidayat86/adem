import pandas as pd
from datetime import datetime as dt
from sklearn import preprocessing

# get room occupancy ground truth
def ro_gt(start_time, end_time, feature_length):
  start_time=dt.strptime(start_time, "%Y-%m-%d");
  end_time=dt.strptime(end_time, "%Y-%m-%d");
  fl = str(feature_length) + "s";
  timestamps=pd.date_range(start_time, end_time, freq=fl);
  dred_df = pd.DataFrame.from_csv('../dataset/Occupancy_data_split.csv');
  # need to use different groupby, that can adjust based on feature length
  dred_fl = dred_df.groupby(pd.TimeGrouper(fl))[u'room'].apply(set).apply(list);
  dred_fl = dred_fl.apply(lambda x: float('NaN') if len(x)==0 else x).dropna();
  mlb = preprocessing.MultiLabelBinarizer();
  dred_bin = mlb.fit_transform(dred_fl);
  dred_bin_df = pd.DataFrame(data=dred_bin, columns=list(mlb.classes_), index=dred_fl.index);
  return dred_bin_df.loc[timestamps].dropna();


def calculate_sad(dred_fl):
  abs_diff =  dred_fl.diff().abs();
  return abs_diff.sum();


def extract_features(dred_df, fl):
  max = dred_df.groupby(pd.TimeGrouper(fl))[u'main'].max();
  mean = dred_df.groupby(pd.TimeGrouper(fl))[u'main'].mean();
  min = dred_df.groupby(pd.TimeGrouper(fl))[u'main'].min();
  std = dred_df.groupby(pd.TimeGrouper(fl))[u'main'].std();  
#  corl = dred_df.groupby(pd.TimeGrouper(fl))[u'main'].autocorr();
  dred_sm_features = pd.DataFrame();
  dred_sm_features['max'] = max;
  dred_sm_features['mean'] = mean;
  dred_sm_features.columns = ['max', 'mean'];
  return dred_sm_features;


# get smart meter features - max, min, average, std
def get_smf(start_time, end_time, start_time_2, end_time_2, feature_length):
  dred_df = pd.DataFrame.from_csv('../dataset/Aggregated_data.csv');
  start_time=dt.strptime(start_time, "%Y-%m-%d");
  end_time=dt.strptime(end_time, "%Y-%m-%d");
  start_time_2=dt.strptime(start_time_2, "%Y-%m-%d");
  end_time_2=dt.strptime(end_time_2, "%Y-%m-%d");  
  fl = str(feature_length) + "s";
  timestamps_1 = pd.date_range(start_time, end_time, freq=fl);
  dred_df_1 = dred_df.loc[timestamps_1];
  timestamps_2 = pd.date_range(start_time_2, end_time_2, freq=fl);
  dred_df_2 = dred_df.loc[timestamps_2];
  # need to use different groupby, that can adjust based on feature length
  dred_sm_features_1 = extract_features(dred_df_1, fl);
  dred_sm_features_2 = extract_features(dred_df_2, fl);  
  return dred_sm_features_1.dropna(), dred_sm_features_2.dropna();

  