import pandas as pd
from datetime import datetime as dt
from sklearn import preprocessing

# get room occupancy ground truth
def ro_gt(start_time, end_time, feature_length):
  start_time=dt.strptime(start_time, "%Y-%m-%d");
  end_time=dt.strptime(end_time, "%Y-%m-%d");
  fl = str(feature_length) + "s";
  timestamps=pd.date_range(start_time, end_time, freq=fl);
  dred_df = pd.DataFrame.from_csv('../dred/Occupancy_data_split.csv');
  # need to use different groupby, that can adjust based on feature length
  dred_fl = dred_df.groupby(pd.TimeGrouper(fl))[u'room'].apply(set).apply(list);
  dred_fl = dred_fl.apply(lambda x: float('NaN') if len(x)==0 else x).dropna();
  mlb = preprocessing.MultiLabelBinarizer();
  dred_bin = mlb.fit_transform(dred_fl);
  dred_bin_df = pd.DataFrame(data=dred_bin, columns=list(mlb.classes_), index=dred_mins.index);
  return dred_bin_df.loc[timestamps].dropna();

def calculate_sad(dred_fl):
  abs_diff =  dred_fl.diff().abs();
  return abs_diff.sum();
  
def compute_feature(dred_fl):
  min = dred_fl.min();
  max = dred_fl.max();
  mean = dred_fl.mean();
  std = dred_fl.std();
  sad = calculate_sad(dred_fl);
  corl = dred_fl.ix[:,0].autocorr();
  dred_sm_feature = [min, max, mean, std, sad, corl];  
  return dred_sm_feature;
  
# get smart meter features - max, min, average, std
def get_smf(start_time, end_time, feature_length):
  start_time=dt.strptime(start_time, "%Y-%m-%d");
  end_time=dt.strptime(end_time, "%Y-%m-%d");
  fl = str(feature_length) + "s";
  timestamps=pd.date_range(start_time, end_time, freq=fl);
  dred_df = pd.DataFrame.from_csv('../dred/Aggregated_data.csv');
  dred_df = dred_df.loc[timestamps];
  # need to use different groupby, that can adjust based on feature length
  dred_sm_features = dred_df.groupby(pd.TimeGrouper(fl))[u'main'].apply(compute_feature);
  dred_sm_features.columns = ['min', 'max', 'mean', 'std', 'sad', 'corl'];
  return .dropna();
