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
  dred_mins = dred_df.groupby(pd.TimeGrouper(fl))[u'room'].apply(set).apply(list);
  dred_mins = dred_mins.apply(lambda x: float('NaN') if len(x)==0 else x).dropna();
  mlb = preprocessing.MultiLabelBinarizer();
  dred_bin = mlb.fit_transform(dred_mins);
  dred_bin_df = pd.DataFrame(data=dred_bin, columns=list(mlb.classes_), index=dred_mins.index);
  return dred_bin_df.loc[timestamps].dropna();

# get smart meter features - max, min, average, std
def get_smf(start_time, end_time, feature_length):
  start_time=dt.strptime(start_time, "%Y-%m-%d");
  end_time=dt.strptime(end_time, "%Y-%m-%d");
  fl = str(feature_length) + "s";
  timestamps=pd.date_range(start_time, end_time, freq=fl);
  sm_df = pd.DataFrame.from_csv('../dred/Aggregated_data.csv');
  # need to use different groupby, that can adjust based on feature length
  dred_mins = dred_df.groupby(pd.TimeGrouper(fl))[u'room'].apply(set).apply(list);
  dred_mins = dred_mins.apply(lambda x: float('NaN') if len(x)==0 else x).dropna();
  mlb = preprocessing.MultiLabelBinarizer();
  dred_bin = mlb.fit_transform(dred_mins);
  dred_bin_df = pd.DataFrame(data=dred_bin, columns=list(mlb.classes_), index=dred_mins.index);
  return dred_bin_df.loc[timestamps].dropna();
