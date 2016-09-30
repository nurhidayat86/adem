import pandas as pd
import os as os

tgt = pd.DataFrame.from_csv('Results' + os.path.sep + 'train_gt.csv');
ogt = pd.DataFrame.from_csv('Results' + os.path.sep + 'ogt.csv');

gt = tgt;
del gt[u'people'];
del gt[u'bathroom'];
gt['occ'] = ogt.convert_objects(convert_numeric=True);

gt.loc['2012-6-2':'2012-6-9'].sum()
