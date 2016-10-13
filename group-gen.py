import pandas as pd

dred = pd.DataFrame.from_csv("Appliance_data.csv");
dred = dred.dropna();
dred['television'] = dred['television'].apply(lambda x: 1 if x>49 else 0);
dred['fan'] = dred['fan'].apply(lambda x: 1 if x>30 else 0);
dred['fridge'] = dred['fridge'].apply(lambda x: 1 if x>91 else 0);
dred['laptop computer'] = dred['laptop computer'].apply(lambda x: 1 if x>25 else 0);
dred['electric heating element'] = dred['electric heating element'].apply(lambda x: 1 if x>43 else 0);
dred['oven'] = dred['oven'].apply(lambda x: 1 if x>718 else 0);
dred['unknown'] = dred['unknown'].apply(lambda x: 1 if x>114 else 0);
dred['washing machine'] = dred['washing machine'].apply(lambda x: 1 if x>338 else 0);
dred['microwave'] = dred['microwave'].apply(lambda x: 1 if x>209 else 0);
dred['toaster'] = dred['toaster'].apply(lambda x: 1 if x>374 else 0);
dred['sockets'] = dred['sockets'].apply(lambda x: 1 if x>46 else 0);
dred['cooker'] = dred['cooker'].apply(lambda x: 1 if x>497 else 0);

dred.to_csv('Appliance_data_bin.csv');

# making basket, still not finished
dred_s = pd.DataFrame();
dred_s[0] = dred['television'].apply(lambda x: 'television' if x>0 else '');
dred_s[1] = dred['fan'].apply(lambda x: 'fan' if x>0 else '');
dred_s[2] = dred['fridge'].apply(lambda x: 'fridge' if x>0 else '');
dred_s[3] = dred['laptop computer'].apply(lambda x: 'laptop computer' if x>0 else '');
dred_s[4] = dred['electric heating element'].apply(lambda x: 'electric heating element' if x>0 else '');
dred_s[5] = dred['oven'].apply(lambda x: 'oven' if x>0 else '');
dred_s[6] = dred['unknown'].apply(lambda x: 'unknown' if x>0 else '');
dred_s[7] = dred['washing machine'].apply(lambda x: 'washing machine' if x>0 else '');
dred_s[8] = dred['microwave'].apply(lambda x: 'microwave' if x>0 else '');
dred_s[9] = dred['toaster'].apply(lambda x: 'toaster' if x>0 else '');
dred_s[10] = dred['sockets'].apply(lambda x: 'sockets' if x>0 else '');
dred_s[11] = dred['cooker'].apply(lambda x: 'cooker' if x>0 else '');
