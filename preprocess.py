#%%
import numpy as np
import pandas as pd



data = pd.read_csv('app/static/data/train.csv')
data2 = pd.read_csv('app/static/data/test.csv')
data = pd.concat([data, data2], axis=0)
data.drop('ID', axis=1, inplace=True)

#####################################
#### COMPONENT_ARBITRARY 
#####################################
data = pd.get_dummies(data, columns=['COMPONENT_ARBITRARY'])

#####################################
#### ANONYMOUS_1 
#####################################
data['ANONYMOUS_1'] = data['ANONYMOUS_1'].astype('float32')
data['ANONYMOUS_1'] = np.log(data['ANONYMOUS_1']+1)
data['ANONYMOUS_1'] = (data['ANONYMOUS_1'] - data['ANONYMOUS_1'].mean()) / data['ANONYMOUS_1'].std()

#####################################
#### YEAR 
#####################################
data['YEAR'] = data['YEAR'].apply(lambda x : 2020 - x)
data = pd.get_dummies(data, columns=['YEAR'])

#####################################
#### SAMPLE_TRANSFER_DAY 
#####################################
data['SAMPLE_TRANSFER_DAY'] = data['SAMPLE_TRANSFER_DAY'].astype('float32')
data['SAMPLE_TRANSFER_DAY'] = data['SAMPLE_TRANSFER_DAY'].fillna(0)
data['SAMPLE_TRANSFER_DAY'] = np.log(data['SAMPLE_TRANSFER_DAY']+1)
data['SAMPLE_TRANSFER_DAY'] = (data['SAMPLE_TRANSFER_DAY'] - data['SAMPLE_TRANSFER_DAY'].mean()) / data['SAMPLE_TRANSFER_DAY'].std()

#####################################
#### ANONYMOUS_2 
#####################################
data['ANONYMOUS_2'] = data['ANONYMOUS_2'].astype('float32')
data['ANONYMOUS_2'] = data['ANONYMOUS_2'].fillna(0)
data['ANONYMOUS_2'] = np.log(data['ANONYMOUS_2']+1)
data['ANONYMOUS_2'] = (data['ANONYMOUS_2'] - data['ANONYMOUS_2'].mean()) / data['ANONYMOUS_2'].std()


#####################################
#### AG 
#####################################

data['AG'] = data['AG'].apply(lambda x : 1 if x > 2 else x)
data = pd.get_dummies(data, columns=['AG'])

#####################################
#### AL 
#####################################
data['AL'] = data['AL'].astype('float32')
data['AL'] = np.log(data['AL']+1)
data['AL'] = (data['AL'] - data['AL'].mean()) / data['AL'].std()

#####################################
#### B 
#####################################
data['B'] = data['B'].astype('float32')
data['B'] = np.log(data['B']+1)
data['B'] = (data['B'] - data['B'].mean()) / data['B'].std()

#####################################
#### BA 
#####################################
data['BA'] = data['BA'].astype('float32')
data['BA'] = np.log(data['BA']+1)
data['BA'] = (data['BA'] - data['BA'].mean()) / data['BA'].std()



#####################################
#### BE 
#####################################
data['BE'] = data['BE'].apply(lambda x : 1 if x != 0 else x)
data = pd.get_dummies(data, columns=['BE'])

#####################################
#### CA 
#####################################
data['CA'] = data['CA'].astype('float32')
data['CA'] = np.log(data['CA']+1)
data['CA'] = (data['CA'] - data['CA'].mean()) / data['CA'].std()

#####################################
#### CD 
#####################################
data = pd.get_dummies(data, columns=['CD'])

#####################################
#### CO 
#####################################
data['CO'] = data['CO'].apply(lambda x : 1 if x != 0 else x)
data = pd.get_dummies(data, columns=['CO'])

#####################################
#### CR 
#####################################
data['CR'] = data['CR'].astype('float32')
data['CR'] = np.log(data['CR']+1)
data['CR'] = (data['CR'] - data['CR'].mean()) / data['CR'].std()

#####################################
#### CU 
#####################################
data['CU'] = data['CU'].astype('float32')
data['CU'] = np.log(data['CU']+1)
data['CU'] = (data['CU'] - data['CU'].mean()) / data['CU'].std()

#####################################
#### FH2O 
#####################################
data['FH2O'] = data['FH2O'].astype('float32')
data['FH2O'] = data['FH2O'].fillna(data['FH2O'].mean())
data['FH2O'] = np.log(data['FH2O']+1)
data['FH2O'] = (data['FH2O'] - data['FH2O'].mean()) / data['FH2O'].std()

#####################################
#### FNOX 
#####################################
data['FNOX'] = data['FNOX'].astype('float32')
data['FNOX'] = data['FNOX'].fillna(data['FNOX'].mean())
data['FNOX'] = np.log(data['FNOX']+1)
data['FNOX'] = (data['FNOX'] - data['FNOX'].mean()) / data['FNOX'].std()


#####################################
#### FOPTIMETHGLY 
#####################################
data['FOPTIMETHGLY'] = data['FOPTIMETHGLY'].fillna(data['FOPTIMETHGLY'].mode()[0])
data['FOPTIMETHGLY'] = data['FOPTIMETHGLY'].apply(lambda x : 0 if x == 0 else 1)
data = pd.get_dummies(data, columns=['FOPTIMETHGLY'])


#####################################
#### FOXID 
#####################################
data['FOXID'] = data['FOXID'].astype('float32')
data['FOXID'] = data['FOXID'].fillna(data['FOXID'].median())
data['FOXID'] = np.log(data['FOXID']+1)
data['FOXID'] = (data['FOXID'] - data['FOXID'].mean()) / data['FOXID'].std()

#####################################
#### FSO4 
#####################################
data['FSO4'] = data['FSO4'].astype('float32')
data['FSO4'] = data['FSO4'].fillna(data['FSO4'].mean())
data['FSO4'] = np.log(data['FSO4']+1)
data['FSO4'] = (data['FSO4'] - data['FSO4'].mean()) / data['FSO4'].std()


#####################################
#### FTBN 
#####################################
data['FTBN'] = data['FTBN'].astype('float32')
data['FTBN'] = data['FTBN'].fillna(data['FTBN'].mean())
data['FTBN'] = np.log(data['FTBN']+1)
data['FTBN'] = (data['FTBN'] - data['FTBN'].mean()) / data['FTBN'].std()

#####################################
#### FE 
#####################################
data['FE'] = data['FE'].astype('float32')
data['FE'] = np.log(data['FE']+1)
data['FE'] = (data['FE'] - data['FE'].mean()) / data['FE'].std()

#####################################
#### FUEL 
#####################################
data['FUEL'] = data['FUEL'].astype('float32')
data['FUEL'] = np.log(data['FUEL']+1)
data['FUEL'] = (data['FUEL'] - data['FUEL'].mean()) / data['FUEL'].std()

#####################################
#### H2O 
#####################################
data['H2O'] = data['H2O'].apply(lambda x : 0 if x == 0 else 1)
data = pd.get_dummies(data, columns=['H2O'])

#####################################
#### K 
#####################################
data['K'] = data['K'].astype('float32')
data['K'] = data['K'].fillna(data['K'].mean())
data['K'] = np.log(data['K']+1)
data['K'] = (data['K'] - data['K'].mean()) / data['K'].std()

#####################################
#### LI 
#####################################
data['LI'] = data['LI'].apply(lambda x : 0 if x == 0 else 1)
data = pd.get_dummies(data, columns=['LI'])

#####################################
#### MG 
#####################################
data['MG'] = data['MG'].astype('float32')
data['MG'] = np.log(data['MG']+1)
data['MG'] = (data['MG'] - data['MG'].mean()) / data['MG'].std()

#####################################
#### MN 
#####################################
data['MN'] = data['MN'].astype('float32')
data['MN'] = data['MN'].fillna(0)
data['MN'] = np.log(data['MN'])
data['MN'] = (data['MN'] - data['MN'].mean()) / data['MN'].std()


#####################################
#### MO 
#####################################
data['MO'] = data['MO'].astype('float32')
data['MO'] = data['MO'].fillna(0)
data['MO'] = np.log(data['MO']+1)
data['MO'] = (data['MO'] - data['MO'].mean()) / data['MO'].std()

#####################################
#### NA 
#####################################
data['NA'] = data['NA'].astype('float32')
data['NA'] = data['NA'].fillna(0)
data['NA'] = np.log(data['NA']+1)
data['NA'] = (data['NA'] - data['NA'].mean()) / data['NA'].std()

#####################################
#### NI 
#####################################
data['NI'] = data['NI'].astype('float32')
data['NI'] = np.log(data['NI']+1)
data['NI'] = (data['NI'] - data['NI'].mean()) / data['NI'].std()

#####################################
#### P 
#####################################
data['P'] = data['P'].astype('float32')
data['P'] = np.log(data['P']+1)
data['P'] = (data['P'] - data['P'].min()) / (data['P'].max() - data['P'].min())

#####################################
#### PB 
#####################################
data['PB'] = data['PB'].apply(lambda x : 0 if x == 0 else 1 )
data = pd.get_dummies(data, columns=['PB'])

#####################################
#### PQINDEX 
#####################################
data['PQINDEX'] = data['PQINDEX'].astype('float32')
data['PQINDEX'] = np.log(data['PQINDEX']+1)
data['PQINDEX'] = (data['PQINDEX'] - data['PQINDEX'].mean()) / data['PQINDEX'].std()

#####################################
#### S 
#####################################
data['S'] = data['S'].astype('float32')
data['S'] = np.log(data['S']+1)
data['S'] = (data['S'] - data['S'].mean()) / data['S'].std()


#####################################
#### SB 
#####################################
data['SB'] = data['SB'].apply(lambda x : x if x < 4 else 4)
data = pd.get_dummies(data, columns=['SB'])

#####################################
#### SI 
#####################################
data['SI'] = data['SI'].astype('float32')
data['SI'] = np.log(data['SI']+1)
data['SI'] = (data['SI'] - data['SI'].mean()) / data['SI'].std()

#####################################
#### SN 
#####################################
data['SN'] = data['SN'].apply(lambda x : x if x < 2 else 2)
data = pd.get_dummies(data, columns=['SN'])

#####################################
#### SOOTPERCENTAGE 
#####################################
data['SOOTPERCENTAGE'] = data['SOOTPERCENTAGE'].astype('float32')
data['SOOTPERCENTAGE'] = data['SOOTPERCENTAGE'].fillna(data['SOOTPERCENTAGE'].mean())
data['SOOTPERCENTAGE'] = np.log(data['SOOTPERCENTAGE']+1)
data['SOOTPERCENTAGE'] = (data['SOOTPERCENTAGE'] - data['SOOTPERCENTAGE'].mean()) / data['SOOTPERCENTAGE'].std()


#####################################
#### TI 
#####################################
data['TI'] = data['TI'].apply(lambda x : x if x == 0 else 1 )
data = pd.get_dummies(data, columns=['TI'])

#####################################
#### U100 
#####################################
data['U100'] = data['U100'].apply(lambda x : 0 if x == 0 else 1)
data = pd.get_dummies(data, columns=['U100'])

#####################################
#### U75 
#####################################
data['U75'] = data['U75'].fillna(data['U75'].mode()[0])
data['U75'] = data['U75'].apply(lambda x : 0 if x == 0 else 1 )
data = pd.get_dummies(data, columns=['U75'])

#####################################
#### U50 
#####################################
data['U50'] = data['U50'].fillna(data['U50'].mode()[0])
data['U50'] = data['U50'].apply(lambda x : 0 if x == 0 else 1  )
data = pd.get_dummies(data, columns=['U50'])

#####################################
#### U25 
#####################################
data['U25'] = data['U25'].astype('float32')
data['U25'] = data['U25'].fillna(data['U25'].mean())
data['U25'] = np.log(data['U25']+1)
data['U25'] = (data['U25'] - data['U25'].mean()) / data['U25'].std()

#####################################
#### U20 
#####################################
data['U20'] = data['U20'].astype('float32')
data['U20'] = data['U20'].fillna(data['U20'].mean())
data['U20'] = np.log(data['U20']+1)
data['U20'] = (data['U20'] - data['U20'].mean()) / data['U20'].std()

#####################################
#### U14 
#####################################
data['U14'] = data['U14'].astype('float32')
data['U14'] = data['U14'].fillna(data['U14'].mean())
data['U14'] = np.log(data['U14']+1)
data['U14'] = (data['U14'] - data['U14'].mean()) / data['U14'].std()

#####################################
#### U6 
#####################################
data['U6'] = data['U6'].astype('float32')
data['U6'] = data['U6'].fillna(data['U6'].mean())
data['U6'] = np.log(data['U6']+1)
data['U6'] = (data['U6'] - data['U6'].mean()) / data['U6'].std()

#####################################
#### U4 
#####################################
data['U4'] = data['U4'].astype('float32')
data['U4'] = data['U4'].fillna(data['U4'].mean())
data['U4'] = np.log(data['U4']+1)
data['U4'] = (data['U4'] - data['U4'].mean()) / data['U4'].std()

#####################################
#### V 
#####################################
data['V'] = data['V'].fillna(data['V'].mode()[0])
data['V'] = data['V'].apply(lambda x : 0 if x == 0 else 1)
data = pd.get_dummies(data, columns=['V'])

#####################################
#### V100 
#####################################
data['V100'] = data['V100'].astype('float32')
data['V100'] = data['V100'].fillna(data['V100'].mean())
data['V100'] = np.log(data['V100']+1)
data['V100'] = (data['V100'] - data['V100'].mean()) / data['V100'].std()

#####################################
#### V40 
#####################################
data['V40'] = data['V40'].astype('float32')
data['V40'] = (data['V40'] - data['V40'].mean()) / data['V40'].std()

#####################################
#### ZN 
#####################################
data['ZN'] = data['ZN'].astype('float32')
data['ZN'] = np.log(data['ZN']+1)
data['ZN'] = (data['ZN'] - data['ZN'].mean()) / data['ZN'].std()
# %%




#%%
data[data['Y_LABEL'].isna() == False ].to_csv('train_mod.csv', index=False)
data[data['Y_LABEL'].isna() == True ].to_csv('test_mod.csv', index=False)

# %%
