
# # Meta-Analysis of Proteomics Sample Preparation workflow

# Import Package
import numpy as np
import pandas as pd
import random as rnd

import seaborn as sns
import matplotlib as mp 
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import lasso_path, enet_path
from sklearn import datasets
import sklearn.cluster as clust

from scipy.stats import ttest_ind
from scipy.spatial.distance import cdist
from itertools import cycle


# Read the data from the data folder

data=pd.read_excel("Dataset_for_Meta_analysis.xlsx")


# check data type 
data.dtypes

#convert column to int64 if required
data['Sample_Quantity'] = data['Sample_Quantity'].astype(np.int64)

data = data.dropna(subset=['Alkylation_Concentration'])
data['Alkylation_Concentration'] = data['Alkylation_Concentration'].astype(np.int64)

# sort data
data= data.sort_values('Number_of_Proteins_Identified', ascending=False)

#take a copy
df_sorted = data.copy()

#encode multi-enzyme
df_sorted["Protease"]=df_sorted["Protease"].replace('Trypsin-LysC-GluC', 'Multi-enzyme')
df_sorted["Protease"]=df_sorted["Protease"].replace('Trypsin-LysC-AspN-GluC-Chymotrypsin-ArgC', 'Multi-enzyme')
df_sorted["Protease"]=df_sorted["Protease"].replace('ArgC-Trypsin-AspN-Chymotrypsin-GluC-LysC', 'Multi-enzyme')


# data cleaning
df_sorted["Mass_Spectrometer"]=df_sorted["Mass_Spectrometer"].replace('Q Exactive ', 'Q Exactive')


# Calculate proteins identified/ug of sample analysed
df_sorted['Proteins/ug']= df_sorted['Number_of_Proteins_Identified']/df_sorted['Sample_Quantity']


# Calculate proteins identified/fraction analysed 
df_sorted["Proteins/Fraction"]=df_sorted['Number_of_Proteins_Identified']/df_sorted['Fraction_Number']

# add unfractionated for nan in fractionation column
df_sorted.Fractionation_Method=df_sorted.Fractionation_Method.fillna('Unfractionated')


# Functions for Data Visualisation
# define functions to create scatterplots

def scatter_plot(x, y, x_label=None, *args, **kwargs):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(x, y, **kwargs)
    plt.xlabel("{}".format(x), fontsize=28, fontweight='bold')
    plt.ylabel("{}".format(y), fontsize=28, fontweight='bold')
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    return ax 

# define a helper function for violinplot with scatterplot

def violin_plot(x, y, x_label=None, *args, **kwargs):
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.violinplot(x, y, **kwargs)
    sns.scatterplot(x,y, color='k', **kwargs)
    ax.set_xlim([-0.5,10])
    ax.set_ylim([-4000, 16000])
    plt.xlabel("{}".format(x), fontsize=28, fontweight='bold')
    plt.ylabel("{}".format(y), fontsize=28, fontweight='bold')
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    return ax 

# define fucntion for boxplot with scatterplot

def box_plot(x, y, x_label=None, *args, **kwargs):
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.boxplot(x, y, **kwargs)
    sns.scatterplot(x,y, color='k', edgecolor='k', alpha=0.6, **kwargs)
    plt.xlabel("{}".format(x), fontsize=28, fontweight='bold')
    plt.ylabel("{}".format(y), fontsize=28, fontweight='bold')
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    return ax 

# define a function for barplot 

def bar_plot(x, y, x_label=None, *args, **kwargs):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x, y, **kwargs)
    plt.xlabel("{}".format(x), fontsize=20, fontweight='bold')
    plt.ylabel("{}".format(y), fontsize=20, fontweight='bold')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    return ax 

# define a function for barhplot 

def barh_plot(y, x, x_label=None, *args, **kwargs):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x, y, **kwargs)
    plt.xlabel("{}".format(x), fontsize=28, fontweight='bold')
    plt.ylabel("{}".format(y), fontsize=28, fontweight='bold')
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    return ax

def barm_plot(x, y, x_label=None, *args, **kwargs):
    fig, ax = plt.subplots(figsize=(18, 12))
    sns.barplot(x, y, ci=None, palette='Paired', **kwargs)
    sns.swarmplot(x,y, color='k', linewidth=1.5, s=12, alpha=0.5, **kwargs)
    plt.xlabel("{}".format(x), fontsize=28, fontweight='bold')
    plt.ylabel("Proteins", fontsize=28, fontweight='bold')
    plt.xticks(rotation = 90, fontsize=28)
    plt.yticks(fontsize=28)
    return ax

# define a function for swarmplot 

def swarm_plot(x, y, x_label=None, *args, **kwargs):
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.swarmplot(x,y, color='lightblue', linewidth=1.5, s=12, alpha=0.7, **kwargs)
    plt.xlabel("{}".format(x), fontsize=28, fontweight='bold')
    plt.ylabel("Proteins", fontsize=28, fontweight='bold')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=28)
    return ax

# define a function for multiplot with colorbar 

def multi_plot(x, y, z, *args, **kwargs):
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.scatter(x, y, z, cmap='Blues', **kwargs)
    plt.colorbar()
    cbar.set_label("{}".format(z), fontsize=28)
    plt.xlabel("{}".format(x), fontsize=28, fontweight='bold')
    plt.ylabel("{}".format(y), fontsize=28, fontweight='bold')
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    return ax 



# # 1. Sample Preparation Methods and Proteome Coverage

# plot 

barm_plot('SamplePreparation_Method', 'Number_of_Proteins_Identified', data=data, edgecolor='k')

# save figure
#fig.tight_layout()
plt.show()


# plot 
fig, ax = plt.subplots(figsize = (14,10))

# define data to be used
a = data['SamplePreparation_Method']
b = data['Number_of_Proteins_Identified']

# create a scatter 
col= plt.scatter(a,b,c = data["Year_of_Study"], cmap='viridis_r', s=200, alpha=0.5, edgecolor= 'k', linewidth=1.2)

# set axis
#ax.set_xlim([-20,300]) 

# set label 
ax.set_xlabel('', fontsize=15, fontweight='bold')
ax.set_ylabel('Proteins', fontsize=28, fontweight='bold')

plt.xticks(color='black', rotation=90, fontsize='28', horizontalalignment='center')
plt.yticks(color='black', rotation=0, fontsize='28', horizontalalignment='right')

# create a colorbar 
cbar = plt.colorbar(col)
cbar.set_label('Year', fontsize = 20, fontweight='bold')
cbar.ax.tick_params(labelsize=20)

# save figure
fig.tight_layout()
plt.show()


# # 2. Sample Quantity and Proteome Coverage

# plot           
fig, ax = plt.subplots(figsize = (26,16))

ax=sns.swarmplot(x='Sample_Quantity', y='Number_of_Proteins_Identified' , hue='SamplePreparation_Method', hue_order = ['ISD', 'iST', 'FASP', 'SISPROT', 'MED-FASP','SP3','In-Gel'], data=df_sorted, palette='Paired', s=20, alpha=1, linewidth=1.2)

# set label
ax.set_xlabel('Quantity (ug)', fontsize=28, fontweight='bold')
ax.set_ylabel('Proteins', fontsize=28, fontweight='bold')
#ax.set_xlim([-10,1000])

plt.xticks(color='black', rotation=0, fontsize='28', horizontalalignment='center')
plt.yticks(color='black', rotation=0, fontsize='28', horizontalalignment='right')

# add legend
ax.legend(title='', loc='lower right',
          fancybox=True, shadow=True, ncol=1, fontsize=28)

 
# save figure
fig.tight_layout()
plt.show()



# plot      
fig, ax = plt.subplots(figsize = (26,16))


ax=sns.swarmplot(x='Sample_Quantity', y='Number_of_Proteins_Identified', hue='Protease', hue_order = ['Trypsin', 'LysC-Trypsin', 'Multi-enzyme'], data=df_sorted, palette=['tab:blue', 'tab:orange','tab:cyan'], s=20, alpha=1, linewidth=1.2)

# set label
ax.set_xlabel('Quantity (ug)', fontsize=28, fontweight='bold')
ax.set_ylabel('Proteins', fontsize=28, fontweight='bold')
#ax.set_xlim([-10.0,500.0])

plt.xticks(color='black', rotation=0, fontsize='28', horizontalalignment='center')
plt.yticks(color='black', rotation=0, fontsize='28', horizontalalignment='right')

# add legend
ax.legend(title='', loc='best',
          fancybox=True, shadow=True, ncol=1, fontsize=28)


# save figure
fig.tight_layout()
plt.show()


# # 3. Protein Reduction and Alkylation and Proteome Coverage

fig, ax = plt.subplots(figsize = (12,10))

ax = sns.scatterplot(data=df_sorted, x='Reduction_Concentration ', y="Number_of_Proteins_Identified", hue='Reducing_Agent', style = 'Alkylating_Agent', markers = ['o', 'v', '*'], s = 180, alpha = 0.6, edgecolor="black", linewidth = 1.2)

# set label
ax.set_xlabel('Reduction Concentration (mM)', fontsize=28, fontweight='bold')
ax.set_ylabel('Proteins', fontsize=28, fontweight='bold')
#ax.set_xlim([1,15.0])

plt.xticks(color='black', rotation=0, fontsize='28', horizontalalignment='center')
plt.yticks(color='black', rotation=0, fontsize='28', horizontalalignment='right')

# add legend
ax.legend(title='', loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True, ncol=1, fontsize=20)


plt.show()

# plot
fig, ax = plt.subplots(figsize = (12,10))

ax = sns.scatterplot(data=df_sorted, x='Reduction_Concentration ', y="Number_of_Proteins_Identified", hue='Reducing_Agent', style = 'Alkylating_Agent', markers = ['o', 'v', '*'], s = 180, alpha = 0.6, edgecolor="black", linewidth = 1.2)

# set label
ax.set_xlabel('Reduction Concentration (mM)', fontsize=28, fontweight='bold')
ax.set_ylabel('Proteins', fontsize=28, fontweight='bold')
ax.set_xlim([1,10.5])

plt.xticks(color='black', rotation=0, fontsize='28', horizontalalignment='center')
plt.yticks(color='black', rotation=0, fontsize='28', horizontalalignment='right')

# add legend
ax.legend(title='', loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True, ncol=1, fontsize=20)


plt.show()

#plot
fig, ax = plt.subplots(figsize = (12,10))

ax = sns.scatterplot(data=df_sorted, x="Alkylation_Concentration", y="Number_of_Proteins_Identified", hue='Reducing_Agent', style = 'Alkylating_Agent', markers = ['o', 'v', '*'], s = 180, alpha = 0.6, edgecolor="black", linewidth = 1.2)

# set label
ax.set_xlabel('Alkylation Concentration (mM)', fontsize=28, fontweight='bold')
ax.set_ylabel('Proteins', fontsize=28, fontweight='bold')
ax.set_xlim([1,115.0])

plt.xticks(color='black', rotation=0, fontsize='28', horizontalalignment='center')
plt.yticks(color='black', rotation=0, fontsize='28', horizontalalignment='right')

# add legend
ax.legend(title='', loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True, ncol=1, fontsize=20)


plt.show()


# plot
fig, ax = plt.subplots(figsize = (18,12))

ax = sns.scatterplot(data=df_sorted, x="Sample_Quantity", y="Number_of_Proteins_Identified", hue="Reduction_Concentration ", size="Alkylation_Concentration", palette = 'viridis_r', legend='full',  sizes=(20, 1000), alpha = 0.7, edgecolor="black", linewidth = 1.2)

# set label
ax.set_xlabel('Quantity (ug)', fontsize=28, fontweight='bold')
ax.set_ylabel('Proteins', fontsize=28, fontweight='bold')
ax.set_xlim([-10.0,330.0])

plt.xticks(color='black', rotation=0, fontsize='28', horizontalalignment='center')
plt.yticks(color='black', rotation=0, fontweight='bold', fontsize='28', horizontalalignment='right')

# add legend
ax.legend(title='', loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True, ncol=1, fontsize=20)


# save figure
fig.tight_layout()

plt.show()


# # 4. Protease and Proteome Coverage

# plot 
box_plot("Number_of_Proteins_Identified", "Protease", data=data, palette='Blues')

# save figure
fig.tight_layout()
plt.show()


# plot
fig, ax = plt.subplots(figsize = (10,8))

ax=sns.swarmplot(x='Protease', y="Number_of_Proteins_Identified", order=['Trypsin','LysC-Trypsin'] , hue='SamplePreparation_Method', hue_order = ['ISD', 'iST', 'FASP', 'SISPROT', 'MED-FASP', 'SP3', 'In-Gel'], data=data, palette='Paired', s=12, alpha=0.7, linewidth=1.2)

# set label
ax.set_xlabel('', fontsize=28, fontweight='bold')
ax.set_ylabel('Proteins', fontsize=28, fontweight='bold')


plt.xticks(color='black', rotation=0, fontsize='28', horizontalalignment='center')
plt.yticks(color='black', rotation=0, fontsize='28', horizontalalignment='right')

# add legend
ax.legend(title='', loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True, ncol=1, fontsize=20)


# save figure
fig.tight_layout()

plt.show()


# plot 
fig, ax = plt.subplots(figsize = (14,12))

# define data to be used
a = df_sorted['Protease']
b = df_sorted['Number_of_Proteins_Identified']

# create a scatter 
col= plt.scatter(a,b,c = df_sorted["Year_of_Study"], cmap='viridis_r', s=200, alpha=0.5, edgecolor= 'k', linewidth=1.2)

# set axis
#ax.set_xlim([-20,300]) 

# set label 
ax.set_xlabel('', fontsize=15, fontweight='bold')
ax.set_ylabel('Proteins', fontsize=28, fontweight='bold')

plt.xticks(color='black', rotation=90, fontsize='20', horizontalalignment='center')
plt.yticks(color='black', rotation=0, fontsize='28', horizontalalignment='right')

# create a colorbar 
cbar = plt.colorbar(col)
cbar.set_label('Year', fontsize = 20, fontweight='bold')
cbar.ax.tick_params(labelsize=20)

# save figure
fig.tight_layout()
plt.show()


# # 5. Fractionation Method and Proteome Coverage


# plot 
box_plot('Number_of_Proteins_Identified', "Fractionation_Method", data=df_sorted, palette='Blues')


# save figure
fig.tight_layout()
plt.show()


# # 6. Fraction Number and Proteome Coverage

#plot
fig, ax = plt.subplots(figsize = (18,12))

ax = sns.scatterplot(data=df_sorted, x="Fractionation_Method", y="Number_of_Proteins_Identified", size="Fraction_Number", hue = "SamplePreparation_Method", palette = 'Paired', legend='full',  sizes=(100, 1000), alpha = 0.6, edgecolor="black", linewidth = 1.2)

# set label
ax.set_xlabel('', fontsize=28, fontweight='bold')
ax.set_ylabel('Proteins', fontsize=28, fontweight='bold')
#ax.set_xlim([-10.0,500.0])

plt.xticks(color='black', rotation=90, fontsize='28', horizontalalignment='center')
plt.yticks(color='black', rotation=0, fontsize='28', horizontalalignment='right')

# add legend
ax.legend(title='', loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True, ncol=1, fontsize=20)


# save figure
fig.tight_layout()

plt.show()


# plot 
# select data with fraction number >1
df_filtered = df_sorted.loc[data['Fraction_Number'] > 1, :]
bar_plot("Fraction_Number", "Proteins/Fraction", data=df_filtered, ci=None, edgecolor='k', linewidth=1.5, color='lightsteelblue')

# save figure 
fig.tight_layout()

plt.show()



# # 7. Desalting Method and Proteome Coverage

# plot 

barm_plot('Desalting_Method ', 'Number_of_Proteins_Identified', data=df_sorted, edgecolor='k')
#plt.xticks(fontsize=15, rotation= 0)

# save figure
fig.tight_layout()


plt.show()


# # 8. LCMS gradient and Proteome Coverage


fig, ax = plt.subplots(figsize = (14,10))

ax = sns.scatterplot(data=df_sorted, x="LC_Gradient", y="Number_of_Proteins_Identified", size="Fraction_Number", hue = "Fraction_Number", palette = 'viridis_r', legend=None,  sizes=(100, 1000), alpha = 0.5, edgecolor="black", linewidth = 1.2)

# set label
ax.set_xlabel('LC Gradient (mins)', fontsize=28, fontweight='bold')
ax.set_ylabel('Proteins', fontsize=28, fontweight='bold')
ax.set_xlim([0,270])

plt.xticks(color='black', rotation=0, fontsize='28', horizontalalignment='center')
plt.yticks(color='black', rotation=0, fontsize='28', horizontalalignment='right')


# save figure
fig.tight_layout()

plt.show()


# # 9. Mass Spectrometers and Proteome Coverage


# plot 
box_plot("Number_of_Proteins_Identified", "Mass_Spectrometer", data=df_sorted, palette='Blues')


# save figure
fig.tight_layout()
plt.show()


# # 10. Data Acquisation and Proteome Coverage

fig, ax = plt.subplots(figsize = (18,14))

ax = sns.scatterplot(data=df_sorted, x="Mass_Spectrometer", y="Number_of_Proteins_Identified", hue="Data_Acquisition_mode ", size="Fraction_Number", palette =['pink','cyan', 'purple'], legend='full',  sizes=(100, 1000), alpha = 0.7, edgecolor="black", linewidth = 1.2)

# set label
ax.set_xlabel('', fontsize=15, fontweight='bold')
ax.set_ylabel('Proteins', fontsize=28, fontweight='bold')
#ax.set_xlim([-10.0,500.0])

plt.xticks(color='black', rotation=90, fontsize='20', horizontalalignment='center')
plt.yticks(color='black', rotation=0, fontsize='28', horizontalalignment='right')

# add legend
ax.legend(title='', loc='center left', bbox_to_anchor=(1.05, 0.5),
          fancybox=True, shadow=True, ncol=1, fontsize=20)


# save figure
fig.tight_layout()

plt.show()


# # Identifying key contributors to Proteome Coverage using LASSO feature selection

# copy data
new_df = df_sorted.copy()

new_df["LC_Gradient"] = new_df["LC_Gradient"].fillna(0.0).astype('int64')


## Correlation Analysis


import pandas as pd
import seaborn as sns

# compute the correlation matrix
corr_matrix = new_df.corr()

plt.figure(figsize=(8, 10))

# plot the correlation matrix as a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f', linewidths=.5, cbar=False, 
            xticklabels=corr_matrix.columns.values, yticklabels=corr_matrix.columns.values)

# set the plot title and axis labels

plt.title('Correlation Matrix')
plt.xlabel('Features')
plt.ylabel('Features')

# display the plot
#plt.show()
plt.savefig('corr_plot.svg', dpi=300, bbox_inches='tight')


# Linear regression model as baseline

# drop columns that will not be assessed in the LASSO-LARS
new_df.drop(['Cell_Line', 'LC_System', 'Data_Acquisition_mode ', 'Proteins/ug', 'Proteins/Fraction'], axis=1, inplace=True)

# create a map for ordinal encoding
mymap = {'ISD':1, 'iST':2, 'FASP':3, 'SISPROT':4, 'MED-FASP':5, 'SP3':6, 'In-Gel':7,
       'SPEED':8, 'fa-SPEED':9, 'STrap':10, 'UPS3':11, 'SP4':12, 'SiTrap':13,
        'TCEP':1, 'DTT':2, 'CAA':1, 'IAA':2, 'DEA':3,
        'Trypsin':1, 'LysC-Trypsin':2, 'LysC':3, 'Multi-enzyme':4, 'GluC':5,'Chymotrypsin':6, 'AspN':7, 'Chymotrypsin-Trypsin':8, 'ArgC-Trypsin':9,
        'HpH':1, 'Brplc':2, 'SAX':3, 'GELFrEE':4, 'SCX':5, 'SDS-PAGE':6, 'Unfractionated':7, 'IF':8, 'HpH-CIF':9, 'CIF':10, 'SP3':11,
        'SepPaks':1, 'StageTips':2, 'SP3':3, 'MCX':4, 'Pierce C18 Tips':5, 'ZipTips':6, 'C18 Tips':7, 'SP4':8,
        'Q Exactive HF':1, 'Orbitrap Fusion':2, 'Orbitrab Fusion':2, 'LTQ Orbitrap Velos':3, 'Q Exactive':4, 'Impact II QTOF ':5, 'Triple-TOF':6, 'Orbitrap Eclipse':7, 'LTQ Orbitrap XL':8, 'Synapt G2-S':9, 'Impact II':10, 'Orbitrap Velos Pro ':11, 'Q Exactive Plus':12, 'LTQ Orbitrap':13, 'maXis Impact':14}

# apply map
new_df = new_df.applymap(lambda s: mymap.get(s) if s in mymap else s)

new_df = new_df.dropna()


# import packages for modelling

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

# drop the number of proteins as it is not required here
Xs = new_df.drop(['Number_of_Proteins_Identified'], axis=1)
y = new_df['Number_of_Proteins_Identified'].values.reshape(-1,1)

# linear regression 
lin_reg = LinearRegression()

MSEs = cross_val_score(lin_reg, Xs, y, scoring='neg_mean_squared_error', cv=5)

mean_MSE = np.mean(MSEs)

print(mean_MSE)


# Hyperparameter Tunning with 5-fold cross validation 

# Compare LASSO with LASSO-LARS

# Lasso parameters

from sklearn.linear_model import LassoLars

lassolars = LassoLars()

parameters = {'alpha': [1e-15,1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}

lassolars_regressor = GridSearchCV(lassolars, parameters, scoring = 'neg_mean_squared_error', cv=5)

lassolars_regressor.fit(Xs, y)


print(lassolars_regressor.best_params_)
print(lassolars_regressor.best_score_)



# Lasso parameters

from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15,1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}

lasso_regressor = GridSearchCV(lasso, parameters, scoring = 'neg_mean_squared_error', cv=5)

lasso_regressor.fit(Xs, y)


print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# train test
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoLars
from sklearn.metrics import mean_squared_error

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)

# Train Lasso model on training set
lasso_model = Lasso(alpha=20)
lasso_model.fit(X_train, y_train)

# Evaluate Lasso model on test set
y_pred_lasso = lasso_model.predict(X_test)
lasso_mse = mean_squared_error(y_test, y_pred_lasso)
print("Lasso Test MSE:", lasso_mse)

# Train LassoLars model on training set
lars_model = LassoLars(alpha=1e-15)
lars_model.fit(X_train, y_train)

# Evaluate LassoLars model on test set
y_pred_lars = lars_model.predict(X_test)
lars_mse = mean_squared_error(y_test, y_pred_lars)
print("LassoLars Test MSE:", lars_mse)


# Evaluate Lasso model on test set and get R2 value
lasso_r2 = lasso_model.score(X_test, y_test)
print("Lasso R2 score:", lasso_r2)

# Evaluate LassoLars model on test set and get R2 value
lars_r2 = lars_model.score(X_test, y_test)
print("LassoLars R2 score:", lars_r2)


# Final LASSO-LARS model


import numpy as np
from sklearn import linear_model

# add numerical columns

scores_ = ['SamplePreparation_Method', 'Sample_Quantity',
           'Reduction_Concentration ', 'Reducing_Agent',
           'Alkylation_Concentration', 'Alkylating_Agent', 'Protease',
           'Fractionation_Method', 'Fraction_Number', 'Desalting_Method ',
           'LC_Gradient', 'Mass_Spectrometer', 'Year_of_Study']

X = new_df[scores_].values
y = new_df["Number_of_Proteins_Identified"].values


X = X / X.std(axis=0) # Standardize data (easier to set the l1_ratio parameter)

print("Computing regularization path using the LARS ...")
LL = linear_model.LassoLars(alpha=1e-15, fit_intercept=False)
LL.fit(X, y)

print(LL.coef_path_)

print(scores_)
print([f"{i:+5.3f}" for i in LL.coef_])

xx = np.sum(np.abs(LL.coef_path_.T), axis=1)
xx /= xx[-1]


# Plot 
fig, ax = plt.subplots(figsize=(20,12))
ax.plot(xx, LL.coef_path_.T)

# set axis
ymin, ymax = ax.get_ylim()
xmin, xmax = ax.get_xlim()
ax.set_xlim(xmin, xmax*1.4)

# add lines
ax.vlines(xx, ymin, ymax, linestyle='--')

# set labels
ax.set_xlabel('|coef| / max|coef|', fontsize=28, fontweight='bold')
ax.set_ylabel('Coefficients', fontsize=28, fontweight='bold')
#ax.set_title('LARS LASSO Path', fontsize=20, fontweight='bold')

plt.xticks(fontsize=28, fontweight='bold')
plt.yticks(fontsize=28, fontweight='bold')

# add legend 
fig.legend(scores_, loc='right', bbox_to_anchor=(0.98, 0.5), fancybox=True, shadow=True, ncol=1, fontsize=20)

# save figure
fig.tight_layout()

plt.show()
#plt.savefig('lasssolars_plot.svg', dpi=300, bbox_inches='tight')


# Feature Importance

# get feature importances
importances = LL.coef_

# get feature names
feature_names = new_df.drop('Number_of_Proteins_Identified', axis=1).columns

# create a dictionary with feature names and their importance
feature_importances = dict(zip(feature_names, importances))

# sort the dictionary by importance
sorted_importances = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)

# print out the top 10 important features
print("Top 10 important features:")
for i in range(13):
    print(f"{sorted_importances[i][0]}: {sorted_importances[i][1]:.4f}")



# get feature importances
importances = LL.coef_

# get feature names
feature_names = new_df.drop('Number_of_Proteins_Identified', axis=1).columns

# create a dictionary with feature names and their importance
feature_importances = dict(zip(feature_names, importances))

# sort the dictionary by importance
sorted_importances = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)

# convert the list to a DataFrame
importances_df = pd.DataFrame(sorted_importances, columns=['feature', 'importance'])

# sort the DataFrame by importance
importances_df = importances_df.sort_values(by='importance', ascending=True)

# plot the feature importances
plt.figure(figsize=(6, 8))
plt.barh(importances_df['feature'], importances_df['importance'])
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
#plt.show()

fig.tight_layout(pad=5.5)
#plt.savefig('lasssolars_importance_plot.svg', dpi=300, bbox_inches='tight')



# Comparison to Elastic Net 
# import packages
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# set up ElasticNet model
enet = ElasticNet()

# set up parameter grid for hyperparameter tuning
param_grid = {
    'alpha': np.logspace(-5, 1, 5),
    'l1_ratio': [0.2, 0.5, 0.8],
    'max_iter': [1000, 5000, 10000],
    'tol': [1e-4, 1e-5, 1e-6]
}

# perform grid search to find best hyperparameters
grid_search = GridSearchCV(enet, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X, y)

# print best hyperparameters and score
print("Best parameters:", grid_search.best_params_)
print("Best score:", -grid_search.best_score_)

# fit model with best hyperparameters
enet_best = ElasticNet(**grid_search.best_params_)
enet_best.fit(X, y)

# print feature importances
coefficients = enet_best.coef_
feature_names = new_df.drop(["Number_of_Proteins_Identified"], axis=1).columns
for feature, coef in zip(feature_names, coefficients):
    print(feature, coef)


# get feature importances
importances = enet_best.coef_

# get feature names
feature_names = new_df.drop('Number_of_Proteins_Identified', axis=1).columns

# create a dictionary with feature names and their importance
feature_importances = dict(zip(feature_names, importances))

# sort the dictionary by importance
sorted_importances = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)

# convert the list to a DataFrame
importances_df = pd.DataFrame(sorted_importances, columns=['feature', 'importance'])

# sort the DataFrame by importance
importances_df = importances_df.sort_values(by='importance', ascending=True)

# plot the feature importances
plt.figure(figsize=(6, 8))
plt.barh(importances_df['feature'], importances_df['importance'])
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
#plt.show()

#fig.tight_layout(pad=5.5)
plt.savefig('elasticnet_importance_plot.svg', dpi=300, bbox_inches='tight')
