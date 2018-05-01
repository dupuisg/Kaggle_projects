import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
import datetime

def LRMSerror(y1, y2):
    err = np.sqrt(np.mean((np.log(y1+1) - np.log(y2+1))**2))
    return err

def MARerror(y1, y2):
#error relative to y2
#y2 never equals 0 - not necessary but adjust for this later
    err = np.mean(np.abs(y1 - y2)/y2)
    return err

##For mapping discrete data to integer values - for discrete numerical features, not categorical  
def map_discrt(x, label_keys):
    for r in label_keys:
        x[x == r] = label_keys[r]
    x[x == 'NA'] = -1

##Label encoding for categorical data, mapping category to integer value
##Returns 1D array of transformed data and dictionary with label:category pairs 
def categ_toint(x):
    #define set of unique items
    categories = np.unique(x)
    codes = {}
    #loop over, mapping to integers
    for i, cat in enumerate(categories):
        x[x==cat] = i
        codes[i] = cat
    return codes, x

def get_data(datafile, num_feats, discrt_feats, cat_feats, rat_codes):
    X_num = np.genfromtxt(datafile, usecols=num_feats, delimiter=',', skip_header=1, missing_values="NA")
    X_discrt = np.loadtxt(datafile, dtype=str, usecols=discrt_feats, delimiter=',', skiprows=1)
    X_categ = np.loadtxt(datafile, dtype=str, usecols=cat_feats, delimiter=',', skiprows=1)

    #Redefine years as ages
    current_yr = datetime.datetime.now().year
    X_num[:, (4,5,23)] = current_yr - X_num[:, (4,5,23)]

    #Map discrete data labels to numerical ratings
    for idx in range(X_discrt.shape[1]):
        map_discrt(X_discrt[:, idx], rat_codes[idx])
    
    #Label encoding for categorical data
    categ_codes = {}
    for idx in range(X_categ.shape[1]):
        categ_codes[idx], X_categ[:,idx] = categ_toint(X_categ[:, idx])

    X_discrt = X_discrt.astype(float)
    X_categ = X_categ.astype(float)

    #Manually set nans
    #print set(np.where(np.isnan(X))[1])
    X_num[np.isnan(X_num[:,0]), 0] = 0
    X_num[np.isnan(X_num[:,6]), 6] = 0
    X_num[np.isnan(X_num[:,23]),23] = -1

    return X_num, X_discrt, X_categ, categ_codes

##Retrieve original feature name given column number, corresponding to data set as passed to fits
def get_feat_name(i, headings, num_feats, discrt_feats, cat_feats, categ_codes, ft_indices, active_fts):
    N_num, N_discrt, N_cat = len(num_feats), len(discrt_feats), len(cat_feats)
    N_enc = len(active_fts)
    if i >= N_enc and i < N_enc + N_num: ##Numerical feature
        idx = i - N_enc
        feature_name = headings[num_feats[idx]]
    if i >= N_enc + N_num: ##Discrete feature
        idx = i - N_enc - N_num
        feature_name = headings[discrt_feats[idx]]
    if i < N_enc: ##Categorical feature 
        col = active_fts[i]
        for j in range(len(ft_indices)-1):
            l, r = ft_indices[j], ft_indices[j+1]
            if col >= l and col < r:
                feature_name = headings[cat_feats[j]]
                label_idx = col - l
                feature_name = feature_name + '-' + categ_codes[j][label_idx]
                break
    return feature_name

plotdir = 'plots/'

with open('train.csv') as f:
    headings = f.readline().strip().split(',')

#for i, title in enumerate(headings):
#  print i, title

num_feats = [3,4,17,18,19,20,26,34,36,37,38,43,44,45,46,47,48,49,50,51,52,54,56,59,61,62,66,67,68,69,70,71,75]
discrt_feats = [7,11,27,28,30,31,32,33,35,40,53,55,57,60,63,64,65,72]
cat_feats = [1,2,5,6,8,9,10,12,13,14,15,16,21,22,23,24,25,29,39,41,42,58,73,74,76,77,78,79]
#
rat_codes = {0:{'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1}, 1:{'Gtl':3, 'Mod':2, 'Sev':1}, 6:{'Gd':4, 'Av':3, 'Mn':2, 'No':1},
11:{'Typ':8, 'Min1':7, 'Min2':6, 'Mod':5, 'Maj1':4, 'Maj2':3, 'Sev':2, 'Sal':1}, 13:{'Fin':3, 'RFn':2, 'Unf':1},
16:{'Y':3, 'P':2, 'N':1}, 17:{'Ex':4, 'Gd':3, 'TA':2, 'Fa':1}}
rat_codes.update(dict.fromkeys([7,8], {'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1}))
rat_codes.update(dict.fromkeys([2,3,4,5,9,10,12,14,15], {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1}))

X_num, X_discrt, X_categ, categ_codes = get_data('train.csv', num_feats, discrt_feats, cat_feats, rat_codes)
X = np.hstack((X_num, X_discrt, X_categ))
y = np.loadtxt('train.csv', usecols=(-1,), delimiter=',', skiprows=1)

Nfeat_num, Nfeat_discrt, Nfeat_cat = X_num.shape[1], X_discrt.shape[1], X_categ.shape[1]
numerical_cols = range(Nfeat_num)
discrete_cols = range(Nfeat_num, Nfeat_num + Nfeat_discrt)
categorical_cols = range(Nfeat_num + Nfeat_discrt, X.shape[1])

## Split into training and cross-validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.7, random_state=25)

hist, bins = np.histogram(y, bins=50)
mode = bins[np.argmax(hist)]
#
plt.figure()
min_price, max_price = y.min(), y.max()
plt.hist(y_train, bins=50)
plt.xlabel('Price ($)')
plt.savefig(plotdir + 'prices_dist.png')
plt.close()

## Scale numerical features and encode categorical
scaler = StandardScaler()
scaler.fit(X_train[:, 0:Nfeat_num])
X_train[:, 0:Nfeat_num] = scaler.transform(X_train[:, 0:Nfeat_num])
#
enc = OneHotEncoder(categorical_features=categorical_cols)
enc.fit(X)
Xtrain_enc = enc.transform(X_train)

##Scale and encode cross-validation set
Xval_scaled = scaler.transform(X_val[:, 0:Nfeat_num])
X_val = np.hstack((Xval_scaled, X_val[:, Nfeat_num:]))
Xval_enc = enc.transform(X_val)

## Standard linear regression 
ln_reg = LinearRegression(fit_intercept=True, normalize=False)
ln_reg.fit(Xtrain_enc, y_train)
#
pred = ln_reg.predict(Xval_enc)
err = LRMSerror(pred,y_val)
rel_err = MARerror(pred,y_val)
print 'Results for linear regresion......'
print 'Log-root-mean-squared error', err
print 'Mean absolute relative error', rel_err
#
plt.figure()
plt.hist(np.sqrt((np.log(pred+1) - np.log(y_val+1))**2), bins=50, color='forestgreen')
plt.savefig(plotdir + 'LR_lrmse_dist.png')
plt.close()
#
plt.figure()
plt.hist((pred - y_val)/y_val, bins=50, color='springgreen')
plt.savefig(plotdir + 'LR_relErr_dist.png')
plt.close()
#
plt.figure()
plt.plot(range(Xtrain_enc.shape[1]), ln_reg.coef_, linestyle='none', marker='+', color='blue')
plt.savefig(plotdir + 'coefficients.png')
plt.close()
#
##Print out the features with largest coefficients (magnitude)
'''
foi = np.argsort(np.abs(ln_reg.coef_))[-20:]
#
for i in foi:
    feature_name = get_feat_name(i, headings, num_feats, discrt_feats, cat_feats, categ_codes, enc.feature_indices_, enc.active_features_)
    if i < len(enc.active_features_):
        num_active = np.sum(Xtrain_enc.todense()[:, i])
        print i, feature_name, ln_reg.coef_[i], num_active
    else:
        print i, feature_name, ln_reg.coef_[i]
'''

## Random forest regression 
'''
N_trees = np.arange(50, 850, 50)
N_maxft = np.arange(25, 275, 25)
lrmse = []
for m in N_maxft: #[100]
    for N in N_trees: #[300]
        print '...............Now running with', N, 'trees and max_features =', m, '...............'
        rf_reg = RandomForestRegressor(n_estimators=N, max_features=m, random_state=101)  
        rf_reg.fit(Xtrain_enc, y_train)
        #
        pred = rf_reg.predict(Xval_enc)
        err = LRMSerror(pred,y_val) 
        lrmse += [err]
        print 'log-root-mean-squared error', err
#
#plt.figure()
#plt.plot(N_maxft, lrmse, color='m')
#plt.savefig(plotdir + 'RF_maxfeatures.png')
#plt.close()
#
lrmse = np.array(lrmse).reshape((len(N_maxft), len(N_trees)))
plt.figure()
plt.contourf(N_trees, N_maxft, lrmse, 100, cmap=plt.cm.viridis)
plt.colorbar()
plt.xlabel('No. of trees')
plt.ylabel('Max no. of features')
plt.savefig(plotdir + 'RF_parameter_scan.png')
plt.close()
#
'''
rf_reg = RandomForestRegressor(n_estimators=500, max_features=75, random_state=101)  
rf_reg.fit(Xtrain_enc, y_train)
#
pred = rf_reg.predict(Xval_enc)
err = LRMSerror(pred,y_val)
rel_err = MARerror(pred,y_val)
print 'Results for random forest regresion......'
print 'Log-root-mean-squared error', err
print 'Mean absolute relative error', rel_err
#
plt.figure()
plt.hist(np.sqrt((np.log(pred+1) - np.log(y_val+1))**2), bins=50, color='mediumblue')
plt.savefig(plotdir + 'RF_lrmse_dist.png')
plt.close()
#
plt.figure()
plt.hist((pred - y_val)/y_val, bins=50, color='cornflowerblue')
plt.savefig(plotdir + 'RF_relErr_dist.png')
plt.close()

## Gradient boost regression
'''
lrmse = []
learning_rates = [0.1, 0.075, 0.05, 0.025, 0.01, 5.0e-3, 1.0e-3] 
N_estimators = np.arange(100, 2100, 100) 
depths = np.arange(1, 11)  
max_feats = np.arange(20, 260, 10) 
leaves = np.arange(1,11)
for l in leaves:
    for d in depths:
        print '...............Now running with min leaf samples =', l, 'and max depth =', d, '...............'
        gb_reg = GradientBoostingRegressor(learning_rate=0.05, n_estimators=700, max_features=100, max_depth=d, min_samples_leaf=l,random_state=200)  
        gb_reg.fit(Xtrain_enc, y_train)
        #
        pred = gb_reg.predict(Xval_enc)
        err = LRMSerror(pred, y_val)
        lrmse += [err]
        print 'log-root-mean-squared error', err
#
#plt.figure()
#plt.plot(max_feats, lrmse, color='navy')
#plt.xlabel('max no. of features')
#plt.ylabel('log RMS error')
#plt.savefig(plotdir + 'GB_maxfeatures.png')
#plt.close()
#
lrmse = np.array(lrmse).reshape((len(leaves), len(depths)))
plt.figure()
plt.contourf(depths, leaves, lrmse, 100, cmap=plt.cm.viridis)
plt.colorbar()
plt.xlabel('maximum depth')
plt.ylabel('min. samples at leaf')
plt.savefig(plotdir + 'GB_parameter_scan.png')
plt.close()
'''

GBparams = {'loss':'ls', 'learning_rate':0.05, 'n_estimators':2000, 'max_depth':3, 'min_samples_leaf':2, 'max_features':100, 'random_state':201}

gb_reg = GradientBoostingRegressor(**GBparams)  
gb_reg.fit(Xtrain_enc, y_train)
#
pred = gb_reg.predict(Xval_enc)
err = LRMSerror(pred, y_val)
rel_err = MARerror(pred, y_val)
print 'Results for gradient boosting regression......'
print 'Log-root-mean-squared error', err
print 'Mean absolute relative error', rel_err
#
plt.figure()
plt.hist(np.sqrt((np.log(pred+1) - np.log(y_val+1))**2), bins=50, color='darkviolet')
plt.savefig(plotdir + 'GB_lrmse_dist.png')
plt.close()
#
plt.figure()
plt.hist((pred - y_val)/y_val, bins=50, color='mediumpurple')
plt.savefig(plotdir + 'GB_relErr_dist.png')
plt.close()
#
##Average score over several random trials
del GBparams['random_state']
errors = []
for i in range(10):
  gb_reg = GradientBoostingRegressor(**GBparams)
  gb_reg.fit(Xtrain_enc, y_train)
#
  pred = gb_reg.predict(Xval_enc)
  errors += [LRMSerror(pred,y_val)] 
print errors
print 'Mean error', np.mean(np.array(errors)), 'with std dev', np.std(np.array(errors))

GBparams['random_state']=200
LRpred = ln_reg.predict(Xval_enc)
RFpred = rf_reg.predict(Xval_enc)
GBpred = gb_reg.predict(Xval_enc)
#
plt.figure(figsize=(13,6))
plt.plot(LRpred, color='forestgreen', linestyle='none', marker='o', markersize=3, label='linear regr.')
plt.plot(RFpred, color='mediumblue', linestyle='none', marker='o', markersize=3, label='random forest regr.')
plt.plot(GBpred, color='darkviolet', linestyle='none', marker='o', markersize=3, label='gradient boost regr.')
plt.plot(y_val, color='k', linestyle='none', marker='o', markersize=3, label='true value')
plt.ylabel('price ($)')
plt.legend()
plt.savefig(plotdir + 'prediction_comparison.png')
plt.close() 

