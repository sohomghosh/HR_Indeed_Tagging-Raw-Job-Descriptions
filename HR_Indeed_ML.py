import numpy as np
import pandas as pd
from __future__ import division
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing, model_selection, metrics, ensemble
from sklearn.preprocessing import LabelEncoder
#import h2o
import xgboost as xgb

data=pd.DataFrame.from_csv("train.tsv",sep='\t', header=0,index_col=None)
tags_list=list(data['tags'])
descrp=list(data['description'])
feature_list=['part-time-job','full-time-job','hourly-wage','salary','associate-needed','bs-degree-needed','ms-or-phd-needed','licence-needed','1-year-experience-needed','2-4-years-experience-needed','5-plus-years-experience-needed','supervising-job','nan']
#y_train= pd.DataFrame(0, index=np.arange(len(data)), columns=feature_list)

# for i in list(y_train.index):
# 	tag=data.iloc[i,:]['tags']
# 	flag=0
# 	for j in range(0,len(feature_list)-1):
# 		if feature_list[j] in str(tag):
# 		    y_train.set_value(index=i,col=feature_list[j],value=1)
# 		    flag=1
# 	if flag==0:
# 		y_train.set_value(index=i,col=feature_list[12],value=1)
# y_train.to_csv("y_train.csv")

y_train=pd.read_csv("y_train.csv")
sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, max_features=3000, sublinear_tf=True)
sklearn_representation = sklearn_tfidf.fit_transform(descrp)
#tfidf=pd.DataFrame(data=sklearn_representation)
tfidf=pd.DataFrame(sklearn_representation.todense())
tfidf.to_csv("train_file_tfidf.csv",index=False)
x_train=tfidf


***************#Try with word2vec


data_test=pd.DataFrame.from_csv("test.tsv",sep='\t', header=0,index_col=None)
#sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, max_features=3000, sublinear_tf=True)
sklearn_representation_test = sklearn_tfidf.fit_transform(list(data_test['description']))
x_test=pd.DataFrame(sklearn_representation_test.todense())

#data_part_full = data[(y_train['part-time-job']==1) | (y_train['full-time-job']==1)] #1213
part=list(y_train['part-time-job']==1)
full=list(y_train['full-time-job']==1)
y_part_full=[]
#p=0 #328
#f=0 #885
#n=0 #3162
for i in range(len(part)):
	if part[i]==True:
		y_part_full.append(2)
#		p=p+1
	if full[i]==True:
		y_part_full.append(1)
#		f=f+1	
	if (part[i]==False) & (full[i]==False):
		y_part_full.append(0)
#		n=n+1

dtrain = xgb.DMatrix(x_train, y_part_full, missing=np.nan)
dtest = xgb.DMatrix(x_test, missing=np.nan)

params = {"objective": "multi:softmax","booster": "gbtree", "nthread": 4, "silent": 1,
                "eta": 0.08, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,
                "min_child_weight": 1, "num_class": 3,
                "seed": 2016, "tree_method": "exact"}

nrounds = 260
watchlist = [(dtrain, 'train')]
bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)
test_preds_part_full = bst.predict(dtest)
submit_part_full = pd.DataFrame({'tagsencodes': test_preds_part_full})
submit_part_full.to_csv("submit_part_full.csv",index=False)






#data_hourly_salary = data[(y_train['hourly-wage']==1) | (y_train['salary']==1)]
hourly=list(y_train['hourly-wage']==1)
salary=list(y_train['salary']==1)
y_hourly_salary=[]
for i in range(len(hourly)):
	if hourly[i]==True:
		y_hourly_salary.append(2)
	if salary[i]==True:
		y_hourly_salary.append(1)	
	if (hourly[i]==False) & (salary[i]==False):
		y_hourly_salary.append(0)

dtrain = xgb.DMatrix(x_train, y_hourly_salary, missing=np.nan)
dtest = xgb.DMatrix(x_test, missing=np.nan)

params = {"objective": "multi:softmax","booster": "gbtree", "nthread": 4, "silent": 1,
                "eta": 0.08, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,
                "min_child_weight": 1, "num_class": 3,
                "seed": 2016, "tree_method": "exact"}

nrounds = 260
watchlist = [(dtrain, 'train')]
bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)
test_preds_hourly_salary = bst.predict(dtest)
submit_hourly_salary = pd.DataFrame({'tagsencodes': test_preds_hourly_salary})
submit_hourly_salary.to_csv("submit_hourly_salary.csv",index=False)





associate = list(y_train['associate-needed']==1)
y_associate=[]
for i in range(len(associate)):
	if associate[i]==True:
		y_associate.append(1)
	else:
		y_associate.append(0)

dtrain = xgb.DMatrix(x_train, y_associate, missing=np.nan)
dtest = xgb.DMatrix(x_test, missing=np.nan)

params = {"objective": "multi:softmax","booster": "gbtree", "nthread": 4, "silent": 1,
                "eta": 0.08, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,
                "min_child_weight": 1, "num_class": 2,
                "seed": 2016, "tree_method": "exact"}

nrounds = 260
watchlist = [(dtrain, 'train')]
bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)
test_preds_associate = bst.predict(dtest)
submit_hourly_associate = pd.DataFrame({'tagsencodes': test_preds_associate})
submit_hourly_associate.to_csv("submit_associate.csv",index=False)



bs=list(y_train['bs-degree-needed']==1)
ms=list(y_train['ms-or-phd-needed']==1)
y_bs_ms=[]
for i in range(len(bs)):
	if bs[i]==True:
		y_bs_ms.append(1)
	if ms[i]==True:
		y_bs_ms.append(2)	
	if (bs[i]==False) & (ms[i]==False):
		y_bs_ms.append(0)

dtrain = xgb.DMatrix(x_train, y_bs_ms, missing=np.nan)
dtest = xgb.DMatrix(x_test, missing=np.nan)

params = {"objective": "multi:softmax","booster": "gbtree", "nthread": 4, "silent": 1,
                "eta": 0.08, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,
                "min_child_weight": 1, "num_class": 3,
                "seed": 2016, "tree_method": "exact"}

nrounds = 260
watchlist = [(dtrain, 'train')]
bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)
test_preds_degree = bst.predict(dtest)
submit_degree = pd.DataFrame({'tagsencodes': test_preds_degree})
submit_degree.to_csv("submit_degree.csv",index=False)



licence = list(y_train['licence-needed']==1)
y_licence=[]
for i in range(len(licence)):
	if licence[i]==True:
		y_licence.append(1)
	else:
		y_licence.append(0)


dtrain = xgb.DMatrix(x_train, y_licence, missing=np.nan)
dtest = xgb.DMatrix(x_test, missing=np.nan)

params = {"objective": "multi:softmax","booster": "gbtree", "nthread": 4, "silent": 1,
                "eta": 0.08, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,
                "min_child_weight": 1, "num_class": 2,
                "seed": 2016, "tree_method": "exact"}

nrounds = 260
watchlist = [(dtrain, 'train')]
bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)
test_preds_licence = bst.predict(dtest)
submit_licence = pd.DataFrame({'tagsencodes': test_preds_licence})
submit_licence.to_csv("submit_licence.csv",index=False)



#1-year-experience-needed or 2-4-years-experience-needed or 5-plus-years-experience-needed [3,2,1,0]
y_exp=[]
one_exp=list(y_train['1-year-experience-needed']==1)
two_exp=list(y_train['2-4-years-experience-needed']==1)
five_exp=list(y_train['5-plus-years-experience-needed']==1)
for i in range(len(one_exp)):
	if one_exp[i]==True:
		y_exp.append(1)
	elif two_exp[i]==True:
		y_exp.append(2)
	elif five_exp[i]==True:
		y_exp.append(3)
	else:
		y_exp.append(0)
			
dtrain = xgb.DMatrix(x_train, y_exp, missing=np.nan)
dtest = xgb.DMatrix(x_test, missing=np.nan)

params = {"objective": "multi:softmax","booster": "gbtree", "nthread": 4, "silent": 1,
                "eta": 0.08, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,
                "min_child_weight": 1, "num_class": 4,
                "seed": 2016, "tree_method": "exact"}

nrounds = 260
watchlist = [(dtrain, 'train')]
bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)
test_preds_experience = bst.predict(dtest)
submit_experience = pd.DataFrame({'tagsencodes': test_preds_experience})
submit_experience.to_csv("submit_experience.csv",index=False)





supervising = list(y_train['supervising-job']==1)
y_supervising=[]
for i in range(len(supervising)):
	if supervising[i]==True:
		y_supervising.append(1)
	else:
		y_supervising.append(0)

dtrain = xgb.DMatrix(x_train, y_supervising, missing=np.nan)
dtest = xgb.DMatrix(x_test, missing=np.nan)

params = {"objective": "multi:softmax","booster": "gbtree", "nthread": 4, "silent": 1,
                "eta": 0.08, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,
                "min_child_weight": 1, "num_class": 2,
                "seed": 2016, "tree_method": "exact"}

nrounds = 260
watchlist = [(dtrain, 'train')]
bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)
test_supervising = bst.predict(dtest)
submit_supervising = pd.DataFrame({'tagsencodes': test_supervising})
submit_supervising.to_csv("submit_supervising.csv",index=False)


#Ensemble of the above
y_test=[]
fp_out=open("tags.tsv",'w')
fp_out.write("tags\n")
for part,sal,asso,degree,licence,exp,supervising in zip(list(test_preds_part_full),list(test_preds_hourly_salary),list(test_preds_associate),list(test_preds_degree),list(test_preds_licence),list(test_preds_experience),list(test_supervising)):
	if part==2.0:
		fp_out.write("part-time-job\t")
	if part==1.0:
		fp_out.write("full-time-job\t")
	if sal==2.0:
		fp_out.write("hourly-wage\t")
	if sal==1.0:
		fp_out.write("salary\t")
	if asso==1.0:
		fp_out.write("associate-needed\t")
	if degree==2.0:
		fp_out.write("ms-or-phd-needed\t")		
	if degree==1.0:
		fp_out.write("bs-degree-needed\t")
	if licence==1.0:
		fp_out.write("licence-needed\t")
	if exp==3.0:
		fp_out.write("5-plus-years-experience-needed\t")
	if exp==2.0:
		fp_out.write("2-4-years-experience-needed\t")
	if exp==1.0:
		fp_out.write("1-year-experience-needed\t")					
	if supervising==1.0:
		fp_out.write("supervising-job ")
	fp_out.write("\n")


fp_out.close()




##Tuning XG-Boost
from sklearn.grid_search import GridSearchCV
cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic'}
optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), 
                            cv_params, 
                             scoring = 'accuracy', cv = 5, n_jobs = -1) 
optimized_GBM.fit(x_train, y_supervising)
GridSearchCV(cv=5, error_score='raise',
       estimator=XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.8,
       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3,
       min_child_weight=1, missing=None, n_estimators=1000, nthread=-1,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=0.8),
       fit_params={}, iid=True, n_jobs=-1,
       param_grid={'min_child_weight': [1, 3, 5], 'max_depth': [3, 5, 7]},
       pre_dispatch='2*n_jobs', refit=True, scoring='accuracy', verbose=0)
optimized_GBM.grid_scores_


cv_params = {'learning_rate': [0.1, 0.01], 'subsample': [0.7,0.8,0.9]}
ind_params = {'n_estimators': 1000, 'seed':0, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic', 'max_depth': 3, 'min_child_weight': 1}


optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), 
                            cv_params, 
                             scoring = 'accuracy', cv = 5, n_jobs = -1)
optimized_GBM.fit(x_train, y_supervising)
GridSearchCV(cv=5, error_score='raise',
       estimator=XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.8,
       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3,
       min_child_weight=1, missing=None, n_estimators=1000, nthread=-1,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=1),
       fit_params={}, iid=True, n_jobs=-1,
       param_grid={'subsample': [0.7, 0.8, 0.9], 'learning_rate': [0.1, 0.01]},
       pre_dispatch='2*n_jobs', refit=True, scoring='accuracy', verbose=0)
optimized_GBM.grid_scores_

xgdmat = xgb.DMatrix(x_train, y_supervising, missing=np.nan)

# Grid Search CV optimized settings from before use here
our_params = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic', 'max_depth':3, 'min_child_weight':1} 


cv_xgb = xgb.cv(params = our_params, dtrain = xgdmat, num_boost_round = 3000, nfold = 5,
                metrics = ['error'], # Make sure you enter metrics inside a list or you may encounter issues!
                early_stopping_rounds = 100) # Look for early stopping that minimizes error
cv_xgb.tail(5)


#Using best setting parameters from before
our_params = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic', 'max_depth':3, 'min_child_weight':1} 

final_gb = xgb.train(our_params, xgdmat, num_boost_round = 432)


#%matplotlib inline
import seaborn as sns
sns.set(font_scale = 1.5)
xgb.plot_importance(final_gb)
importances = final_gb.get_fscore()
importances
importance_frame = pd.DataFrame({'Importance': list(importances.values()), 'Feature': list(importances.keys())})
importance_frame.sort_values(by = 'Importance', inplace = True)
#change figure size (8,8) if needed
importance_frame.plot(kind = 'barh', x = 'Feature', figsize = (8,8), color = 'orange')

testdmat = xgb.DMatrix(X_test)
from sklearn.metrics import accuracy_score
y_pred = final_gb.predict(testdmat) # Predict using our testdmat
y_pred

y_pred[y_pred > 0.5] = 1
y_pred[y_pred <= 0.5] = 0
y_pred

accuracy_score(y_pred, y_test), 1-accuracy_score(y_pred, y_test)

#With inputs from https://jessesw.com/XG-Boost/





###Using h2o needs to be changed
import h2o
h2o.init() 
h2o.connect()


#train_h2o=h2o.import_file("train_file_h2o.csv")

#X_train_h20=h2o.H2OFrame(tfidf.values.tolist())
#Y_train_h2o=h2o.H2OFrame(y_train.values.tolist())


X_train_h2o=h2o.H2OFrame(x_train.values.tolist())
Y_train_h2o=h2o.H2OFrame(y_part_full,column_names=['response'],column_types=['categorical'])
train=X_train_h2o.cbind(Y_train_h2o)
test=h2o.H2OFrame(x_test.values.tolist())
r = train.runif()
train_split = train[r  < 0.8]
valid_split = train[r >= 0.8]
features = list(np.setdiff1d(train.names, ['response']))

from h2o.estimators.gbm import H2OGradientBoostingEstimator
model_gbm = H2OGradientBoostingEstimator(distribution='multinomial',
                                    ntrees=100,
                                    max_depth=4,
                                    learn_rate=0.1)
model_gbm.train(x=features, y="response", training_frame=train_split, validation_frame=valid_split)
print(model_gbm)
pred = model_gbm.predict(test)
pred.head()
submit_pred= pred['predict']
submit_pred.head()
#submission_dataframe =(test[:,'member_id']).cbind(submit_pred)
submit_pred.set_name(0,"tags")
h2o.h2o.export_file(submit_pred, path ="submission_rf_1.csv")




X_train_h2o=h2o.H2OFrame(x_train.values.tolist())
Y_train_h2o=h2o.H2OFrame(y_hourly_salary,column_names=['response'],column_types=['categorical'])
train=X_train_h2o.cbind(Y_train_h2o)
test=h2o.H2OFrame(x_test.values.tolist())
r = train.runif()
train_split = train[r  < 0.8]
valid_split = train[r >= 0.8]
features = list(np.setdiff1d(train.names, ['response']))

from h2o.estimators.gbm import H2OGradientBoostingEstimator
model_gbm = H2OGradientBoostingEstimator(distribution='multinomial',
                                    ntrees=100,
                                    max_depth=4,
                                    learn_rate=0.1)
model_gbm.train(x=features, y="response", training_frame=train_split, validation_frame=valid_split)
print(model_gbm)
pred = model_gbm.predict(test)
pred.head()
submit_pred_hourly_salary= pred['predict']
submit_pred_hourly_salary.head()
#submission_dataframe =(test[:,'member_id']).cbind(submit_pred)
submit_pred.set_name(0,"tags")
h2o.h2o.export_file(submit_pred_hourly_salary, path ="y_hourly_salary.csv")




X_train_h2o=h2o.H2OFrame(x_train.values.tolist())
Y_train_h2o=h2o.H2OFrame(y_associate,column_names=['response'],column_types=['categorical'])
train=X_train_h2o.cbind(Y_train_h2o)
test=h2o.H2OFrame(x_test.values.tolist())
r = train.runif()
train_split = train[r  < 0.8]
valid_split = train[r >= 0.8]
features = list(np.setdiff1d(train.names, ['response']))

from h2o.estimators.gbm import H2OGradientBoostingEstimator
model_gbm = H2OGradientBoostingEstimator(distribution='multinomial',
                                    ntrees=100,
                                    max_depth=4,
                                    learn_rate=0.1)
model_gbm.train(x=features, y="response", training_frame=train_split, validation_frame=valid_split)
print(model_gbm)
pred = model_gbm.predict(test)
pred.head()
submit_pred_associate= pred['predict']
submit_pred_associate.head()
#submission_dataframe =(test[:,'member_id']).cbind(submit_pred)
submit_pred_associate.set_name(0,"tags")
h2o.h2o.export_file(submit_pred, path ="associate.csv")





X_train_h2o=h2o.H2OFrame(x_train.values.tolist())
Y_train_h2o=h2o.H2OFrame(y_bs_ms,column_names=['response'],column_types=['categorical'])
train=X_train_h2o.cbind(Y_train_h2o)
test=h2o.H2OFrame(x_test.values.tolist())
r = train.runif()
train_split = train[r  < 0.8]
valid_split = train[r >= 0.8]
features = list(np.setdiff1d(train.names, ['response']))

from h2o.estimators.gbm import H2OGradientBoostingEstimator
model_gbm = H2OGradientBoostingEstimator(distribution='multinomial',
                                    ntrees=100,
                                    max_depth=4,
                                    learn_rate=0.1)
model_gbm.train(x=features, y="response", training_frame=train_split, validation_frame=valid_split)
print(model_gbm)
pred = model_gbm.predict(test)
pred.head()
submit_pred_bs_ms= pred['predict']
submit_pred_bs_ms.head()
#submission_dataframe =(test[:,'member_id']).cbind(submit_pred)
submit_pred_bs_ms.set_name(0,"tags")
h2o.h2o.export_file(submit_pred_bs_ms, path ="bs_ms.csv")







X_train_h2o=h2o.H2OFrame(x_train.values.tolist())
Y_train_h2o=h2o.H2OFrame(y_licence,column_names=['response'],column_types=['categorical'])
train=X_train_h2o.cbind(Y_train_h2o)
test=h2o.H2OFrame(x_test.values.tolist())
r = train.runif()
train_split = train[r  < 0.8]
valid_split = train[r >= 0.8]
features = list(np.setdiff1d(train.names, ['response']))

from h2o.estimators.gbm import H2OGradientBoostingEstimator
model_gbm = H2OGradientBoostingEstimator(distribution='multinomial',
                                    ntrees=100,
                                    max_depth=4,
                                    learn_rate=0.1)
model_gbm.train(x=features, y="response", training_frame=train_split, validation_frame=valid_split)
print(model_gbm)
pred = model_gbm.predict(test)
pred.head()
submit_pred_licence= pred['predict']
submit_pred_licence.head()
#submission_dataframe =(test[:,'member_id']).cbind(submit_pred)
submit_pred_licence.set_name(0,"tags")
h2o.h2o.export_file(submit_pred_licence, path ="bs_ms.csv")






X_train_h2o=h2o.H2OFrame(x_train.values.tolist())
Y_train_h2o=h2o.H2OFrame(y_exp,column_names=['response'],column_types=['categorical'])
train=X_train_h2o.cbind(Y_train_h2o)
test=h2o.H2OFrame(x_test.values.tolist())
r = train.runif()
train_split = train[r  < 0.8]
valid_split = train[r >= 0.8]
features = list(np.setdiff1d(train.names, ['response']))

from h2o.estimators.gbm import H2OGradientBoostingEstimator
model_gbm = H2OGradientBoostingEstimator(distribution='multinomial',
                                    ntrees=100,
                                    max_depth=4,
                                    learn_rate=0.1)
model_gbm.train(x=features, y="response", training_frame=train_split, validation_frame=valid_split)
print(model_gbm)
pred = model_gbm.predict(test)
pred.head()
submit_pred_exp= pred['predict']
submit_pred_exp.head()
#submission_dataframe =(test[:,'member_id']).cbind(submit_pred)
submit_pred_exp.set_name(0,"tags")
h2o.h2o.export_file(submit_pred_exp, path ="exp.csv")




X_train_h2o=h2o.H2OFrame(x_train.values.tolist())
Y_train_h2o=h2o.H2OFrame(y_supervising,column_names=['response'],column_types=['categorical'])
train=X_train_h2o.cbind(Y_train_h2o)
test=h2o.H2OFrame(x_test.values.tolist())
r = train.runif()
train_split = train[r  < 0.8]
valid_split = train[r >= 0.8]
features = list(np.setdiff1d(train.names, ['response']))

from h2o.estimators.gbm import H2OGradientBoostingEstimator
model_gbm = H2OGradientBoostingEstimator(distribution='multinomial',
                                    ntrees=100,
                                    max_depth=4,
                                    learn_rate=0.1)
model_gbm.train(x=features, y="response", training_frame=train_split, validation_frame=valid_split)
print(model_gbm)
pred = model_gbm.predict(test)
pred.head()
submit_pred_super= pred['predict']
submit_pred_super.head()
#submission_dataframe =(test[:,'member_id']).cbind(submit_pred)
submit_pred_super.set_name(0,"tags")
h2o.h2o.export_file(submit_pred_super, path ="supervising.csv")











###Other Approach####
train=pd.concat([tfidf, y_train], axis=1)

train.to_csv("train_file_h2o.csv",index=False)
features = list(set(train.columns)-set(feature_list))

responses_dec=[]
for i in list(y_train.index):
	#ro=list(y_train.iloc[i,:])
	responses_dec.append(int(''.join(str(x) for x in list(y_train.iloc[i,:])),2)-1)

lbl = LabelEncoder()
lbl.fit(responses_dec)
responses_dec_encoded = lbl.transform(responses_dec)



dtrain = xgb.DMatrix(train[features], responses_dec_encoded, missing=np.nan)
dtest = xgb.DMatrix(tfidf_test, missing=np.nan)

params = {"objective": "multi:softmax","booster": "gbtree", "nthread": 4, "silent": 1,
                "eta": 0.08, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,
                "min_child_weight": 1, "num_class": max(responses_dec_encoded)+1,
                "seed": 2016, "tree_method": "exact"}

nrounds = 260
watchlist = [(dtrain, 'train')]
bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)
test_preds = bst.predict(dtest)

submit_encoded = pd.DataFrame({'tagsencodes': test_preds})
submit_encoded.to_csv("submit_encoded.csv",index=False)

submit = pd.DataFrame({'Trip_ID': test['Trip_ID'], 'Surge_Pricing_Type': test_preds})
submit.to_csv("XGB.csv", index=False)

submit_encoded=pd.read_csv("/home/sohom/Desktop/HackerRank_Indeed_ML/submit_encoded.csv")

sub_en1=lbl.inverse_transform(test_preds)#in decimal
sub_en2= ["{0:b}".format(i) for i in sub_en1] # Need to change


#*#12 models creatre
#*#directly predict -> (000100)

