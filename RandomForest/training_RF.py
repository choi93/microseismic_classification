import numpy as np
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score
import joblib

#file name of the input data
train_data=np.loadtxt('x_train.dat',delimiter=',',dtype=np.float32)
train_label=np.loadtxt('y_train.dat',dtype=np.float32)
test_data=np.loadtxt('x_test.dat',delimiter=',',dtype=np.float32)
test_label=np.loadtxt('y_test.dat',dtype=np.float32)

#file name of the ouput model
fmodel='rf_model.sav'

# fit the Random Forest model
model=rfc(n_estimators=2000, max_depth=18,max_leaf_nodes=250, class_weight=[{1:1,2:1,3:1,4:1,5:2}], oob_score=True, n_jobs=-1, random_state=42)
model.fit(train_data_prepared,train_lable)

#save the model to disk
joblib.dump(model,fmodel)

print(grid_search.best_params_)

print('================================================')
print('=================Model trainig==================')
print('Training score(oob score): ', model.oob_score_)
print('================================================')
y_train_pred=model.predict(train_data_prepared)
print('prediction_accuracy: ',accuracy_score(y_train_pred,train_label))
print(confusion_matrix(train_label,y_train_pred))

#predict using test data set
pred_result=model.predict(test_data_prepared)

#caculate accuracy of prediction
print('==================Model test====================')
print('test_accuracy: ',accuracy_score(pred_result,test_label))
print(confusion_matrix(test_label,pred_result))

