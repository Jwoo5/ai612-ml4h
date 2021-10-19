import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

# train_x = np.load('./X_train_lr_norm.npy')
train_x = np.load('./20218078/X_train_logistic.npy')
train_y = np.load('./20218078/y_train.npy')

# test_x = np.load('./X_test_lr_norm.npy')
test_x = np.load('./20218078/X_test_logistic.npy')
test_y = np.load('./20218078/y_test.npy')

model = LogisticRegression(class_weight= 'balanced', max_iter = 10000, solver = 'liblinear')
model.fit(train_x, train_y)

print(roc_auc_score(train_y, model.decision_function(train_x)))
print(average_precision_score(train_y, model.decision_function(train_x)))
print(roc_auc_score(test_y, model.decision_function(test_x)))
print(average_precision_score(test_y, model.decision_function(test_x)))

pickle.dump(model, open('./20218078/lr_trained.pkl', 'wb'))