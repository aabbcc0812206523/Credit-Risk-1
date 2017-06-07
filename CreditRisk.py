import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pydotplus
from sklearn.metrics import roc_curve, auc

####################################################################################

# Reading the CSV File

df=pd.read_csv('C:/Users/IBM_ADMIN/Downloads/Credit Risk/Credit Risk.csv')
df.drop('score', axis=1, inplace=True)

#####################################################################################

# Hot Encoding - Numarical Labelizing of the categories

lb = LabelEncoder()
df['applied_online'] = lb.fit_transform(df['applied_online'])
df['car_loan_holder'] = lb.fit_transform(df['car_loan_holder'])
df['higher_education'] = lb.fit_transform(df['higher_education'])
df['home_loan_holder'] = lb.fit_transform(df['home_loan_holder'])

#####################################################################################

# Random Partitioning into 80%-20% Train & Test Sets

train, test = train_test_split(df, test_size = 0.2, random_state=33)

#####################################################################################

# Predictive Classification Modeling using Decision Tree - CART (Gini)

# Selecting the independent and dependent variables

X_Train=train[train.columns[:-1]]
Y_Train=train[train.columns[-1]]
X_Test=test[test.columns[:-1]]
Y_Test=test[test.columns[-1]]

# Fitting the predictive model on Training data

clf_dt = tree.DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_leaf=5)
clf_dt.fit(X_Train, Y_Train)

# Prediction on the Test data

Y_Pred = clf_dt.predict(X_Test)

#####################################################################################

# Tried other algorithms, but bad performance

#from sklearn.neural_network import MLPClassifier
#mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
#mlp.fit(X_Train, Y_Train)
#Y_PredNN = mlp.predict(X_Test)

#from sklearn.svm import SVC
#clfsvm = SVC()
#clfsvm.fit(X_Train, Y_Train)
#Y_PredSVM = clfsvm.predict(X_Test)

#####################################################################################

# Accuracy Scores

print("Accuracy on training set:")
print (clf_dt.score(X_Train, Y_Train))

print("Accuracy on test set:")
print (clf_dt.score(X_Test, Y_Test))

#####################################################################################

# Confusion Matrix Chart

class_names = ['0','1']

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_Test, Y_Pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')

plt.show()

#############################################################################

# Plotting decision Tree

dot_data = tree.export_graphviz(clf_dt.fit(X_Train, Y_Train), out_file=None, feature_names=train.columns[:-1], class_names=['0','1'], filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("CreditRiskTree.pdf")

########################################################################################

# Metrices

roc_auc_score(Y_Test, Y_Pred)

precision_score(Y_Test, Y_Pred, average='binary')
recall_score(Y_Test, Y_Pred, average='binary')
clf_dt.feature_importances_

#######################################################################################

# ROC AUC

fpr, tpr, _ = roc_curve(Y_Test, clf_dt.predict_proba(X_Test)[:,1])

roc_auc = auc(fpr, tpr)

print('ROC AUC: %0.2f' % roc_auc)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

test['PredictedY'] = Y_Pred

train.to_csv('C:/Users/IBM_ADMIN/Downloads/Credit Risk/CreditRiskTrain.csv')
test.to_csv('C:/Users/IBM_ADMIN/Downloads/Credit Risk/CreditRiskTest.csv')
