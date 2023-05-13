from plot_func import plot_confusion_matrix, plot_ROC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
import numpy as np

# --------------------- CONSTANTS --------------------- #
class Parameters:
    SVM_KERNEL = 'rbf'
    LR_WEIGHT_CLASS = 'balanced'
    MAX_ITER = 10000
    N_NEIGHBORS = 5
    N_DECISION_TREES = 5
    N_ADABOOST_ESTIMATORS = 5
    N_XGBOOST_ESTIMATORS = 5

class ModelNames:
    SVM_MODEL = 'Support Vector Machine'
    LOG_REG = 'Logistic Regression'
    DECISION_TREE = 'Decision Tree'
    GAUSS_NB = 'Gauss Naive Bayes'
    KNN = 'K Nearest Neighbor'
    RNDFR = 'Random Forest'
    ADABOOST = 'ADABoost'
    XGBOOST = 'XGBoost'

class FeatSelectorNames:
    RFE = 'Recursive Feature Elimination'
    SKB = 'Select K Best'
    PCA = 'Principal Component Analysis'

# --------------------- FUNCTIONS --------------------- #
def perform_cross_validation(clf, x_train, y_train, splits, scorers={'roc_auc' : 'roc_auc'}):
    cv = KFold(n_splits=splits, random_state=1, shuffle=True)
    scores = cross_validate(clf, x_train, y_train, scoring=scorers, cv=cv, n_jobs=-1)
    for key, value in scorers.items():
        print(str(key) + ' on train set using 5-Fold-CV: %.3f (%.3f)' % (np.mean(scores['test_' + key]), np.std(scores['test_' + key])))
       
def make_plots(y_test, y_pred, score_test, classname):
    print('Plotting curves and tables...\n')
    plot_confusion_matrix(y_test, y_pred, cn=classname, classes=["class LOW", "class HIGH"], normalize=True)
    print('Plotting ROC...\n')
    plot_ROC(y_test, score_test, cn=classname)