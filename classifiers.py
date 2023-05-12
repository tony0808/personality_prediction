from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn import preprocessing
from sklearn import metrics
import numpy as np
from featureSelection import FeatureSelector
from utils import *

class Classifier:
    
    def __init__(self, data, labels, trait, splits = 5, scorers = {'roc_auc' : 'roc_auc'}, feature_selector = None):
        self.data = data
        self.labels = labels
        self.selector = feature_selector
        self.trait = trait
        self.splits = splits
        self.scorers = scorers
        self.model_dict = {
            ModelNames.SVM_MODEL : self.svmModel,
            ModelNames.LOG_REG : self.logRegression,
            ModelNames.GAUSS_NB : self.gaussianNB,
            ModelNames.DECISION_TREE : self.decisionTreeModel,
            ModelNames.KNN : self.knnModel,
            ModelNames.RNDFR : self.randomForestModel,
            ModelNames.ADABOOST : self.adaboostModel,
            ModelNames.XGBOOST : self.xgboostModel
        }
    
    def train_model(self, model_name):
        return self.model_dict[model_name]()

    def xgboostModel(self):
        # split data
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.2, random_state=42)
        
        # create pipeline
        xgb_clf = xgb.XGBClassifier(n_estimators=Parameters.N_XGBOOST_ESTIMATORS, random_state=42)
        pipeline = Pipeline([('scaler', preprocessing.StandardScaler()), 
                             ('feature_selector', self.selector), 
                             ('xgb_clf', xgb_clf)])
        
        perform_cross_validation(pipeline, x_train, y_train, self.splits, self.scorers)

        # make predictions
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        
        # make predictions
        y_pred_prob = pipeline.predict_proba(x_test)
        classification_metrics = metrics.classification_report(y_test, y_pred, digits=4)
        make_plots(y_test, y_pred, y_pred_prob[:,1], self.trait)
        print(classification_metrics)
        return accuracy, classification_metrics

    def adaboostModel(self):
        # split data 
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.2, random_state=42)

        # create pipeline
        ada = AdaBoostClassifier(n_estimators=Parameters.N_ADABOOST_ESTIMATORS, random_state=42)
        pipeline = Pipeline([('scaler', preprocessing.StandardScaler()), 
                             ('feature_selector', self.selector), 
                             ('ada', ada)])
        
        perform_cross_validation(pipeline, x_train, y_train, self.splits, self.scorers)

        # make predictions
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        # report results
        accuracy = metrics.accuracy_score(y_test, y_pred)
        y_pred_prob = pipeline.predict_proba(x_test)
        classification_metrics = metrics.classification_report(y_test, y_pred, digits=4)
        make_plots(y_test, y_pred, y_pred_prob[:,1], self.trait)
        print(classification_metrics)
        return accuracy, classification_metrics

    def randomForestModel(self):
        # split data
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.2, random_state=42)

        # create pipeline
        rf = RandomForestClassifier(n_estimators=Parameters.N_DECISION_TREES, random_state=42)
        pipeline = Pipeline([('scaler', preprocessing.StandardScaler()), 
                             ('feature_selector', self.selector), 
                             ('rf', rf)])
        
        perform_cross_validation(pipeline, x_train, y_train, self.splits, self.scorers)

        # make predictions
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        # report results
        accuracy = metrics.accuracy_score(y_test, y_pred)
        y_pred_prob = pipeline.predict_proba(x_test)
        classification_metrics = metrics.classification_report(y_test, y_pred, digits=4)
        make_plots(y_test, y_pred, y_pred_prob[:,1], self.trait)
        print(classification_metrics)
        return accuracy, classification_metrics

    def knnModel(self):
        # split data
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.2, random_state=42)

        # create pipeline
        knn = KNeighborsClassifier(n_neighbors=Parameters.N_NEIGHBORS)
        pipeline = Pipeline([('scaler', preprocessing.StandardScaler()), 
                             ('feature_selector', self.selector), 
                             ('knn', knn)])
        
        perform_cross_validation(pipeline, x_train, y_train, self.splits, self.scorers)

        # make predictions
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        # report results
        accuracy = metrics.accuracy_score(y_test, y_pred)
        y_pred_prob = pipeline.predict_proba(x_test)
        classification_metrics = metrics.classification_report(y_test, y_pred, digits=4)
        make_plots(y_test, y_pred, y_pred_prob[:,1], self.trait)
        print(classification_metrics)
        return accuracy, classification_metrics

    def decisionTreeModel(self):
        # split data
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.2, random_state=42)

        # create pipeline
        dt = DecisionTreeClassifier(random_state=42)
        pipeline = Pipeline([('scaler', preprocessing.StandardScaler()), 
                             ('select', self.selector), 
                             ('dt', dt)])
        
        perform_cross_validation(pipeline, x_train, y_train, self.splits, self.scorers)

        # make predictions
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        # report results
        accuracy = metrics.accuracy_score(y_test, y_pred)
        y_pred_prob = pipeline.predict_proba(x_test)
        classification_metrics = metrics.classification_report(y_test, y_pred, digits=4)
        make_plots(y_test, y_pred, y_pred_prob[:,1], self.trait)
        print(classification_metrics)
        return accuracy, classification_metrics

    def gaussianNB(self):
        # split data
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.2, random_state=42)

        # create pipeline
        gnb = GaussianNB()
        pipeline = Pipeline([('scaler', preprocessing.StandardScaler()), 
                             ('select', self.selector), 
                             ('gnb', gnb)])
        
        perform_cross_validation(pipeline, x_train, y_train, self.splits, self.scorers)

        # make predictions
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        # report results
        accuracy = metrics.accuracy_score(y_test, y_pred)
        y_pred_prob = pipeline.predict_proba(x_test)
        classification_metrics = metrics.classification_report(y_test, y_pred, digits=4)
        make_plots(y_test, y_pred, y_pred_prob[:,1], self.trait)
        print(classification_metrics)
        return accuracy, classification_metrics

    def logRegression(self):
        # split data
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.2, random_state=42)

        # create pipeline
        lr = LogisticRegression(class_weight=Parameters.LR_WEIGHT_CLASS, max_iter=Parameters.MAX_ITER)
        pipeline = Pipeline([('scaler', preprocessing.StandardScaler()),
                             ('select', self.selector), 
                             ('lr', lr)])

        # optimize hyperparameters
        parameters = {'lr__C':np.logspace(-4,4,50)}
        clf = GridSearchCV(pipeline, parameters, n_jobs=4, cv=5, verbose=0, scoring='balanced_accuracy', refit=True)
        perform_cross_validation(pipeline, x_train, y_train, self.splits, self.scorers)

        # make predictions
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        y_pred_prob = clf.predict_proba(x_test)

        # report results
        accuracy = metrics.accuracy_score(y_test, y_pred)
        classification_metrics = metrics.classification_report(y_test, y_pred, digits=4)
        make_plots(y_test, y_pred, y_pred_prob[:,1], self.trait)
        print(classification_metrics)
        return accuracy, classification_metrics

    def svmModel(self):
        # split the data
        x_train, x_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.2, random_state=42)

        # create a pipeline
        svmModel = svm.SVC(kernel= Parameters.SVM_KERNEL, class_weight='balanced')
        pipeline = Pipeline([('scaler', preprocessing.StandardScaler()), 
                             ('select', self.selector), 
                             ('svm', svmModel)])

        # optimize hyperparameters
        parameters = {'svm__C':[0.001, 0.01, 0.1, 1, 2, 3, 5], 'svm__gamma':[0.0001, 0.001, 0.01, 0.1, 1]}
        clf = GridSearchCV(pipeline, parameters, n_jobs=4, cv=5, verbose=0, scoring='balanced_accuracy', refit=True)
        perform_cross_validation(pipeline, x_train, y_train, self.splits, self.scorers)

        # make predictions
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        # report results
        accuracy = metrics.accuracy_score(y_test, y_pred)
        score_test = clf.decision_function(x_test)
        classification_metrics = metrics.classification_report(y_test, y_pred,digits=4)
        make_plots(y_test, y_pred, score_test, self.trait)
        print(classification_metrics)
        return accuracy, classification_metrics


        