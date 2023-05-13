from dataAccess import DataAccessManager
from featureSelection import *
import xgboost as xgb
from classifiers import Classifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from utils import *
from featureSelection import FeatureSelector

# obtain the data
dtAccess = DataAccessManager()
data = dtAccess.get_data('o')
labels = data['class']
data = data.drop('class', axis=1)

# this column has zero variance and is causing problem to feature selection methods
data = data.drop('SF_min', axis=1)

# select scorer, classifier, feature selection method
scorers = {'BA': 'balanced_accuracy', 'prec': 'precision'}
model_name = ModelNames.GAUSS_NB
feat_selector_name = FeatSelectorNames.PCA
featSelector = FeatureSelector(feat_selector_name, model_name, 20)
selector = featSelector.get_feature_selector()

# train model
classifier = Classifier(data, labels, 'openess', scorers=scorers, feature_selector=selector)
result = classifier.train_model(model_name)