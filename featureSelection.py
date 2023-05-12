from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from utils import *

class FeatureSelector:
    def __init__(self, selector_name = None, model_name = None, n_features = 5):
        self.n_features = n_features
        self.model_name = model_name
        self.selector_name = selector_name
        self.selector = None
        self.selector_dict = {
            FeatSelectorNames.RFE : self.recursiveFeatureSelector,
            FeatSelectorNames.SKB : self.selectKbestFeatureSelector,
            FeatSelectorNames.PCA : self.pcaFeatureSelector
        }

    def get_feature_selector(self):
        self.selector_dict[self.selector_name]()
        return self.selector

    def pcaFeatureSelector(self):
        self.selector = PCA(n_components=self.n_features)
    
    def selectKbestFeatureSelector(self):
        self.selector = SelectKBest(score_func=f_regression, k=self.n_features)

    def recursiveFeatureSelector(self):
        selector_dict = {
            ModelNames.LOG_REG: LogisticRegression(class_weight=Parameters.LR_WEIGHT_CLASS, max_iter=Parameters.MAX_ITER),
            ModelNames.DECISION_TREE: DecisionTreeClassifier()
        }

        self.selector = selector_dict.get(self.model_name)
        if self.selector is None:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        self.selector = RFE(estimator=self.selector, n_features_to_select=self.n_features)



