from sklearn.feature_selection import SequentialFeatureSelector as SFS
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
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
            ModelNames.DECISION_TREE: DecisionTreeClassifier(),
            ModelNames.KNN: KNeighborsClassifier(n_neighbors=Parameters.N_NEIGHBORS),
            ModelNames.RNDFR: RandomForestClassifier(n_estimators=Parameters.N_DECISION_TREES, random_state=42),
            ModelNames.ADABOOST: AdaBoostClassifier(n_estimators=Parameters.N_ADABOOST_ESTIMATORS, random_state=42),
            ModelNames.XGBOOST: xgb.XGBClassifier(n_estimators=Parameters.N_XGBOOST_ESTIMATORS, random_state=42)
        }

        self.selector = selector_dict.get(self.model_name)
        if self.selector is None:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        self.selector = RFE(estimator=self.selector, n_features_to_select=self.n_features)



