from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from modules.BNClassifier import BNClassifier

clfs_dict = {
    'knn': KNeighborsClassifier(weights='distance', algorithm='brute'),
    'svm': SVC(probability=True, kernel='linear'),
    'rf': RandomForestClassifier(class_weight='balanced'),
    'xgb': XGBClassifier(enable_categorical=True, scale_pos_weight=4.7),
    # scale_pos_weight is the weight of positive class
    'bn': BNClassifier()
}
