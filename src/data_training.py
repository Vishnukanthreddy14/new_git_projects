from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import joblib

def train_model(X,y):
	model=LogisticRegression(max_iter=1000)
	model.fit(X,y)
	
	rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
	rf_model.fit(X, y)

# XGBoost
	xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
	xgb_model.fit(X, y)

	joblib.dump(model,"model.pkl")
	return model
