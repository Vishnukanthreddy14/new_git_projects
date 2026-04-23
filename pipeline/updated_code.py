rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X, y)

# XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X, y)
