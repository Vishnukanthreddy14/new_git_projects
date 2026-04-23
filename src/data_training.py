from sklearn.linear_model import LogisticRegression
import joblib

def train_model(X,y):
	model=LogisticRegression(max_iter=1000)
	model.fit(X,y)

	joblib.dump(model,"model.pkl")
	return model
