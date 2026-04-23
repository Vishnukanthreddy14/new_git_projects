from src.data_preprocessing import load_data,preprocess
from src.data_training import train_model
from src.evaluate import evaluate

def run_pipeline():
	df=load_data("C:\\Users\\Vishnus\\MLOPs Experiments VSCODE\\MLOPs EXP 4\\data\\Telecom Customer Churn.csv")
	X,y=preprocess(df)
	model=train_model(X,y)
	score=evaluate(model,X,y)
	X,y=preprocess(df)
	rf_model=train_model(X,y)
	rf_score=evaluate(rf_model,X,y)
	X,y=preprocess(df)
	xg_model=train_model(X,y)
	xg_score=evaluate(xg_model,X,y)
	print("Model Accuracy score",score)
	print("Random Forest Accuracy score",rf_score)
	print("XGBoost Accuracy score",xg_score)
if __name__=="__main__":
	run_pipeline()

