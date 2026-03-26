import mlflow
import mlflow.sklearn
import pandas as pd 
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("one")
df = pd.read_csv("data.csv")
df = df.drop(columns=["id","Unnamed: 32"])
y = df["diagnosis"]
x =df.drop(columns = ["diagnosis"])
lr = LabelEncoder()
y = lr.fit_transform(y)
xt,xte,yt,yte = train_test_split(x,y,random_state=42,test_size=0.2)
n_estimator = 10
max_depth = 5
with mlflow.start_run():
    rfc = RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimator,random_state=42)
    rfc.fit(xt,yt)
    yp = rfc.predict(xte)
    accuracy = accuracy_score(yte,yp)
    cm = confusion_matrix(yte,yp)

    mlflow.log_param("n_estimator",n_estimator)
    mlflow.log_param("maxdepth",max_depth)
    mlflow.log_metric("accuracy",accuracy)
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.set_tags({"ceator":"anmol","editor":"dubey"})
    mlflow.sklearn.log_model(rfc)
    mlflow.log_artifact(__file__)

    print(accuracy,cm)

