import mlflow
import mlflow.sklearn
import pandas as pd 
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("third")
mlflow.sklearn.autolog()

df = pd.read_csv("data.csv")
df = df.drop(columns=["id","Unnamed: 32"])
y = df["diagnosis"]
x =df.drop(columns = ["diagnosis"])
lr = LabelEncoder()
y = lr.fit_transform(y)
xt,xte,yt,yte = train_test_split(x,y,random_state=42,test_size=0.2)
with mlflow.start_run() as parent:
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {"n_estimators":[10,50,100],"max_depth":[10,20,30]}
    grid_search = GridSearchCV(estimator=rfc,param_grid=param_grid,cv=5,n_jobs=-1,verbose=2)
    grid_search.fit(xt,yt)
    for i in range(len(grid_search.cv_results_["params"])):
        with mlflow.start_run(nested=True) as child:
            mlflow.set_tags({"no":i,"abcd": 1*i})
    mlflow.log_artifact(__file__)





        
