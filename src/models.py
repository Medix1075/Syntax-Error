import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def fit_evaluate(df, graph_enhanced=False, seed=42):
    base=["distance_km","planned_minutes","route_type","dispatch_hour"]
    graph=[c for c in df.columns if c.startswith("origin_") or c.startswith("destination_")]
    features=base+(graph if graph_enhanced else [])
    cat=["route_type"]; num=[c for c in features if c not in cat]
    pre=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat),("num","passthrough",num)])
    pipe=Pipeline([("pre",pre),("model",RandomForestRegressor(n_estimators=250,min_samples_leaf=3,random_state=seed,n_jobs=-1))])
    cut=int(len(df)*.8); train,test=df.iloc[:cut],df.iloc[cut:]
    pipe.fit(train[features],train.actual_minutes); pred=pipe.predict(test[features])
    mae=mean_absolute_error(test.actual_minutes,pred); within=float(np.mean(np.abs(pred-test.actual_minutes)/test.actual_minutes<=.15)*100)
    return pipe,{"mae":float(mae),"within_15pct":within,"n_test":len(test)},pred,test
