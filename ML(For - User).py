import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

MODEL_FILE = "model.pkl"   #model baniba au file re rakhideba 
PIPELINE_FILE = "pipeline.pkl"  #jou future ku jou data asiba suppose your hr give some again extra data about housing ,can you again perform handling (null values(simpleimutr,),stndardscelaer)
#pehere tu au thare goi goti kari operation karibuna kn sehti paine purba ru jau opreation karithili data re taku store kari deba ae file re ,au file ku use kariki sabu operation lagaideba sange sange
 #these two file make s by joblib

def build_pipeline(num_attribs,cat_attribs):
    #For numerical columns
    num_pipline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # For categorical columns
    cat_pipline = Pipeline([ 
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Construct the full pipeline
    full_pipeline = ColumnTransformer([
        ("num", num_pipline, num_attribs), 
        ('cat', cat_pipline, cat_attribs)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):    #jadi model file exist karunathiba ama ku train karibaku heba 
    #lets train the model(load kara ,train set,test bhara kara ,jou operation karithila saisabu kari pariba etcc.)
    # 1. Load the dataset
    housing = pd.read_csv("housing.csv")

    # 2. Create a stratified test set
    housing['income_cat'] = pd.cut(housing["median_income"], 
                                bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], 
                                labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing['income_cat']):
        housing.loc[test_index].drop("income_cat",axis=1).to_csv("input.csv",index=False)  #savingt the test_index into csv named as input.csv file
        housing = housing.loc[train_index].drop("income_cat", axis=1) # We will work on this data # here we directly take train data

        # 3. Seperate features and labels from this train data 
    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1)

    # 4. List the numerical and categorical columns
    num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    #5.BUild pipeline 
    pipeline = build_pipeline(num_attribs,cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)
    # print(housing_prepared)

    #let's train the model and also fit
    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared,housing_labels)

    #lets use joblib, jauta model ku MODEL_FILE re au pipeline ku PIPELINE_FILE re  dump kariba (dump mean-sehi file rahiba)
    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)        #after run -> it store model and pipeline and give two pipeline
    print("Model is trained. Congrats!")
else:   
    #jadi mo model file exists kala,mu inference karbi hela
    #lets do inference    # for incoming data
    model = joblib.load(MODEL_FILE)  #loading model file
    pipeline = joblib.load(PIPELINE_FILE)  #loading pipeline file

    input_data = pd.read_csv('input.csv')   #input data ku read kale
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data['median_house_value'] = predictions

    input_data.to_csv("output.csv",index=False) #input data ku output csv ku push kale au index ku kati dele
    print("Inference is complete, results saved to output.csv Enjoy!")








