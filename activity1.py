# Import libraries

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
import time

start_time = time.time()
# random forest generation returning the mean absolute error
from sklearn.ensemble import RandomForestRegressor

def get_random_forest_mae(X_trn,X_tst,y_trn,y_tst):
  mdl = RandomForestRegressor(random_state=1)
  mdl.fit(X_trn,y_trn)
  y_tst_prd = mdl.predict(X_tst)
  mae = mean_absolute_error(y_tst,y_tst_prd)
  return(mae)


df = pd.read_csv("../austin_311.csv")

# Quick visualization of the dataset
# import seaborn as sns
# sns.pairplot(df)
# sns.imshow()


"""
Finding the mean absolute error using random forest model

"""



# define our variables, we want to determine the status so that is our y, all other factors are treated as independent to that event
y = df.loc[:,["SR Status"]]
x = df.drop(["SR Status","SR Location","Created Date", "Closed Date", "(Latitude.Longitude)", "Street Number", "Service Request (SR) Number"], axis = 1)

# selecting numeric features of the dataset
num_columns = [c for c in x.columns if x[c].dtype in ['int64', 'float64']]
x_numeric = x[num_columns]

# using x_numeric.isna() I see there are NaN values in the numeric columns, so I will replace with the mean
x_numeric = x_numeric.fillna(x_numeric.mean())

# Transforming the categorical features
cat_obj = [c for c in x.columns if x[c].dtype == 'object' and x[c].nunique() < 144]
x_catergoric = x[cat_obj]
x_catergoric.fillna("hamza", inplace=True)

le = preprocessing.LabelEncoder()

for col in cat_obj:
    x_catergoric[col] = le.fit_transform(x_catergoric[col])

# Putting it all back together
frames = [x_numeric, x_catergoric]
x_test = pd.concat(frames, axis =1 ,ignore_index=True)

# Setting up the y
y_columns = [c for c in y.columns]

for c in y_columns:
    y[c] = le.fit_transform(y[c])

# Running the Random Forest Model
x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(x_test, y, test_size=0.2, random_state=1)
mae = get_random_forest_mae(x_train_t, x_test_t, y_train_t, y_test_t)
print(f"The MAE from the Random Forest: {mae}\n")



"""
Finding the MAE from the gradient boosted tree

"""


from xgboost import XGBRegressor

mdlXgb = XGBRegressor(n_estimators = 5000, learning_rate = 0.01, max_depth = 5)
mdlXgb.fit(x_train_t, y_train_t)
y_test_pred = mdlXgb.predict(x_test_t)
mae = mean_absolute_error(y_test_t, y_test_pred)
print(f"The MAE from the Gradient Boosted Tree is : {mae}\n")



"""
Finding the Mean Resolution from the Dates

"""

date1 = df.loc[:,["Created Date"]]
date2 = df.loc[:,["Closed Date"]]

# Function to make the dates into a correctly formatted list of dates
def date_list_generator(data):
    app = []
    len(data)
    for i in data:
        for j in data[i]:
            app.append(str(j).replace("-","/")[2:])
    return app

# Creating the lists
date_1_list = date_list_generator(date1)
date_2_list = date_list_generator(date2)


# Making arrays of date object 1
from datetime import datetime

date1_obj = []
for i in range(len(date_1_list)):
    datetime_object_1 = datetime.strptime(date_1_list[i], '%y/%m/%d %H:%M:%S')
    date1_obj.append(datetime_object_1)

# Making array of date object 2, but there were formatting issues so I back filled
date2_obj = []
for i in range(len(date_2_list)):
    if date_2_list[i] == "n": 
        date_2_list[i] = date_2_list[i-1]
    datetime_object_2 = datetime.strptime(date_2_list[i], '%y/%m/%d %H:%M:%S')
    date2_obj.append(datetime_object_2)

# Making the array of date differences between open and closed
date_difference = []
for i in range(len(date_2_list)):
    date_difference.append(date2_obj[i] - date1_obj[i])

average_timedeltavg = pd.to_timedelta(pd.Series(date_difference)).mean()

print(f"Mean time to resolution: {average_timedeltavg}\n")
print(f"Time in seconds since the start:  {time.time() - start_time} seconds")