import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# 1. Loading the dataset
housing_df = pd.read_csv(r"ML/housing.csv")

# 2. splitting the train and test set
housing_df["income_cat"] = pd.cut(housing_df["median_income"], 
                                  bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                  labels=[1, 2, 3, 4, 5])


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for tr_index, tst_index in sss.split(housing_df, housing_df["income_cat"]):
    s_train_set = housing_df.loc[tr_index].drop("income_cat", axis=1)
    s_test_set = housing_df.loc[tst_index].drop("income_cat", axis=1)

# 3. making copy of our train set
housing_df = s_train_set.copy()

# 4. separating labels and features
df_labels = housing_df["median_house_value"].copy()
housing_df = housing_df.drop("median_house_value", axis=1)

# 5. separating numerical and categorical columns
num_cols = housing_df.drop("ocean_proximity", axis=1).columns.tolist()
cat_cols = ["ocean_proximity"]

# 6. creating pipeline for numerical data
num_pipeline = Pipeline([
                        ("impute", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()),
                         ])


# 7. creating pipeline for categorical data
cat_pipeline = Pipeline([
    ("oneHot_encoding", OneHotEncoder(handle_unknown="ignore"))
])

# 8. making full pipeline
full_pipeline = ColumnTransformer([
    ("nums", num_pipeline, num_cols),
    ("categoric", cat_pipeline, cat_cols),])

# 9. Transforming the data
prepared_df = full_pipeline.fit_transform(housing_df)

# # 10. checking the accuracy of every model

# linear regressor model
lin_reg = LinearRegression()
lin_reg.fit(prepared_df, df_labels)
lin_preds = lin_reg.predict(prepared_df)
# lin_rmse = root_mean_squared_error(df_labels,lin_preds)
lin_rmses = -cross_val_score(lin_reg, prepared_df, df_labels, scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(lin_rmses).describe())
# print("the rmse of linear regressor is", lin_rmse)
print()

# decisiontree regressor model
dec_reg = DecisionTreeRegressor()
dec_reg.fit(prepared_df, df_labels)
dec_preds = dec_reg.predict(prepared_df)
# dec_rmse = root_mean_squared_error(df_labels,dec_preds)
dec_rmses = -cross_val_score(dec_reg, prepared_df, df_labels, scoring="neg_root_mean_squared_error", cv=10)
# print("the rmse of Decision tree regressor is", dec_rmse)
print(pd.Series(dec_rmses).describe())

  
# random tree regressor model
ran_reg = RandomForestRegressor()
ran_reg.fit(prepared_df, df_labels)
ran_preds = ran_reg.predict(prepared_df)
# ran_rmse = root_mean_squared_error(df_labels,ran_preds)
# print("the rmse of randomtree regressor is", ran_rmse)
ran_rmses = -cross_val_score(ran_reg, prepared_df, df_labels, cv=10, scoring="neg_root_mean_squared_error")
print(pd.Series(ran_rmses).describe())
