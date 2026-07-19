import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
import data_visualization as viz


df = pd.read_csv(r'C:\Users\rames\OneDrive\Desktop\Earthquake\database.csv')
df.head()

df.drop_duplicates(inplace=True)
df.fillna(
    df.median(numeric_only=True),
    inplace=True
)



df['Date'] = pd.to_datetime(df['Date'], errors='coerce')



df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Hour'] = df['Date'].dt.hour
df['Minute'] = df['Date'].dt.minute


df.drop('Date', axis=1, inplace=True)

print(df[['Year', 'Month', 'Day', 'Hour', 'Minute']].head())
df['Risk'] = np.where(df['Magnitude'] >= 7, 1, 0)


X = df[[
'Latitude',
'Longitude',

'Year',
'Month',
'Day',
'Hour',
'Minute'
]]

y = df['Risk']
print(y.unique())
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print(y_train.value_counts())
print(y_test.value_counts())


rf = RandomForestClassifier()

xgb = XGBClassifier()

lgbm = LGBMClassifier()

rf.fit(X_train,y_train)

xgb.fit(X_train,y_train)

lgbm.fit(X_train,y_train)

# Random Forest
rf_pred = rf.predict(X_test)

print("\nRandom Forest Accuracy:")
print(accuracy_score(y_test, rf_pred))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))


# XGBoost
xgb_pred = xgb.predict(X_test)

print("\nXGBoost Accuracy:")
print(accuracy_score(y_test, xgb_pred))

print("\nXGBoost Confusion Matrix:")
print(confusion_matrix(y_test, xgb_pred))

print("\nXGBoost Classification Report:")
print(classification_report(y_test, xgb_pred))


# LightGBM
lgbm_pred = lgbm.predict(X_test)

print("\nLightGBM Accuracy:")
print(accuracy_score(y_test, lgbm_pred))

print("\nLightGBM Confusion Matrix:")
print(confusion_matrix(y_test, lgbm_pred))

print("\nLightGBM Classification Report:")
print(classification_report(y_test, lgbm_pred))


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create K-Fold object
kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Random Forest
rf_scores = cross_val_score(
    rf,
    X,
    y,
    cv=kfold,
    scoring='accuracy'
)

# XGBoost
xgb_scores = cross_val_score(
    xgb,
    X,
    y,
    cv=kfold,
    scoring='accuracy'
)

# LightGBM
lgbm_scores = cross_val_score(
    lgbm,
    X,
    y,
    cv=kfold,
    scoring='accuracy'
)

# Print results
print("Random Forest Scores:", rf_scores)
print("Random Forest Mean Accuracy:", rf_scores.mean())

print("\nXGBoost Scores:", xgb_scores)
print("XGBoost Mean Accuracy:", xgb_scores.mean())

print("\nLightGBM Scores:", lgbm_scores)
print("LightGBM Mean Accuracy:", lgbm_scores.mean())




rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

rf_grid = GridSearchCV(
    estimator=rf,
    param_grid=rf_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

print("Best Parameters:", rf_grid.best_params_)
print("Best Accuracy:", rf_grid.best_score_)




xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1]
}

xgb_grid = GridSearchCV(
    estimator=xgb,
    param_grid=xgb_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

xgb_grid.fit(X_train, y_train)

print("Best Parameters:", xgb_grid.best_params_)
print("Best Accuracy:", xgb_grid.best_score_)




lgbm_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'learning_rate': [0.01, 0.1]
}

lgbm_grid = GridSearchCV(
    estimator=lgbm,
    param_grid=lgbm_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

lgbm_grid.fit(X_train, y_train)

print("Best Parameters:", lgbm_grid.best_params_)
print("Best Accuracy:", lgbm_grid.best_score_)


import joblib

# Dictionary containing model and score
models = {
    "Random Forest": [rf_grid.best_estimator_, rf_grid.best_score_],
    "XGBoost": [xgb_grid.best_estimator_, xgb_grid.best_score_],
    "LightGBM": [lgbm_grid.best_estimator_, lgbm_grid.best_score_]
}

best_model_name = None
best_model = None
best_score = 0

for name, (model, score) in models.items():
    print(name, ":", score)

    if score > best_score:
        best_score = score
        best_model_name = name
        best_model = model

print("\nBest Model:", best_model_name)
print("Best Score:", best_score)

# Save the best model
joblib.dump(best_model, "earthquake_model.pkl")
print(model.n_features_in_)

print("Best model saved successfully!")


viz.magnitude_distribution(df)

viz.earthquake_locations(df)

viz.depth_vs_magnitude(df)

viz.risk_distribution(df)

viz.correlation_heatmap(df)

viz.feature_importance(rf, X)

viz.confusion_matrix_plot(y_test, rf_pred)