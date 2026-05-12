import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import StandardScaler

# load data
df = pd.read_csv('census_tracts.csv')
df = df[df['city'] == 'Washington'].copy()

#build gentrification label from demographic columns
high_edu = df['educational_attainment'] > df['educational_attainment'].median()
high_white = df['white_alone'] > df['white_alone'].median()
low_black = df['black_alone'] < df['black_alone'].median()
df['gentrification_risk'] = ((high_edu) & (high_white) & (low_black)).astype(int)

features = ['total_population','median_income', 'median_home_value',
            'hispanic_or_latino', 'total_population_25_over']

X = df[features]
y = df['gentrification_risk']

# split data 80/20
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2,random_state=42, stratify=y)

# scale for logistic regression
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

#logistic regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_s,y_train)
lr_preds = lr.predict(X_test_s)

print('Logistic Regression')
print('Accuracy:', round(accuracy_score(y_test,lr_preds),3))
print('F1:', round(f1_score(y_test, lr_preds),3))

# random forest
rf = RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print('\nRandom Forest')
print('Accuracy:', round(accuracy_score(y_test, rf_preds), 3))
print('F1:', round(f1_score(y_test,rf_preds),3))

importances = pd.Series(rf.feature_importances_,index=features).sort_values(ascending=False)
print('\nFeature importances:')
print(importances)
