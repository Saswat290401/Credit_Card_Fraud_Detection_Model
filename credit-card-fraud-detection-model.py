import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('creditcard.csv')

print(df.head())

df.info()

print(df.describe())

class_counts = df['Class'].value_counts()
print(class_counts)


counts = df['Class'].value_counts()
normal_count = counts[0]
fraud_count = counts.get(1, 0)
print("Normal Transactions:", normal_count)
print("Fraudulent Transactions:", fraud_count)


plt.bar(['Normal', 'Fraud'], [normal_count, fraud_count])
plt.title("Distribution of Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.show()


for col in cols:
    plt.figure(figsize=(10, 5))
    sns.displot(df[col], kde=False)
    plt.title(f'Distribution of {col}')
    plt.show()


for col in cols:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()


df.hist(bins=20, figsize=(20, 20))
plt.show()


corr_matrix = df.corr()

plt.figure(figsize=(35, 40))

sns.heatmap(corr_matrix, cmap='Spectral', annot=True)

plt.title('Correlation Matrix of Credit Card Fraud Dataset')

plt.show()



sns.jointplot(x='Time', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V1', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V2', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V3', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V4', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V5', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V6', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V7', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V8', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V9', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V10', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V11', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V12', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V13', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V14', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V15', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V16', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V17', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V18', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V19', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V20', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V21', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V22', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V23', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V24', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V25', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V26', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V27', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='V28', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='Amount', y='Amount', data=df, kind='scatter')
plt.show()

sns.jointplot(x='Class', y='Amount', data=df, kind='scatter')
plt.show()



fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist(df['Time'], bins=50)
ax[0].set_title('Distribution of Time')
ax[1].hist(df['Amount'], bins=50)
ax[1].set_title('Distribution of Amount')
plt.show()


from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


imputer = SimpleImputer(strategy='mean')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


label_encoder = LabelEncoder()
df['Class'] = label_encoder.fit_transform(df['Class'])


df['Class'] = df['Class'].astype('category')
df['Class'] = df['Class'].cat.codes


print(df['Class'].value_counts())



df.drop_duplicates(inplace=True)
df.fillna(method='ffill', inplace=True)

encoder = LabelEncoder()
df['Class'] = encoder.fit_transform(df['Class'])

scaler = StandardScaler()
numerical_cols = ['Time', 'Amount'] + [col for col in df.columns if col.startswith('V')]
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

X = df.drop('Class', axis=1)
y = df['Class']


from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import warnings


df['Class'] = df['Class'].apply(lambda x: 1 if x == 'Fraud' else 0)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    selector = SelectKBest(f_classif, k=10)
    X_new = selector.fit_transform(X, y)


feature_names = np.array(df.drop('Class', axis=1).columns)[selector.get_support()]

pca = PCA(n_components=10)
pca_features = pca.fit_transform(df.drop(['Class'], axis=1))


plt.scatter(pca_features[:, 0], pca_features[:, 1], c=y)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


print('Shape of X_train:', X_train.shape)
print('Shape of y_train:', y_train.shape)
print('Shape of X_test:', X_test.shape)
print('Shape of y_test:', y_test.shape)


unique, counts = np.unique(y_train, return_counts=True)
print(dict(zip(unique, counts)))


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)
logreg_predictions = logreg.predict(X_test_scaled)
print("Logistic Regression Results:")
print(classification_report(y_test, logreg_predictions))


rf = RandomForestClassifier()
rf.fit(X_train_scaled, y_train)
rf_predictions = rf.predict(X_test_scaled)
print("Random Forest Results:")
print(classification_report(y_test, rf_predictions))


mlp = MLPClassifier(hidden_layer_sizes=(64, 64, 64), random_state=42)
mlp.fit(X_train_scaled, y_train)
mlp_predictions = mlp.predict(X_test_scaled)
print("Neural Network Results:")
print(classification_report(y_test, mlp_predictions))


dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)
dt_predictions = dt.predict(X_test_scaled)
print("Decision Tree Results:")
print(classification_report(y_test, dt_predictions))


gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train_scaled, y_train)
gb_predictions = gb.predict(X_test_scaled)
print("Gradient Boosting Results:")
print(classification_report(y_test, gb_predictions))


y_pred_logreg = logreg.predict_proba(X_test)[:, 1]
y_pred_rf = rf.predict_proba(X_test)[:, 1]
y_pred_mlp = mlp.predict_proba(X_test)[:, 1]
y_pred_dt = dt.predict_proba(X_test)[:, 1]
y_pred_gb = gb.predict_proba(X_test)[:, 1]


precision_logreg, recall_logreg, _ = precision_recall_curve(y_true, y_pred_logreg)
precision_rf, recall_rf, _ = precision_recall_curve(y_true, y_pred_rf)
precision_mlp, recall_mlp, _ = precision_recall_curve(y_true, y_pred_mlp)
precision_dt, recall_dt, _ = precision_recall_curve(y_true, y_pred_dt)
precision_gb, recall_gb, _ = precision_recall_curve(y_true, y_pred_gb)


auprc_logreg = auc(recall_logreg, precision_logreg)
auprc_rf = auc(recall_rf, precision_rf)
auprc_mlp = auc(recall_mlp, precision_mlp)
auprc_dt = auc(recall_dt, precision_dt)
auprc_gb = auc(recall_gb, precision_gb)


plt.figure(figsize=(8, 6))
plt.plot(recall_logreg, precision_logreg, label='Logistic Regression (AUPRC = {:.4f})'.format(auprc_logreg))
plt.plot(recall_rf, precision_rf, label='Random Forest (AUPRC = {:.4f})'.format(auprc_rf))
plt.plot(recall_mlp, precision_mlp, label='Neural Network (Multi-layer Perceptron) (AUPRC = {:.4f})'.format(auprc_mlp))
plt.plot(recall_dt, precision_dt, label='Decision Tree (AUPRC = {:.4f})'.format(auprc_dt))
plt.plot(recall_gb, precision_gb, label='Gradient Boosting (AUPRC = {:.4f})'.format(auprc_gb))


plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Multiple ML Algorithms')
plt.legend(loc='lower left')
plt.show()

import pickle

with open('D:\Credit Card Fraud\fraud_detection_model.pkl', 'wb') as f:
    pickle.dump(model, f)