from formating_data import get_df
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

df_test = get_df(folder_path='test/*.wav', mode='test')
df = get_df(folder_path='train/*.wav', mode='train')

# Фичи (все столбцы, кроме последнего)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

scaler = StandardScaler()
X = scaler.fit_transform(X)
test = scaler.transform(df_test)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

clf = RandomForestClassifier(max_depth=5)
clf.fit(X, y)
y_pred = clf.predict(test)
print(y_pred)
df = pd.DataFrame(y_pred, index=df_test.index)
df.to_csv('answers.tsv', header=False, index=True, sep="\t")