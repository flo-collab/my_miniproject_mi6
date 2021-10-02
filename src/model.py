import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings
warnings.filterwarnings("ignore")

data= pd.read_csv(r'../data/CURATED/data_clean.csv', encoding="UTF8", sep=',')

df=data.copy()
df.Sex = df.Sex.map({"male":0, "female":1}) #essayer avec/et sans dummies
df = pd.get_dummies(data=df, columns=['Pclass', 'Embarked'])

X = df.drop(['Survived'], axis=1)
y = df.Survived
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



RFC = RandomForestClassifier()
RFC.fit(X_train, y_train)
pickle.dump(RFC,open('RFC_model.pkl','wb'))