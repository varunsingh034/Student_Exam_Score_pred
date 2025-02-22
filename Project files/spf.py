import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from sklearn.linear_model import LinearRegression

spf_df = pd.read_csv('D:\Programs\GIT\Student_Exam_Score_pred\Dataset\spf.csv')

spf_df.drop(columns=['School_Type','Parental_Education_Level','Gender','Internet_Access',
                     'Learning_Disabilities','Distance_from_Home','Extracurricular_Activities',
                     'Parental_Involvement','Family_Income','Teacher_Quality'], inplace=True)

spf_df['Exam_Score'].replace(101,100,inplace=True)

x_train, x_test, y_train, y_test = train_test_split(spf_df.iloc[:,:9], spf_df.iloc[:,9], test_size=0.2,random_state=42)

oe_atr = OrdinalEncoder(categories=[['Low','Medium','High']])
x_train_atr = oe_atr.fit_transform(x_train[['Access_to_Resources']])
x_test_atr = oe_atr.transform(x_test[['Access_to_Resources']])

oe_ml = OrdinalEncoder(categories=[['Low','Medium','High']])
x_train_ml = oe_ml.fit_transform(x_train[['Motivation_Level']])
x_test_ml = oe_ml.transform(x_test[['Motivation_Level']])

oe_pi = OrdinalEncoder(categories=[['Negative','Neutral','Positive']])
x_train_pi = oe_pi.fit_transform(x_train[['Peer_Influence']])
x_test_pi = oe_pi.transform(x_test[['Peer_Influence']])

x_train_rem = x_train.drop(columns=['Access_to_Resources','Motivation_Level','Peer_Influence'])
x_test_rem = x_test.drop(columns=['Access_to_Resources','Motivation_Level','Peer_Influence'])

x_train_main = np.concatenate((x_train_rem,x_train_atr,x_train_ml,x_train_pi),axis=1)
x_test_main = np.concatenate((x_test_rem,x_test_atr,x_test_ml,x_test_pi),axis=1)

lr = LinearRegression()
lr.fit(x_train_main,y_train)

y_pred_lr = lr.predict(x_test_main)

print(r2_score(y_test, y_pred_lr))
