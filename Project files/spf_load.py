import numpy as np
import pickle

oe_atr = pickle.load(open("D:\Programs\GIT\Student_Exam_Score_pred\Model\oe_atr.pkl", "rb"))
oe_ml = pickle.load(open("D:\Programs\GIT\Student_Exam_Score_pred\Model\oe_ml.pkl", "rb"))
oe_pi = pickle.load(open("D:\Programs\GIT\Student_Exam_Score_pred\Model\oe_pi.pkl", "rb"))
lr = pickle.load(open("D:\Programs\GIT\Student_Exam_Score_pred\Model\lr_model.pkl", "rb"))

test_input = np.array([10,70,7,80,4,4,'High','Medium','Neutral'],dtype=object).reshape(1, -1)

test_input_atr = oe_atr.transform(test_input[:,6].reshape(1, 1))
test_input_ml = oe_ml.transform(test_input[:,7].reshape(1, 1)) 
test_input_pi = oe_pi.transform(test_input[:,8].reshape(1, 1))

test_input_transformed = np.concatenate((test_input[:,0:6], test_input_atr, test_input_ml, test_input_pi), axis=1)

test_pred = lr.predict(test_input_transformed)

print(test_pred)

