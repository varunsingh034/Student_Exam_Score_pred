import numpy as np
import pickle
import gradio as gr

oe_atr = pickle.load(open("D:\Programs\GIT\Student_Exam_Score_pred\Model\oe_atr.pkl", "rb"))
oe_ml = pickle.load(open("D:\Programs\GIT\Student_Exam_Score_pred\Model\oe_ml.pkl", "rb"))
oe_pi = pickle.load(open("D:\Programs\GIT\Student_Exam_Score_pred\Model\oe_pi.pkl", "rb"))
lr = pickle.load(open("D:\Programs\GIT\Student_Exam_Score_pred\Model\lr_model.pkl", "rb"))

# test_input = np.array([10,70,7,80,4,4,'High','Medium','Neutral'],dtype=object).reshape(1, -1)
# test_input_transformed = np.concatenate((test_input[:,0:6], test_input_atr, test_input_ml, test_input_pi), axis=1)
# test_pred = lr.predict(test_input_transformed)
# print(test_pred)


  
def handle_submission(num1, num2, num3, num4, num5, num6, cat1, cat2, cat3):
    test_input = np.array([num1, num2, num3, num4, num5, num6, cat1, cat2, cat3],dtype=object).reshape(1, -1)
    test_input_atr = oe_atr.transform(test_input[:,6].reshape(1, 1))
    test_input_ml = oe_ml.transform(test_input[:,7].reshape(1, 1)) 
    test_input_pi = oe_pi.transform(test_input[:,8].reshape(1, 1)) 
     
    test_input_transformed = np.concatenate((test_input[:,0:6], test_input_atr, test_input_ml, test_input_pi), axis=1)
    # print(test_input_transformed)
    test_pred = lr.predict(test_input_transformed)
    
    return test_pred[0]

with gr.Blocks() as demo:
    gr.Markdown("# Enter 6 Numbers and Select 3 Categories")
    
    with gr.Row():
        num1 = gr.Number(label="Hours_Studied")
        num2 = gr.Number(label="Attendance")
        num3 = gr.Number(label="Sleep_Hours")
    
    with gr.Row():
        num4 = gr.Number(label="Previous_Scores")
        num5 = gr.Number(label="Tutoring_Sessions")
        num6 = gr.Number(label="Physical_Activities")
    
    with gr.Row():
        cat1 = gr.Dropdown(choices=["Low", "Medium", "High"], label="Access_to_Resources")
        cat2 = gr.Dropdown(choices=["Low", "Medium", "High"], label="Motivation_Level")
        cat3 = gr.Dropdown(choices=["Negative", "Neutral", "Positive"], label="Peer_Influence")
    
    submit = gr.Button("Submit Data")
    output = gr.Textbox(label="Predicted Score")
    
    submit.click(handle_submission, [num1, num2, num3, num4, num5, num6, cat1, cat2, cat3], output)

demo.launch(share=True)