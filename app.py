%%writefile app.py
import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('/content/drive/My Drive/nishi_model.pkl', 'rb')) 
# Feature Scaling
dataset = pd.read_csv('/content/drive/My Drive/main.csv')
# Extracting independent variable:
X = dataset.iloc[:, [1,2,3,4,5,6,7]].values
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

def predict_note_authentication(Gender,Glucose,BP,SkinThickness, Insulin, BMI , PedigreeFunction):
  output= model.predict([[Gender,Glucose,BP,SkinThickness, Insulin, BMI , PedigreeFunction]])
  print("heartdisease", output)
  if output==[0]:
    prediction="No Heartdisease"
  else :
    prediction="Heartdisease"
  
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:Brown;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:black;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:black;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:black;margin-top:10px;"Machine Learning Endterm lab Experiment</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Heartdisease Predictor")

    Gender = st.number_input('Insert a Gender',0,1)
    Glucose= st.number_input('Insert a 1-4 ')
    BP=st.number_input('Insert a BP reading ')
    SkinThickness = st.number_input('Insert between 100 - 200 ')
    Insulin = st.number_input('Insert between 100 - 400 ')
    BMI = st.number_input('Insert 0 or 1 ')
    PedigreeFunction = st.number_input('Insert  ')

    
   
    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(Gender,Glucose,BP,SkinThickness, Insulin, BMI , PedigreeFunction)
      st.success('Model has predicted {}'.format(result))
      
    if st.button("About"):
      st.subheader("Developed by Nishi Singh")
      st.subheader("Head , Department of Computer Engineering")

if __name__=='__main__':
  main()
