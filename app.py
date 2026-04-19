import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

st.title('Will you Survive the Titanic?')
pname = st.text_input('Enter Name')
age = st.number_input('Age',0,100)
sex = st.selectbox('Gender',['male','female'])
pclass = st.slider('Select the Class between 1 to 3',1,3)
fare = st.number_input('Fare paid by the passenger')
sibsp = st.slider('Number of siblings/spouses',0,8)
embarked = st.selectbox('Select the station from where your journey started',['Southampton','Chebourg','Queenstown']) 
parch = st.slider('Number of parents/children',0,6)

data = pd.DataFrame([{'Pclass': pclass, 'Sex': sex, 'SibSp': sibsp, 'Parch': parch, 'Fare': fare, 'Embarked': embarked}])
data1 = pd.DataFrame([{'Name': pname, 'Age': age, 'Pclass': pclass, 'Sex': sex, 'SibSp': sibsp, 'Parch': parch, 'Fare': fare, 'Embarked': embarked}])
if st.button('Passenger Details'):
    st.write('The input data is:', data1)

model = load_model('model.h5')
with open('label_encoder.pkl','rb') as file:
    label = pickle.load(file)
with open('onehot_encoder.pkl','rb') as file:
    onehot = pickle.load(file)
with open('scalar_encoder.pkl','rb') as file:
    scalar = pickle.load(file)

data['Sex'] = label.transform(data['Sex'])

embarked = onehot.transform(data[['Embarked']])
embarked = pd.DataFrame(embarked, columns = onehot.get_feature_names_out())
data =pd.concat([data.drop(columns = ['Embarked']),embarked],axis=1)

data[['Pclass','SibSp','Parch','Fare']] = scalar.transform(data[['Pclass','SibSp','Parch','Fare']])

y = model.predict(data)
y = y[0][0]
def Chance(y):
    if y>0.5:
        return('The passenger is likely to survive.')
    else:
        return('The passenger is not likely to survive.')
    
if st.button('Show Prediction'):
    st.write('Probability of survival:', y)
    st.write(Chance(y))