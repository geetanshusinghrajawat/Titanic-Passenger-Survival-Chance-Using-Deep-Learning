import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

st.title('Will you Survive the Titanic?')

pclass = st.slider('Enter the Passenger Class from 1 to 3',1,3)
sex = st.selectbox('Enter Gender',['male','female'])
sibsp = st.slider('Enter the number of siblings/spouses of the passenger',0,8)
parch = st.slider('Enter the number of parents/children of the passenger',0,6)
fare = st.number_input('Enter the fare paid by the passenger')
embarked = st.selectbox('Enter the station where the journey started',['Southampton','Chebourg','Queenstown']) 

data = pd.DataFrame([{'Pclass': pclass, 'Sex': sex, 'SibSp': sibsp, 'Parch': parch, 'Fare': fare, 'Embarked': embarked}])
if st.button('Passenger Details'):
    st.write('The input data is:', data)

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