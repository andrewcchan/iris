import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
import plotly.express as px

# model from here: https://colab.research.google.com/drive/1RROEcn1EaI564_TlptQquU9HF68q7E38?usp=sharing

# APP

st.title('Iris Dataset')
'''https://colab.research.google.com/drive/10O1KYBqC_Cw1lMTbS2x4bCSRTsEdS7_r#scrollTo=hIOxoY8S7dE2'''


# sepal length (cm) min 4.3 max 7.9
# sepal width (cm) min 2.0 max 4.4
# petal length (cm) min 1.0 max 6.9
# petal width (cm) min 0.1 max 2.5


sepal_length = st.slider('sepal length', 4.3, 7.9, (4.3+7.9)/2)
sepal_width = st.slider('sepal width', 2.0, 4.4, (2.0+4.4)/2)
petal_length = st.slider('petal length', 1.0, 6.9, (1.0+6.9)/2)
petal_width = st.slider('petal width', 0.1, 2.5, (0.1+2.5)/2)

# load clf
clf = load('./iris.joblib')


df = pd.DataFrame(dict(
    r=[sepal_length,sepal_width, petal_length,petal_width],
    theta=['sepal_length','sepal_width','petal_length','petal_width']]))
fig = px.line_polar(df, r='r', theta='theta', line_close=True)
st.plotly_chart(fig)


'''Prediction'''
res = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
st.write(res)
