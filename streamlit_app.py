import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
import plotly.express as px


# APP

st.title('Iris Dataset')
'''https://colab.research.google.com/drive/10O1KYBqC_Cw1lMTbS2x4bCSRTsEdS7_r#scrollTo=hIOxoY8S7dE2'''

clf = load('./iris.joblib')
testRes = clf.predict([[-0.8, -1]])
testRes

sepal_length = st.slider('sepal length', 0.0, 5.0, 2.5)
sepal_width = st.slider('sepal width', 0.0, 5.0, 2.5)

'''dynamic'''
res = clf.predict([[sepal_length, sepal_width]])
res


df = pd.DataFrame(dict(
    r=[sepal_length,sepal_width],
    theta=['sepal_length','sepal_width']))
fig = px.line_polar(df, r='r', theta='theta', line_close=True)
st.plotly_chart(fig)