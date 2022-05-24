import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load

# APP

st.title('Iris Dataset')
'''https://colab.research.google.com/drive/10O1KYBqC_Cw1lMTbS2x4bCSRTsEdS7_r#scrollTo=hIOxoY8S7dE2'''

clf = load('./iris.joblib')
testRes = clf.predict([[-0.8, -1]])
testRes