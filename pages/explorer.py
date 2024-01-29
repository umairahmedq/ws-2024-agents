# read the research db and display the results from the table literature
import streamlit as st
import pandas as pd
import sqlite3



st.title("data explorer")
connection = sqlite3.connect('research.db')
data = pd.read_sql('SELECT * FROM literature', connection)
st.write(data)