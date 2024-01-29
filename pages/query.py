import streamlit as st

from streamlit_ace import st_ace

def save_query(query):
    with open('pages/query.txt', 'w') as f:
        f.write(query)    

st.title("Query builder")
#create an input box (large text box) where the user can create the query
query = st_ace(open('pages/query.txt', 'r').read(),
               )
                     
if query:
    save_query(query)