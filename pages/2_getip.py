import streamlit as st
from requests import get

p = st.text_input('passwd')
st.write(p)
ip = get('https://api.ipify.org').content.decode('utf8')
if p == '0240':
    st.subheader('IP: {}'.format(ip))
