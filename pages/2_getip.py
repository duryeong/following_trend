import streamlit as st
from requests import get

ip = get('https://api.ipify.org').content.decode('utf8')
st.subheader('My public IP address is: {}'.format(ip))