import hashlib
import streamlit as st
from requests import get
import hmac

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(hashlib.sha256(st.session_state["password"].encode()).hexdigest(), 'db2e09a74a317a27218bad97bdedf4aabda69c6cb7ebf9910b7fbce6f9565074'):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the passward is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False

if check_password():
    ip = get('https://api.ipify.org').content.decode('utf8')
    st.subheader('IP: {}'.format(ip))
