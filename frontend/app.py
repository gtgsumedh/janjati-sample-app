import streamlit as st
import requests
import os

st.title("🏹 Janjati Knowledge System: Connection Test")
st.write("This UI tests the connection to the FastAPI backend.")

# Use localhost for local testing, but allow an environment variable for production
API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:10000")

user_name = st.text_input("Enter your name:")

if st.button("Send to Backend"):
    if user_name:
        try:
            # Make a POST request to the FastAPI endpoint
            response = requests.post(
                f"{API_BASE_URL}/api/greet", 
                json={"name": user_name}
            )
            
            if response.status_code == 200:
                data = response.json()
                st.success(f"Backend Reply: {data['message']}")
            else:
                st.error(f"Error {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("🚨 Could not connect to the backend. Is FastAPI running?")
    else:
        st.warning("Please enter a name first.")