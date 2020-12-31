"""Home page shown when the user enters the application"""
import streamlit as st
import streamlit.components.v1 as components


# pylint: disable=line-too-long
def display():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading About ..."):
        st.title("About")
        st.markdown(
            """
            ## Contributions
            This is a simple streamlit app. 
            
            The app has three apps:
            - About: Simple markdown and iframe
            - Machine Learning: Simple scikit-learn training and visualization
            - Visualization: NY Crash map plots


            
            
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Helpful Streamlit Reference Guide")

        components.iframe("https://share.streamlit.io/daniellewisdl/streamlit-cheat-sheet/app.py",height=2000,scrolling=False)
