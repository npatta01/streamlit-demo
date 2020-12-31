from logging import getLogger

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px




import apps.ml.app
import apps.about.app
import apps.visualization.app






logger = getLogger(__file__)
logger.info(f"Running Streamlit app")





PAGES = {
    "About": apps.about.app,
    "Machine Learning": apps.ml.app,
    "Visualization": apps.visualization.app,
    
}



def main():
    """Main function of the App"""


    st.set_page_config(
                page_title=f"Demo Streamlit App",
                page_icon="🧊",
                layout="wide",
                initial_sidebar_state="expanded",
                )


    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    
    page = PAGES[selection]

    with st.spinner(f"Loading {selection} ..."):
        page.display()
        
        


    st.sidebar.title("About")
    st.sidebar.info(
        """
        This is a demo app showing the beauty of [streamlit](https://www.streamlit.io/).
        
        The code can be found [here]()

        """
    )


if __name__ == "__main__":
    main()