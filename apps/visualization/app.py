"""Home page shown when the user enters the application"""
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import os

@st.cache(persist=True)
def load_data(nun_rows:int):
    path = os.path.abspath(os.path.join(os.path.dirname( __file__ ),  "nyc_motor_collision.parquet.gzip"))

    df = pd.read_parquet(path)

    df.dropna(subset=['LATITUDE', 'LONGITUDE'], inplace=True)
    midpoint = (np.median(df["LATITUDE"]), np.median(df["LONGITUDE"]))


    #midpoint = (40.730610 , -73.935242 ) for nyc
    # remove some invalid locations
    df = df[ (df.LATITUDE < midpoint[0]+10 ) &  (df.LATITUDE > midpoint[0]-10 )  ]
    df = df[ (df.LONGITUDE < midpoint[1]+10 ) &  (df.LONGITUDE > midpoint[1]-10 )  ]


    df = df.head(nun_rows)


    df.loc[:,'crash_date_crash_time'] = pd.to_datetime(df.CRASH_DATE.astype(str)+' '+df.CRASH_TIME.astype(str))



    
    lowercase = lambda x: str(x).lower()
    df.rename(lowercase, axis="columns", inplace=True)
    df.rename(columns={"crash_date_crash_time": "date_time"}, inplace=True)
            #data = data[['date/time', 'latitude', 'longitude']]
    return df




# pylint: disable=line-too-long
def display():
    """Used to write the page in the app.py file"""


    with st.spinner("Loading Visualization ..."):
        st.title("Motor Vehicle Collisions in New York City")
        st.markdown(
            """
            This application is a Streamlit dashboard that can be used 
               to analyze motor vehicle collisions in NYC 🗽💥🚗
            """,
            unsafe_allow_html=True,
            )



        data = load_data(50_000)

        midpoint = (np.median(data["latitude"]), np.median(data["longitude"]))


        st.header("Where are the most people injured in NYC?")
        injured_people = st.slider("Number of persons injured in vehicle collisions", 0, 19)
        st.map(data.query("injured_persons >= @injured_people")[["latitude", "longitude"]].dropna(how="any"))

        st.header("How many collisions occur during a given time of day?")
        hour = st.slider("Hour to look at", 0, 23)
        original_data = data
        data = data[data["date_time"].dt.hour == hour]
        st.markdown("Vehicle collisions between %i:00 and %i:00" % (hour, (hour + 1) % 24))

        

        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state={
                "latitude": midpoint[0],
                "longitude": midpoint[1],
                "zoom": 11,
                "pitch": 50,
            },
            layers=[
                pdk.Layer(
                "HexagonLayer",
                data=data[['date_time', 'latitude', 'longitude']],
                get_position=["longitude", "latitude"],
                auto_highlight=True,
                radius=100,
                extruded=True,
                pickable=True,
                elevation_scale=4,
                elevation_range=[0, 1000],
                ),
            ],
        ))
        if st.checkbox("Show raw data", False):
            st.subheader("Raw data by minute between %i:00 and %i:00" % (hour, (hour + 1) % 24))
            st.write(data)



        col_1, col_2 = st.beta_columns((2,2))

        with col_1:
            st.subheader("Breakdown by minute between %i:00 and %i:00" % (hour, (hour + 1) % 24))
            filtered = data[
                (data["date_time"].dt.hour >= hour) & (data["date_time"].dt.hour < (hour + 1))
            ]
            hist = np.histogram(filtered["date_time"].dt.minute, bins=60, range=(0, 60))[0]
            chart_data = pd.DataFrame({"minute": range(60), "crashes": hist})

            fig = px.bar(chart_data, x='minute', y='crashes', hover_data=['minute', 'crashes'], height=400)
            st.write(fig)




        with col_2:
            st.header("Top 5 dangerous streets by affected class")
            select = st.selectbox('Affected class', ['Pedestrians', 'Cyclists', 'Motorists'])

            if select == 'Pedestrians':
                st.write(original_data.query("injured_pedestrians >= 1")[["on_street_name", "injured_pedestrians"]].sort_values(by=['injured_pedestrians'], ascending=False).dropna(how="any")[:5])

            elif select == 'Cyclists':
                st.write(original_data.query("injured_cyclists >= 1")[["on_street_name", "injured_cyclists"]].sort_values(by=['injured_cyclists'], ascending=False).dropna(how="any")[:5])

            else:
                st.write(original_data.query("injured_motorists >= 1")[["on_street_name", "injured_motorists"]].sort_values(by=['injured_motorists'], ascending=False).dropna(how="any")[:5])


        
        
        with st.beta_container():

            st.markdown(
                """
                ## Acknowledgments
                This app was inspired by the Coursera course [Build a Machine Learning Web App with Streamlit and Python](https://www.coursera.org/learn/machine-learning-streamlit-python/home/welcome)

                
                
                """,
                unsafe_allow_html=True,
            )
