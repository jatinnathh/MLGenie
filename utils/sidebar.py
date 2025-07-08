import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

def custom_sidebar():
    with st.sidebar:
        selected = option_menu(
            menu_title=None,  # No title
            options=["Dashboard", "Visualize", "Feature Engineering", "ML Training", "DL Training"],
            icons=["speedometer2", "bar-chart", "cpu", "robot", "robot"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {
                    "background-color": "#0e0e0e",
                    "padding": "10px",
                    "border-radius": "20px",
                    "radius": "20px",
                },
                "icon": {"color": "#ffffff", "font-size": "20px"},
                "nav-link": {
                    "color": "#ffffff",
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "5px 0px",
                    "--hover-color": "#333333",
                },
                "nav-link-selected": {
                    "background-color": "#2c2c2c",
                    "color": "#ffffff",
                },
            }
        )
    return selected
