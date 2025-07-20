import streamlit as st
from streamlit_option_menu import option_menu

def custom_sidebar():
    menu_options = ["Dashboard", "Visualize", "Feature Engineering", "ML Training", "DL Training"]

    # Initialize if not set
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"

    with st.sidebar:
        selected = option_menu(
            menu_title=None,
            options=menu_options,
            icons=["speedometer2", "bar-chart", "cpu", "robot", "robot"],
            menu_icon="cast",
            default_index=menu_options.index(st.session_state.page)
            if st.session_state.page in menu_options else 0,
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

        # Only update if user actually selected something different
        if selected != st.session_state.page:
            st.session_state.page = selected
            st.rerun()
