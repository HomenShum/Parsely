import nest_asyncio
nest_asyncio.apply()


import sys
from pathlib import Path

# Add the parent directory to the PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))
import streamlit as st
from utils import UserSession, redirect_pages
from streamlit_extras.buy_me_a_coffee import button as buy_me_a_coffee_button
from streamlit_extras.switch_page_button import switch_page
from streamlit_card import card
import pandas as pd
import base64
from icecream import ic
from page_0_home import home_page
from page_1_chatallfiles import chatallfiles_page
from page_3_productrec import product_recommendation_builder

##### Global Settings #########################################################################################
st.set_page_config(
    page_title="Parsely", 
    page_icon="🌱", 
    layout="wide",
    initial_sidebar_state="expanded"
)
##### Sidebar Elements #####################################################################################
##### Test: sac.menu ###########################################################################################
import streamlit_antd_components as sac

with st.sidebar:
    menu = sac.menu([
        sac.MenuItem('Home', icon='house-fill'),
        sac.MenuItem('Products', icon='box2-heart-fill', children=[
            sac.MenuItem('General', icon='box', children=[
                sac.MenuItem('Chat for All Files', icon='chat'),
                sac.MenuItem('Excel Data Classification', icon='file-earmark-spreadsheet'),
            ]),
            sac.MenuItem('B.Y.O.B.', icon='robot', children=[
                sac.MenuItem('Product Rec Builder', icon='star'),
                sac.MenuItem('Booking Agent Builder', icon='calendar-plus'),
            ]),
            sac.MenuItem('File Activity', icon='file-earmark', children=[
                # organized by, summarizations, files preview
                sac.MenuItem('Organized by', icon='folder'),
                sac.MenuItem('Summarizations', icon='file-earmark-text'),
                sac.MenuItem('Files Preview', icon='eye'),
            ]),
        ]),
    ], open_all=True)

    ic(menu)

if menu:
    # sac.alert(f"Redirect to {str(menu)} Page", color="success",banner=True, icon=True, closable=True)
    st.toast(f"Redirect to {str(menu)} Page")

##### Main Elements ###########################################################################################
if menu == "Home":
    buy_me_a_coffee_button('homenshum', bg_color="#6f4e37", font_color="#fcf9f8", coffee_color="#376f4e")
    home_page()

if menu == "Chat for All Files":
    chatallfiles_page()

if menu == "Excel Data Classification":
    st.write("Excel Data Classification")

if menu == "Product Rec Builder":
    # st.write("Product Rec Builder")
    product_recommendation_builder()

if menu == "Booking Agent Builder":
    st.write("Booking Agent Builder")

if menu == "Organized by":
    st.write("Organized by")

if menu == "Summarizations":
    st.write("Summarizations")

if menu == "Files Preview":
    st.write("Files Preview")



