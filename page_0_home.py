import sys
from pathlib import Path

# Add the parent directory to the PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_card import card
import pandas as pd
import base64

def home_page():
    st.title("🌱 Welcome to Parsely")

    if "username" not in st.session_state:
        st.session_state["username"] = "Glad we got to meet at this Rag-a-thon!"

    st.divider()

    st.header(f"🙌 Nice to meet you! {st.session_state['username']} 😊") 
    st.markdown("""
    **Parsely** - Boost your productivity. Spend less time on tasks and more on what truly matters. Our toolkit enhances your workspace for both personal and business use cases.

    We're currently building our website. Pardon our dust! We're striving to get everything up and running swiftly. Meanwhile, explore our tools and share your thoughts. We appreciate your feedback! 😊
    """)
    st.divider()

    st.header("📲 Quick Start")

    # 3 columns
    col1, col2, col3 = st.columns(3)

    with col1:
        with open("assets/images/fluency_ai_img.jpg", "rb") as f:
            data = f.read()
            encoded = base64.b64encode(data)
        col1_img = "data:image/png;base64," + encoded.decode("utf-8")

        col1_Clicked = card(
            title="Fluency AI",
            text="Transform your work process when using PDF, Excel, JPG",
            image=col1_img,
        )

        if col1_Clicked:
            switch_page("app - fluency ai")

    with col2:
        with open("assets/images/meeting_ai_img.jpg", "rb") as f:
            data = f.read()
            encoded = base64.b64encode(data)
        col2_img = "data:image/png;base64," + encoded.decode("utf-8")

        col2_Clicked = card(
            title="Meeting AI",
            text="Ease your meetings with auto-transcription, auto-summarization, and more!",
            image=col2_img,
        )

        if col2_Clicked:
            switch_page("app - meeting ai (soon)")

    with col3:
        with open("assets/images/live_ai_img.jpg", "rb") as f:
            data = f.read()
            encoded = base64.b64encode(data)
        col3_img = "data:image/png;base64," + encoded.decode("utf-8")

        col3_Clicked = card(
            title="LIVE AI",
            text="Describe your live events with video and audio recordings transformed into long-form conversations",
            image=col3_img,
        )

        if col3_Clicked:
            switch_page("app - live ai (soon)")

    st.divider()
    #################### Descriptions #####################################
    st.header("📝 What do they do?")

    # description 3 columns
    desc_col1, desc_col2, desc_col3 = st.columns(3)

    with desc_col1:
        st.subheader("Fluency AI")
        with st.expander("Description", expanded=True):
            st.write("This is a full suite solution for works using PDF, Excel, JPG")
            st.write("It includes:")
            st.write("Optical Character Recognition (OCR)")
            st.write("Text Summarization")
            st.write("Text Translation")
            st.write("Documentation Generation")

    with desc_col2:
        st.subheader("Meeting AI")
        with st.expander("Description", expanded=True):
            st.write("This is a solution to help you with your meetings")
            st.write("It includes:")
            st.write("Live Transcription during online meetings")
            st.write("Live Summarization")
            st.write("Live Translation")
            st.write("Meeting Insight and Analytics")

    with desc_col3:
        st.subheader("LIVE AI")
        with st.expander("Description", expanded=True):
            st.write("This is a solution to help you with your live events through video or audio recordings")
            st.write("For example:")
            st.write("Transcribing a live event and describing the scene based on video with audio")
            st.write("Live translation of a live event")
            st.write("Live summarization of a live event")

    st.divider()
    #################### Pricing Information ##############################
    st.header("💳 Pricing Information")

    st.table(pd.DataFrame({"Monthly Plan": ["Free", "Basic", "Premium"], 
                            "Price": ["$0", "$20", "$95"], 
                            "Features": ["Unlimited OCR Text and Table Extraction, Unlimited Translation, 10 Queries of Image/PDF with Table/Excel RAG Chat, 1 High Performance Mode", 
                                        "Unlimited OCR Text and Table Extraction, Unlimited Translation, Unlimited Image/PDF with Table/Excel RAG Chat, 10 High Performance Mode", 
                                        "OCR, Summarization, Translation, Documentation Generation, Chat, Deployable Chatbot using your URL"]}))
        
    st.divider()
    #################### Documentation ####################################
    st.header("📚 Documentation")

    st.markdown("In the process of preparing *documentations* and *gif demos* for you to get started, but feel free to explore intuitively for now! :)")
    st.markdown("If anything does not feel intuitive to use or seems confusing, please reach out to us at email address at end of the page.")
    # 3rd 3 columns
    third_col1, third_col2, third_col3 = st.columns(3)
    with third_col1:
        st.markdown("### Example 1: OCR Text Extraction")
        imgocrtoolv1_video_file = open('assets\\videos\\imgocrtoolv1.mp4', 'rb')
        imgocrtoolv1_video_bytes = imgocrtoolv1_video_file.read()
        st.video(imgocrtoolv1_video_bytes)
        st.info("This is a video of the OCR Text Extraction tool in action. It is a tool that allows you to extract text from images, PDFs, and Excel files. It also allows you to extract information from receipt or business cards onto an organized excel document.")
        
        st.write("If you have any questions, feel free to reach out to us at cafecornerwork@gmail.com")
    with third_col2:
        st.markdown("### Coming Soon!")
    with third_col3:
        st.markdown("### Coming Soon!")