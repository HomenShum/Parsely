import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st

class UserSession:
    def welcome_user(self):
        # initialize username session state
        if "username" not in st.session_state:
            st.session_state["username"] = "Guest"

        if "username_input_bool" not in st.session_state:
            st.session_state["username_input_bool"] = False

        if st.session_state["username_input_bool"] == False:
            with st.sidebar.expander("Set Username:", expanded=True):
                prompt = st.text_input("Enter your name:")

                # display welcome message
                if prompt:
                    st.session_state["username"] = prompt
                    st.session_state["username_input_bool"] = True
                    st.toast(f"Welcome back, {st.session_state['username']}!")
                    st.sidebar.success(f"Welcome back, {st.session_state['username']}!")
                    
                    st.rerun()

    # def typewriter(self, text: str, speed: int):
    #     tokens = text.split()
    #     container = st.empty()
    #     for index in range(len(tokens) + 1):
    #         curr_full_text = " ".join(tokens[:index])
    #         container.markdown(curr_full_text)
    #         time.sleep(1 / speed)

    def table_of_contents(self, markdowns):
        # Table of Contents Expander
        with st.sidebar.expander("Table of Contents", expanded=True):
            st.markdown(markdowns)

    def about_us(self):
        # About Us Expander
        with st.sidebar.expander("About Us", expanded=True):
            st.markdown("## About Us")
            
            st.markdown("## Our Mission")
            st.markdown("1. We are creating a day to day technology that can be accessible on web, phone, and desktop application to make your day to day seamlessly fluent. We understand your time waste, and we design your time efficiency tool.")
            st.markdown("2. Make flaws and mistakes to while producing fast and iterative testable prototype applications to enhance your daily efficiencies. The goal is to achieve the end result as quickly as possible and then improve upon that.")
            st.markdown("3. The best intellectual tools and ideas will be featured as a part of the Fluency AI ecosystem where you get to profit share from user’s usages.")
            
            st.markdown("## Our Vision")
            st.markdown("1. Anyone experiencing any time constraints during their daily work activities should benefit from the added efficiency produced through automation pipelines. No more distractions, no more time wastes. You manual tasks are incrementally deducted through this product, where your result and impact is highlighted and maximized.")
            
            st.markdown("## Our Value")
            st.markdown("1. Time cost spent on meetings, notes and organizations, insight extractions. We create an AI that just works, seamlessly.")
            
            st.markdown("## Our Team")
            st.subheader("🤝 Automate your work process here, efficiently!")
            st.write("For us to design these more directed for your need, make sure to leave cafecornerwork@gmail.com the exact work flow, input, and result - think about teaching us like a junior/analyst so we can live through the process. You can also check out the [documentation](https://oval-brain-91c.notion.site/Docs-Cafe-Corner-655a9db04185481fa2c58fd28e393a28?pvs=4) for more information.")

    def privacy_security_terms_of_use_faq(self):
        # Privacy, Security, Terms of Use, FAQ Expander
        with st.sidebar.expander("Privacy, Security, Terms of Use, FAQ", expanded=True):
            st.markdown("## Privacy")
            st.markdown("""
                - OpenAI API Key usage
                - Data handling and storage
                - Cookies and third-party services
            """)
            st.markdown("## Security")
            st.markdown("""
                - Data encryption methods
                - Data access and authorization
            """)
            st.markdown("## Terms of Use")
            st.markdown("""
                - Compliance with laws
                - User conduct and content guidelines
                - AI Assistant usage
            """)
            st.markdown("## FAQ")
            st.markdown("""
                - Common questions and answers
                - Contact information for further queries
            """)

class st_files:
    def upload_file():
        uploaded_file = st.sidebar.file_uploader("Upload a file")
        if uploaded_file is not None:
            st.session_state["file"] = uploaded_file

    def display_file():
        if "file" in st.session_state:
            st.sidebar.write("Selected file:")
            st.sidebar.selectbox(st.session_state["file"])

class redirect_pages:
    def redirect_nav(self):
        # st.divider()
        # st.subheader("🧭 Cafe Corner Apps Nav Bar")
        rd_nav_col0, rd_nav_col1, rd_nav_col2, rd_nav_col3, rd_nav_col4, rd_nav_col5, rd_nav_col6, rd_nav_col7, rd_nav_col8, rd_nav_col9 = st.columns(10)
        with rd_nav_col0:
            rd_nav_button0 = st.button("Go Home")
            if rd_nav_button0:
                switch_page("home")
        with rd_nav_col1:
            rd_nav_button1 = st.button("Fluency AI")
            if rd_nav_button1:
                switch_page("app - fluency ai")
        with rd_nav_col2:
            rd_nav_button2 = st.button("Meeting AI")
            if rd_nav_button2:
                switch_page("app - meeting ai (soon)")
        with rd_nav_col3:
            rd_nav_button3 = st.button("LIVE AI")
            if rd_nav_button3:
                switch_page("app - live ai (soon)")
        # st.divider()

class uauth_tools:
    def send_email(self, receiver_email, subject, message):
        # Sender's email credentials
        sender_email = 'cafecornerwork@gmail.com'
        sender_password = 'CafeCornerWorkisFun!'

        # Create a multipart message
        email_message = MIMEMultipart()
        email_message['From'] = sender_email
        email_message['To'] = receiver_email
        email_message['Subject'] = subject

        # Add the message body
        email_message.attach(MIMEText(message, 'plain'))

        print("Sending email...")
        print(email_message)

        # Create a secure connection with the SMTP server
        with smtplib.SMTP_SSL('smtp.example.com', 465) as server:
            # Login to the sender's email account
            server.login(sender_email, sender_password)

            # Send the email
            server.send_message(email_message)

