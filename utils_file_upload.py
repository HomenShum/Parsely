import streamlit as st
import streamlit_antd_components as sac
import os
import re
import asyncio
import aiohttp
from aiohttp import ClientTimeout
from openai import AsyncOpenAI
import instructor
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Any, Optional, Union
import json
from icecream import ic
import io
import tempfile
import time
import logging
from llama_index.core import Document
import requests
from bs4 import BeautifulSoup

######################################################################
################### Test the APIs ####################################
######################################################################

SUPPORTED_EXTENSIONS = [
    ".docx", ".doc", ".odt", ".pptx", ".ppt", ".xlsx", ".csv", ".tsv", ".eml", ".msg",
    ".rtf", ".epub", ".html", ".xml", ".pdf", ".png", ".jpg", ".jpeg", ".txt"
]

# Patch the OpenAI client with Instructor
aclient = instructor.apatch(AsyncOpenAI(api_key = st.secrets["OPENAI_API_KEY"]))

# Define Pydantic model for response validation
class DocumentInfo(BaseModel):
    source_name: Union[str, Any]
    index: Optional[int]
    title: Union[str, Any]
    hashtags: Any
    hypothetical_questions: Any
    summary: Any

# Define a semaphore to limit the number of requests per minute
openai_request_limit = 10000
sem = asyncio.Semaphore(openai_request_limit) # 10000 requests per minute (OpenAI's limit) 
retry_decorator = retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=1, max=60))

#############################################
######### Better Metadata Generation ########
@retry_decorator
async def rate_limited_query_message_info_async(message_data: dict, sem: asyncio.Semaphore) -> DocumentInfo:
    async with sem:
        # print(f"Processing message: {str(message_data)}")
        class_response = await aclient.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            response_model=DocumentInfo,
            messages=[
                {"role": "system", "content": """
                    Generate the following schema for the document chunk:
                    0. source_name: str = File name or URL of the document
                    1. index: int = index number of the document chunk
                    2. title: str = Generate a title for the document chunk
                    3. hashtags: str = Use # to classify common categories
                    4. hypothetical_questions: List[str] = Relevant questions from the document
                    5. summary: str = Provide a detailed summary of the document
                    """},
                {"role": "user", "content": str(message_data)},
            ],
            max_retries=3,
        )
        json_response = class_response.model_dump()
        json_response['text_chunk'] = str(message_data).strip().replace('\n\n', '. ')
        return json_response

######################
######### PDF ########
@retry_decorator
async def post_pdf_upload(session, url, filepath, sem):
    data = aiohttp.FormData()
    data.add_field('file',
                   open(filepath, 'rb'),
                   filename=os.path.basename(filepath),
                   content_type='application/pdf')

    timeout = ClientTimeout(total=120)
    async with sem, session.post(url, data=data, timeout=timeout) as response:
        response_text = await response.text()
        response_dict = json.loads(response_text)
        filename = os.path.basename(filepath)
        if "extracted_text" in response_dict:
            extracted_texts = response_dict["extracted_text"]["0"]
        else:
            extracted_texts = ["Key 'extracted_text' not found in the response."]  # or handle this situation differently

        tasks = []
        count = 0
        for chunk in extracted_texts:
            post_chunk_content = f'filename: {filename}\n index: {count}\nchunk: {chunk}'
            task = asyncio.create_task(
                rate_limited_query_message_info_async(
                    post_chunk_content,
                    sem
                )
            )
            tasks.append(task)
            count += 1

        message_info = await asyncio.gather(*tasks)

        return message_info
        
async def process_pdf_file(files):
    if 'processed_pdf_files_metadata' not in st.session_state:
        st.session_state['processed_pdf_files_metadata'] = {}

    url = 'https://txtparseapis-azure.ashysky-c2a561fc.westus2.azurecontainerapps.io/process_upload'
    async with aiohttp.ClientSession() as session:
        tasks = []
        for index, file in enumerate(files, start=1):
            unique_file_name = f"{os.path.splitext(file.name)[0]}_{index}{os.path.splitext(file.name)[1]}"
            temp_file_path = os.path.join(tempfile.gettempdir(), unique_file_name)
            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(file.read())
                tasks.append(post_pdf_upload(session, url, temp_file.name, sem))

        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for response in responses:
            if isinstance(response, Exception):
                logging.error(f"Error occurred: {response}")
            else:
                for i, response_json in enumerate(response):
                    st.session_state['processed_pdf_files_metadata'][f"index_{i}_{response_json['source_name']}"] = response_json

        return st.session_state['processed_pdf_files_metadata']

# print(f"Time taken for processing two files: {two_files_time}")
# print(f"Time taken for processing all files: {all_files_time}")
# # Time taken for processing two files: 37.01477813720703
# # Time taken for processing four files: 54.508569955825806
# Depends on available resources, size of the files, and the number of files to process.

######################################################################
########## Image #####################################################
@retry_decorator
async def post_image_upload(session, url, filepath, sem):
    logging.info(f"Starting upload for {filepath}")
    data = aiohttp.FormData()
    data.add_field('file',
                   open(filepath, 'rb'),
                   filename=os.path.basename(filepath),
                   content_type='application/octet-stream')

    timeout = ClientTimeout(total=300)  # Increase the timeout to 120 seconds
    start_time = time.time()
    try:
        async with sem, session.post(url, data=data, timeout=timeout) as response:
            logging.info(f"Finished upload for {filepath}")
            response_text = await response.text()
            # use response_text for the rate_limited_query_message_info_async
            post_chunk_content = response_text
            message_info = await rate_limited_query_message_info_async(post_chunk_content, sem)

            return message_info  # return the structured message info
    except Exception as e:
        logging.error(f"Error occurred: {e}")
    finally:
        end_time = time.time()
        logging.info(f"Execution time: {end_time - start_time} seconds")

async def process_image_file(files):
    url = 'https://txtparseapis-azure.ashysky-c2a561fc.westus2.azurecontainerapps.io/process_upload'
    async with aiohttp.ClientSession() as session:
        tasks = []
        for index, file in enumerate(files, start=1):
            # Create a unique file name with the same base name as the uploaded file and the index as a unique identifier
            unique_file_name = f"{os.path.splitext(file.name)[0]}_{index}{os.path.splitext(file.name)[1]}"
            temp_file_path = os.path.join(tempfile.gettempdir(), unique_file_name)
            # Write the uploaded file's contents to the temporary file
            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(file.read())
                # Add a task to post the temporary file
                tasks.append(post_image_upload(session, url, temp_file.name, sem))
        responses = await asyncio.gather(*tasks)
        # ic(responses)
        # print(type(responses))
        for i, response_list in enumerate(responses):
            # store in session state
            # ic(response_list)
            st.session_state['processed_image_files_metadata'][f"index_{str(i)}_{str(response_list['source_name'])}"] = response_list
 
        return responses

######################################################################
########## Excel & CSV #####################################################
import pandas as pd
import numpy as np

# Corrected: Initialize processed files metadata as dictionaries
if 'processed_excel_files_metadata' not in st.session_state:
    st.session_state['processed_excel_files_metadata'] = {}

if 'processed_csv_files_metadata' not in st.session_state:
    st.session_state['processed_csv_files_metadata'] = {}

# Corrected: Processing of Excel files
async def process_excel_file(files):
    for file in files:
        file_name = file.name
        excel_file = pd.ExcelFile(file)
        sheet_names = excel_file.sheet_names
        sheet_data = []

        for sheet in sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet)
            rows = df.to_dict('index')
            sheet_data.append({
                'SheetName': sheet,
                'Rows': rows,
                'DataFrame': df
            })

        st.session_state['processed_excel_files_metadata'][file_name] = {
            'FileName': file_name,
            'Sheets': sheet_data
        }

    return st.session_state['processed_excel_files_metadata']

# Corrected: Processing of CSV files
async def process_csv_file(files):
    for file in files:
        file_name = file.name
        df = pd.read_csv(file)
        rows = df.to_dict('index')
        sheet_data = [{'SheetName': 'na', 'Rows': rows, 'DataFrame': df}]

        st.session_state['processed_csv_files_metadata'][file_name] = {
            'FileName': file_name,
            'Sheets': sheet_data
        }

    return st.session_state['processed_csv_files_metadata']


######################################################################
########## Other Files ###############################################
# For other file types, you can use the same pattern as the PDF processing
@retry_decorator
async def post_other_file_upload(session, url, filepath, sem):
    data = aiohttp.FormData()
    data.add_field('file',
                   open(filepath, 'rb'),
                   filename=os.path.basename(filepath),
                   content_type='application/octet-stream')

    timeout = ClientTimeout(total=120)  # Increase the timeout to 120 seconds
    async with session.post(url, data=data, timeout=timeout) as response:
        logging.info(f"Finished upload for {filepath}")
        response_text = await response.text()
        response_dict = json.loads(response_text)
        filename = os.path.basename(filepath)
        if "extracted_text" in response_dict:
            extracted_texts = response_dict["extracted_text"]["0"]
        else:
            extracted_texts = ["Key 'extracted_text' not found in the response."]  # or handle this situation differently

        tasks = []
        count = 0
        for chunk in extracted_texts:
            post_chunk_content = f'filename: {filename}\n\n index: {count}\n\n chunk: {chunk}'
            task = asyncio.create_task(
                rate_limited_query_message_info_async(
                    post_chunk_content,
                    sem
                )
            )
            tasks.append(task)
            count += 1

        message_info = await asyncio.gather(*tasks)

        return message_info  # return the structured message info

async def process_other_file(files):
    if 'processed_other_files_metadata' not in st.session_state:
        st.session_state['processed_other_files_metadata'] = {}

    url = 'https://txtparseapis-azure.ashysky-c2a561fc.westus2.azurecontainerapps.io/process_upload'
    async with aiohttp.ClientSession() as session:
        tasks = []
        for index, file in enumerate(files, start=1):
            unique_file_name = f"{os.path.splitext(file.name)[0]}_{index}{os.path.splitext(file.name)[1]}"
            temp_file_path = os.path.join(tempfile.gettempdir(), unique_file_name)
            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(file.read())
                tasks.append(post_other_file_upload(session, url, temp_file.name, sem))

        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for response in responses:
            if isinstance(response, Exception):
                logging.error(f"Error occurred: {response}")
            else:
                for i, response_json in enumerate(response):
                    st.session_state['processed_other_files_metadata'][f"index_{i}_{response_json['source_name']}"] = response_json

        return st.session_state['processed_other_files_metadata']

######################################################################
########## Function to run all above #############################################
async def run_all_file_processing(unprocessed_pdf_files_list, unprocessed_img_files_list, unprocessed_excel_files_list, unprocessed_csv_files_list, unprocessed_other_files_list):
    await asyncio.gather(
        process_pdf_file(unprocessed_pdf_files_list),
        process_image_file(unprocessed_img_files_list),
        process_excel_file(unprocessed_excel_files_list),
        process_csv_file(unprocessed_csv_files_list),
        process_other_file(unprocessed_other_files_list),
    )

######################################################################
########## File Uploader CLASS #############################################
class FileUploader:
    def __init__(self, supported_extensions):
        self.supported_extensions = supported_extensions
        if 'uploaded_files' not in st.session_state:
            st.session_state['uploaded_files'] = {}
        if 'selected_files' not in st.session_state:
            st.session_state['selected_files'] = []
        if 'processed_files_metadata' not in st.session_state:
            st.session_state['processed_pdf_files_metadata'] = {}
        if 'processed_image_files_metadata' not in st.session_state:
            st.session_state['processed_image_files_metadata'] = {}
        if 'processed_excel_files_metadata' not in st.session_state:
            st.session_state['processed_excel_files_metadata'] = {}
        if 'processed_csv_files_metadata' not in st.session_state:
            st.session_state['processed_csv_files_metadata'] = {}
        if 'processed_other_files_metadata' not in st.session_state:
            st.session_state['processed_other_files_metadata'] = {}
        if 'processed_html_files_metadata' not in st.session_state:
            st.session_state['processed_html_files_metadata'] = {}
        if 'llama_index_node_documents' not in st.session_state:
            st.session_state['llama_index_node_documents'] = {}
        if 'llama_parse_documents_list' not in st.session_state:
            st.session_state['llama_parse_documents_list'] = []

    @st.cache_data
    def extract_links_and_download_html(_self, url):
        # Improved sanitize_filename function to handle URLs ending with a slash
        def sanitize_filename(filename):
            if not filename:  # If filename is empty, provide a default name
                filename = "default_name"
            sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
            return sanitized.strip('_')

        logging.info("Extracting links from URL: %s", url)

        html_files = []
        file_counter = {}

        # Function to generate unique filename
        def generate_unique_filename(base_name):
            if base_name in file_counter:
                file_counter[base_name] += 1
            else:
                file_counter[base_name] = 0
            return f"{base_name}_{file_counter[base_name]}.html"

        # Adjust URL splitting logic to handle URLs ending with a slash
        url_parts = url.rstrip('/').split('/')  # Remove trailing slash if present
        base_name = sanitize_filename(url_parts[-1] if url_parts[-1] else url_parts[-2])

        # Download the main URL's HTML content
        try:
            response = requests.get(url)
            if response.status_code == 200:
                file_name = generate_unique_filename(base_name)
                temp_file_path = os.path.join(tempfile.gettempdir(), file_name)
                with open(temp_file_path, "w", encoding="utf-8") as f:
                    f.write(response.text)
                html_files.append(temp_file_path)
            else:
                logging.error(f"Failed to download {url}: {response.status_code}")
        except requests.RequestException as e:
            logging.error(f"Failed to download {url}: {e}")

        # The rest of the function remains unchanged

        logging.info("Downloaded HTML files: %s", html_files)
        return html_files


    async def process_html_files(self, grouped_html_files):
        url = 'https://txtparseapis-azure.ashysky-c2a561fc.westus2.azurecontainerapps.io/process_upload'
        sem = asyncio.Semaphore(10)  # Limit the number of concurrent requests
        async with aiohttp.ClientSession() as session:
            tasks = []
            for index, file in enumerate(grouped_html_files, start=1):
                tasks.append(post_other_file_upload(session, url, file, sem))
            responses = await asyncio.gather(*tasks)
            for response_list in responses:
                for i, response_json in enumerate(response_list):
                    key = f"index_{str(i)}_{str(response_json['source_name'])}"
                    st.session_state['processed_html_files_metadata'][key] = response_json
                    logging.debug(f"Added metadata for {key}: {response_json}")
        logging.info("Processed HTML files metadata: %s", st.session_state['processed_html_files_metadata'])
        return responses

    def upload_files(self):
        if 'processed_html_files_metadata' not in st.session_state:
            st.session_state['processed_html_files_metadata'] = {}

        from urllib.parse import urlparse

        def is_valid_url(url):
            try:
                result = urlparse(url)
                return all([result.scheme, result.netloc])
            except ValueError:
                return False

        def url_upload():
            url_input = st.text_input("Enter URL to scrape and process:")

            # Check if there's no new URL input
            if url_input == st.session_state.get('last_processed_url', ''):
                return  # No new input, return early
            
            if url_input:
                if not is_valid_url(url_input):
                    st.error("Invalid URL. Please enter a valid URL.")
                    return

                # Check if the URL has already been processed
                if 'grouped_html_files_by_url' in st.session_state and url_input in st.session_state['grouped_html_files_by_url']:
                    st.info(f"The URL {url_input} has already been processed.")
                    return

                start_time = time.time()
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                os.environ["LLAMA_CLOUD_API_KEY"] = st.secrets['LLAMA_CLOUD_API_KEY']
                os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
                logging.info("Starting URL scraping process...")
                
                st.toast(f'Starting URL scraping process for {url_input}...')
                grouped_html_files = self.extract_links_and_download_html(url_input)
                
                if not grouped_html_files:
                    st.error("No HTML files were downloaded. The website may restrict scraping.")
                    return

                logging.info("Running async process for HTML files...")
                loop.run_until_complete(self.process_html_files(grouped_html_files))
                logging.info("Finished processing HTML files.")

                # Group processed HTML files by URL link
                html_files_metadata = st.session_state['processed_html_files_metadata']
                grouped_by_url = {}
                for key, metadata in html_files_metadata.items():
                    url = metadata['source_name']  # Assuming 'source_name' is the URL
                    if url not in grouped_by_url:
                        grouped_by_url[url] = []
                    grouped_by_url[url].append(metadata)

                st.session_state['grouped_html_files_by_url'] = grouped_by_url

                # Display the selection options based on URL
                if grouped_by_url:
                    st.write("Select the processed HTML files based on URL:")
                    selected_urls = st.multiselect(
                        "Select URLs",
                        options=list(grouped_by_url.keys()),
                        default=[url_input] if url_input in grouped_by_url else []
                    )
                    # Collect all chunks from the selected URLs
                    selected_html_files = []
                    for url in selected_urls:
                        selected_html_files.extend(grouped_by_url[url])

                    st.session_state['selected_files'].extend(selected_html_files)
                    st.write(f"Selected files: {selected_urls}")
                else:
                    st.error("No processed HTML files found.")

                # Update last processed URL
                st.session_state['last_processed_url'] = url_input


        def files_upload(self):
            # Handle file uploads and set processed_bool
            uploaded_files = st.file_uploader("📥 Limit < 2000MB", type=SUPPORTED_EXTENSIONS, accept_multiple_files=True)

            # Generate a set of uploaded file names
            current_uploaded_file_names = {file.name for file in uploaded_files} if uploaded_files else set()

            # Check if there's no new file upload
            if current_uploaded_file_names == st.session_state.get('last_uploaded_file_names', set()):
                return  # No new uploads, return early
    

            if uploaded_files:
                start_time = time.time()
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                os.environ["LLAMA_CLOUD_API_KEY"] = st.secrets['LLAMA_CLOUD_API_KEY']
                os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

                from llama_parse import LlamaParse

                for index, file in enumerate(uploaded_files, start=1):
                    file_base_name = os.path.basename(file.name)
                    file_short_name = f"{index}_{file_base_name if len(file_base_name) <= 20 else file_base_name[:10] + '...' + file_base_name[-10:]}"
                    file_category = None

                    if file.name.endswith(".pdf"):
                        file_category = 'pdf'
                    elif file.name.endswith(".png") or file.name.endswith(".jpg"):
                        file_category = 'img'
                    elif file.name.endswith(".xlsx"):
                        file_category = 'excel'
                    elif file.name.endswith(".csv"):
                        file_category = 'csv'
                    else:
                        file_category = 'others'

                    if file_category:
                        if file_category not in st.session_state['uploaded_files']:
                            st.session_state['uploaded_files'][file_category] = {}
                        st.session_state['uploaded_files'][file_category][file.name] = {'file': file, 'processed_bool': False, 'file_short_name': file_short_name}

                        if file_category == 'pdf' and st.session_state['llama_parse_mode'] == 'True':
                            llama_parse_documents = LlamaParse(result_type="markdown").load_data(file_path=file)
                            st.session_state['llama_parse_documents_list'].append(llama_parse_documents)

                unprocessed_files = {category: [file_info['file'] for file_info in st.session_state['uploaded_files'].get(category, {}).values() if not file_info['processed_bool']] for category in ['pdf', 'img', 'excel', 'csv', 'others']}
                
                loop.run_until_complete(run_all_file_processing(unprocessed_files['pdf'], unprocessed_files['img'], unprocessed_files['excel'], unprocessed_files['csv'], unprocessed_files['others']))

                for category in unprocessed_files:
                    for file in unprocessed_files[category]:
                        st.session_state['uploaded_files'][category][file.name]['processed_bool'] = True

                total_unprocessed = sum(len(files) for files in unprocessed_files.values())
                if total_unprocessed > 0:
                    st.success(f"Time taken for processing {total_unprocessed} new files: {time.time() - start_time} seconds")
                total_processed = sum(len(files) for files in st.session_state['uploaded_files'].values() if files)
                st.success(f"Processed {total_processed} files. Processed File's Names: {format({file_info['file_short_name'] for files in st.session_state['uploaded_files'].values() for file_info in files.values()})}")

            # Update last uploaded file names
            st.session_state['last_uploaded_file_names'] = current_uploaded_file_names

        # File upload and selection section
        with st.expander("📁 File Upload and Selection:", expanded=True):
            url_upload()
            files_upload(self)

            # Process PDF files
            for key, value in st.session_state['processed_pdf_files_metadata'].items():
                index = value['index']
                source_name = value['source_name']
                jointed_text_for_node = str(value)
                if source_name not in st.session_state['llama_index_node_documents']:
                    st.session_state['llama_index_node_documents'][source_name] = {}
                st.session_state['llama_index_node_documents'][source_name][index] = Document(text=jointed_text_for_node)

            # Process image files
            for key, value in st.session_state['processed_image_files_metadata'].items():
                index = value['index']
                source_name = value['source_name']
                jointed_text_for_node = str(value)
                if source_name not in st.session_state['llama_index_node_documents']:
                    st.session_state['llama_index_node_documents'][source_name] = {}
                st.session_state['llama_index_node_documents'][source_name][index] = Document(text=jointed_text_for_node)

            # Process Excel files
            for file_name, file_data in st.session_state['processed_excel_files_metadata'].items():
                # logging.debug(f"Processing Excel file: {file_name}")
                for sheet in file_data['Sheets']:
                    sheet_name = sheet['SheetName']
                    rows = sheet['Rows']
                    for index, record in rows.items():
                        record_texts = [f"{k}: {v}" for k, v in record.items()]
                        joint_text = '. '.join(record_texts)
                        jointed_text_for_node = {
                            'index': index,
                            'source_name': f"{file_name}_{sheet_name}",
                            'text_chunk': joint_text,
                        }
                        source_name = f"{file_name}_{sheet_name}"
                        if source_name not in st.session_state['llama_index_node_documents']:
                            st.session_state['llama_index_node_documents'][source_name] = {}
                        st.session_state['llama_index_node_documents'][source_name][index] = Document(text=str(jointed_text_for_node))

            # Process CSV files
            for file_name, file_data in st.session_state['processed_csv_files_metadata'].items():
                # logging.debug(f"Processing CSV file: {file_name}")
                for sheet in file_data['Sheets']:
                    sheet_name = sheet['SheetName']
                    rows = sheet['Rows']
                    for index, record in rows.items():
                        record_texts = [f"{k}: {v}" for k, v in record.items()]
                        joint_text = '. '.join(record_texts)
                        jointed_text_for_node = {
                            'index': index,
                            'source_name': f"{file_name}_{sheet_name}",
                            'text_chunk': joint_text,
                        }
                        source_name = f"{file_name}_{sheet_name}"
                        if source_name not in st.session_state['llama_index_node_documents']:
                            st.session_state['llama_index_node_documents'][source_name] = {}
                        st.session_state['llama_index_node_documents'][source_name][index] = Document(text=str(jointed_text_for_node))


            # Process other files
            for key, value in st.session_state['processed_other_files_metadata'].items():
                index = value['index']
                source_name = value['source_name']
                jointed_text_for_node = str(value)
                unique_key = f"{source_name}_{index}"
                if unique_key not in st.session_state['llama_index_node_documents']:
                    st.session_state['llama_index_node_documents'][unique_key] = Document(text=jointed_text_for_node)

            # Process HTML files
            grouped_by_url = st.session_state.get('grouped_html_files_by_url', {})
            for url, file_metadatas in grouped_by_url.items():
                logging.debug(f"Processing URL: {url} with {len(file_metadatas)} files")
                for file_metadata in file_metadatas:
                    index = file_metadata['index']
                    source_name = file_metadata['source_name']
                    jointed_text_for_node = str(file_metadata)
                    unique_key = f"{source_name}_{index}"
                    logging.debug(f"Generated unique key: {unique_key} for file metadata: {file_metadata}")
                    if unique_key not in st.session_state['llama_index_node_documents']:
                        st.session_state['llama_index_node_documents'][unique_key] = Document(text=jointed_text_for_node)
                        logging.info(f"Added document with key {unique_key} to llama_index_node_documents")



            # st.write(st.session_state['llama_index_node_documents'].items())

            # Combine all documents (including HTML) into the selection UI
            all_documents = []
            short_key_and_documents_for_selected_files = {}
            for key, value in st.session_state['llama_index_node_documents'].items():
                short_key = f"{key if len(key) <= 20 else key[:10] + '...' + key[-10:]}"
                # Since value is a Document, add it directly to the lists
                all_documents.append(value)
                short_key_and_documents_for_selected_files[short_key] = value

            # Select Files
            st.markdown("🤌 File Selection")
            all_selected_files = sac.chip(
                items=[sac.ChipItem(label="All Files")],
                radius='md',
                multiple=True,
                key="all_files_chip"  # Unique key for this widget
            )                

            # st.write(short_key_and_documents_for_selected_files)

            selected_files = sac.chip(
                items=[
                    sac.ChipItem(label=short_key) 
                    for short_key in short_key_and_documents_for_selected_files.keys()
                ],
                radius='md',
                multiple=True,
                key="selected_files_chip"  # Unique key for this widget
            )


            st.write("📝 Selected files")

            if "All Files" in all_selected_files:
                st.session_state['selected_files'] = list(st.session_state['llama_index_node_documents'].values())
                st.write("Selected: All Files")

            if "All Files" not in all_selected_files:
                selected_documents = [short_key_and_documents_for_selected_files[sk] for sk in selected_files]
                st.session_state['selected_files'] = selected_documents

            st.info("Select All Files to index all documents. Unselect All Files & Select individual files to index only those files.")


            if st.button("Reset Files"):
                st.session_state['uploaded_files'] = {}
                st.session_state['selected_files'] = []
                st.session_state['processed_pdf_files_metadata'] = {}
                st.session_state['processed_image_files_metadata'] = {}
                st.session_state['processed_excel_files_metadata'] = {}
                st.session_state['processed_csv_files_metadata'] = {}
                st.session_state['processed_other_files_metadata'] = {}
                st.session_state['processed_html_files_metadata'] = {}
                st.session_state['llama_index_node_documents'] = {}
                st.session_state['llama_parse_documents_list'] = []
                st.rerun()

            return st.session_state['uploaded_files'], st.session_state['selected_files'], st.session_state['llama_parse_documents_list']
