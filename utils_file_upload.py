import streamlit as st
import streamlit_antd_components as sac
import os
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
                tasks.append(post_pdf_upload(session, url, temp_file.name, sem))
        responses = await asyncio.gather(*tasks)
        # ic(responses)
        for response_list in responses:
            for i, response_json in enumerate(response_list):
                # store in session state
                st.session_state['processed_pdf_files_metadata'][f"index_{str(response_json['index'])}_{str(response_json['source_name'])}_{i}"] = response_json
        return responses

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

async def process_excel_file(files):
    for file in files:
        excel_file = pd.ExcelFile(file)
        sheet_names = excel_file.sheet_names
        sheet_data = []

        for sheet in sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet)
            # Transform the DataFrame to a dictionary, where each key-value pair represents a row
            rows = df.to_dict('index')
            sheet_data.append({
                'SheetName': sheet,
                'Rows': rows,
                'DataFrame': df  # Store the DataFrame in the session state
            })

        st.session_state['processed_excel_files_metadata'].append({
            'FileName': file.name,
            'Sheets': sheet_data
        })

    return st.session_state['processed_excel_files_metadata']

async def process_csv_file(files):
    for file in files:
        df = pd.read_csv(file)
        # Transform the DataFrame to a dictionary, where each key-value pair represents a row
        rows = df.to_dict('index')
        sheet_data = [{'SheetName': 'na', 'Rows': rows, 'DataFrame': df}]  # Store the DataFrame in the session state

        st.session_state['processed_csv_files_metadata'].append({
            'FileName': file.name,
            'Sheets': sheet_data
        })

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
                tasks.append(post_other_file_upload(session, url, temp_file.name, sem))
        responses = await asyncio.gather(*tasks)
        # ic(responses)
        for response_list in responses:
            for i, response_json in enumerate(response_list):
                # store in session state
                st.session_state['processed_other_files_metadata'][f"index_{str(i)}_{str(response_json['source_name'])}"] = response_json
        return responses

######################################################################
########## Function to run all above #############################################
async def run_all_file_processing(unprocessed_pdf_files_list, unprocessed_img_files_list, unprocessed_excel_files_list, unprocessed_csv_files_list, unprocessed_other_files_list):
    await asyncio.gather(
        process_pdf_file(unprocessed_pdf_files_list),
        process_image_file(unprocessed_img_files_list),
        process_excel_file(unprocessed_excel_files_list),
        process_csv_file(unprocessed_csv_files_list),
        process_other_file(unprocessed_other_files_list)
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


    def upload_files(self):
        if 'processed_html_files_metadata' not in st.session_state:
            st.session_state['processed_html_files_metadata'] = {}

        with st.expander("📁 File Upload and Selection:"):
            url_input = st.text_input("Enter URL to scrape and process:")
            if url_input:
                start_time = time.time()
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                os.environ["LLAMA_CLOUD_API_KEY"] = st.secrets['LLAMA_CLOUD_API_KEY']
                os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

                from llama_parse import LlamaParse

                if url_input:
                    grouped_html_files = self.extract_links_and_download_html(url_input)
                    loop.run_until_complete(self.process_html_files(grouped_html_files))

            uploaded_files = st.file_uploader("📥 Limit < 2000MB", type=self.supported_extensions, accept_multiple_files=True)
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
                    if file.name.endswith(".pdf"):
                        if 'pdf' not in st.session_state['uploaded_files']:
                            st.session_state['uploaded_files']['pdf'] = {}
                        if file.name not in st.session_state['uploaded_files']['pdf']:
                            st.session_state['uploaded_files']['pdf'][file.name] = {'file': file, 'processed_bool': False, 'file_short_name': file_short_name}
                            if st.session_state['llama_parse_mode'] == 'True':
                                llama_parse_documents = LlamaParse(result_type="markdown").load_data(file_path=file)
                                st.session_state['llama_parse_documents_list'].append(llama_parse_documents)
                    elif file.name.endswith(".png") or file.name.endswith(".jpg"):
                        if 'img' not in st.session_state['uploaded_files']:
                            st.session_state['uploaded_files']['img'] = {}
                        if file.name not in st.session_state['uploaded_files']['img']:
                            st.session_state['uploaded_files']['img'][file.name] = {'file': file, 'processed_bool': False, 'file_short_name': file_short_name}
                    elif file.name.endswith(".xlsx"):
                        if 'excel' not in st.session_state['uploaded_files']:
                            st.session_state['uploaded_files']['excel'] = {}
                        if file.name not in st.session_state['uploaded_files']['excel']:
                            st.session_state['uploaded_files']['excel'][file.name] = {'file': file, 'processed_bool': False, 'file_short_name': file_short_name}
                    elif file.name.endswith(".csv"):
                        if 'excel' not in st.session_state['uploaded_files']:
                            st.session_state['uploaded_files']['csv'] = {}
                        if file.name not in st.session_state['uploaded_files']['excel']:
                            st.session_state['uploaded_files']['csv'][file.name] = {'file': file, 'processed_bool': False, 'file_short_name': file_short_name}
                    else:
                        if 'others' not in st.session_state['uploaded_files']:
                            st.session_state['uploaded_files']['others'] = {}
                        if file.name not in st.session_state['uploaded_files']['others']:
                            st.session_state['uploaded_files']['others'][file.name] = {'file': file, 'processed_bool': False, 'file_short_name': file_short_name}

                unprocessed_pdf_files_list = [file_info['file'] for file_info in st.session_state['uploaded_files'].get('pdf', {}).values() if not file_info['processed_bool']]
                unprocessed_img_files_list = [file_info['file'] for file_info in st.session_state['uploaded_files'].get('img', {}).values() if not file_info['processed_bool']]
                unprocessed_excel_files_list = [file_info['file'] for file_info in st.session_state['uploaded_files'].get('excel', {}).values() if not file_info['processed_bool']]
                unprocessed_csv_files_list = [file_info['file'] for file_info in st.session_state['uploaded_files'].get('csv', {}).values() if not file_info['processed_bool']]
                unprocessed_other_files_list = [file_info['file'] for file_info in st.session_state['uploaded_files'].get('others', {}).values() if not file_info['processed_bool']]

                loop.run_until_complete(run_all_file_processing(unprocessed_pdf_files_list, unprocessed_img_files_list, unprocessed_excel_files_list, unprocessed_csv_files_list, unprocessed_other_files_list))

                for file in unprocessed_pdf_files_list:
                    st.session_state['uploaded_files']['pdf'][file.name]['processed_bool'] = True

                for file in unprocessed_img_files_list:
                    st.session_state['uploaded_files']['img'][file.name]['processed_bool'] = True

                for file in unprocessed_excel_files_list:
                    st.session_state['uploaded_files']['excel'][file.name]['processed_bool'] = True

                for file in unprocessed_csv_files_list:
                    st.session_state['uploaded_files']['csv'][file.name]['processed_bool'] = True

                for file in unprocessed_other_files_list:
                    st.session_state['uploaded_files']['others'][file.name]['processed_bool'] = True

                len_of_unprocessed_files = len(unprocessed_pdf_files_list)+len(unprocessed_img_files_list)+len(unprocessed_excel_files_list)+len(unprocessed_csv_files_list)+len(unprocessed_other_files_list)
                if len_of_unprocessed_files > 0:
                    st.success(f"Time taken for processing {len_of_unprocessed_files} new files: {time.time() - start_time} seconds")
                count_processed_files = sum([len(category_files) for category_files in st.session_state['uploaded_files'].values() if category_files])
                st.success(f"Processed {count_processed_files} files. Processed File's Names: {format({file_info['file_short_name'] for category_files in st.session_state['uploaded_files'].values() for file_info in category_files.values()})}")

            for key, value in st.session_state['processed_pdf_files_metadata'].items():
                index = value['index']
                source_name = value['source_name']

                jointed_text_for_node = str(value)
                if source_name not in st.session_state['llama_index_node_documents']:
                    st.session_state['llama_index_node_documents'][source_name] = {}
                st.session_state['llama_index_node_documents'][source_name][index] = Document(text=jointed_text_for_node)

            for key, value in st.session_state['processed_image_files_metadata'].items():
                index = value['index']
                source_name = value['source_name']

                jointed_text_for_node = str(value)
                if source_name not in st.session_state['llama_index_node_documents']:
                    st.session_state['llama_index_node_documents'][source_name] = {}
                st.session_state['llama_index_node_documents'][source_name][index] = Document(text=jointed_text_for_node)

            for file in st.session_state['processed_excel_files_metadata']:
                file_name = file['FileName']
                for sheet in file['Sheets']:
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

            for file in st.session_state['processed_csv_files_metadata']:
                file_name = file['FileName']
                for sheet in file['Sheets']:
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

            for key, value in st.session_state['processed_other_files_metadata'].items():
                index = value['index']
                source_name = value['source_name']

                jointed_text_for_node = str(value)
                unique_key = f"{source_name}_{index}"
                if unique_key not in st.session_state['llama_index_node_documents']:
                    st.session_state['llama_index_node_documents'][unique_key] = Document(text=jointed_text_for_node)

            for url, files_metadata in st.session_state['processed_html_files_metadata'].items():
                for value in files_metadata:
                    index = value['index']
                    source_name = value['source_name']

                    jointed_text_for_node = str(value)
                    unique_key = f"{source_name}_{index}"
                    if unique_key not in st.session_state['llama_index_node_documents']:
                        st.session_state['llama_index_node_documents'][unique_key] = Document(text=jointed_text_for_node)

            all_documents = []
            short_key_and_documents_for_selected_files = {}
            for key, value in st.session_state['llama_index_node_documents'].items():
                file_extension = os.path.splitext(key)[1]
                short_key = f"{key if len(key) <= 20 else key[:10] + '...' + key[-10:]}"
                all_documents.append(value)
                short_key_and_documents_for_selected_files[short_key] = value

            st.markdown("🤌 File Selection")
            all_selected_files = sac.chip(
                items = [sac.ChipItem(label="All Files")],
                radius='md',
                multiple=True
            )                

            selected_files = sac.chip(
                items = [
                    sac.ChipItem(label=short_key) 
                    for short_key in short_key_and_documents_for_selected_files.keys()
                ],
                radius='md',
                multiple=True
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



    @st.cache_data
    def extract_links_and_download_html(_self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        links = soup.find_all("a", href=True)
        
        html_files = []
        for link in links:
            href = link['href']
            if href.startswith("http"):  # Ensure it's a full URL
                response = requests.get(href)
                if response.status_code == 200:
                    file_name = href.split("/")[-1] + ".html"
                    temp_file_path = os.path.join(tempfile.gettempdir(), file_name)
                    with open(temp_file_path, "w", encoding="utf-8") as f:
                        f.write(response.text)
                    html_files.append(temp_file_path)
        return html_files

    async def process_html_files(self, files):
        url = 'https://txtparseapis-azure.ashysky-c2a561fc.westus2.azurecontainerapps.io/process_upload'
        async with aiohttp.ClientSession() as session:
            tasks = []
            for file in files:
                tasks.append(post_other_file_upload(session, url, file, sem))
            responses = await asyncio.gather(*tasks)
            for response_list in responses:
                for i, response_json in enumerate(response_list):
                    st.session_state['processed_html_files_metadata'][f"index_{str(i)}_{str(response_json['source_name'])}"] = response_json
        return responses


    @st.cache_data
    def download_html(_self, url):
        response = requests.get(url)
        if response.status_code == 200:
            file_name = url.split("/")[-1] + ".html"
            temp_file_path = os.path.join(tempfile.gettempdir(), file_name)
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            return [temp_file_path]
        return []

    def reset_all_files(self):
        st.session_state['uploaded_files'] = {}
        st.session_state['selected_files'] = []
        st.session_state['processed_pdf_files_metadata'] = {}
        st.session_state['processed_image_files_metadata'] = {}
        st.session_state['processed_excel_files_metadata'] = {}
        st.session_state['processed_csv_files_metadata'] = {}
        st.session_state['processed_other_files_metadata'] = {}
        st.session_state['llama_index_node_documents'] = {}
        st.session_state['llama_parse_documents_list'] = []
