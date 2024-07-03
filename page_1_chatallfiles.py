import os
import sys
import json
import ast
import re
import time
import logging
import shutil
from typing import Any, List
import asyncio
import aiohttp

import pandas as pd
import streamlit as st
import streamlit_antd_components as sac
from streamlit_searchbox import st_searchbox

import cohere
import openai
from icecream import ic

from utils_file_upload import FileUploader
from utils_parsely_core import process_in_default_mode

from llama_index.core import (
    SimpleDirectoryReader,
    ServiceContext,
    Document,
    VectorStoreIndex,
    Settings,
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
# from llama_index.embeddings.huggingface_optimum import OptimumEmbedding
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.indices.managed.vectara import VectaraIndex

from llama_parse import LlamaParse


logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

azure_endpoint=st.secrets["AOAIEndpoint"]
api_key=st.secrets["AOAIKey"]
api_version="2024-02-15-preview"

llm = AzureOpenAI(
    model="gpt-4o",
    deployment_name="gpt-4o",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

# You need to deploy your own embedding model as well as your own chat completion model
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-3-small",
    deployment_name="text-embedding-3-small",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

Settings.llm = llm
Settings.embed_model = embed_model

os.environ['VECTARA_API_KEY'] = 'zwt_BSY6BZDNszBvS4OIN3S1667IxZ2FntdvwWLY-A'
os.environ['VECTARA_CORPUS_ID'] = 'ragathon'
os.environ['VECTARA_CUSTOMER_ID'] = '86391301'

# Create Vectara Index
vectara_index = VectaraIndex()

OpenAI.api_key = st.secrets["OPENAI_API_KEY"]


SUPPORTED_EXTENSIONS = [
    ".docx", ".doc", ".odt", ".pptx", ".ppt", ".xlsx", ".csv", ".tsv", ".eml", ".msg",
    ".rtf", ".epub", ".html", ".xml", ".pdf", ".png", ".jpg", ".jpeg", ".txt"
]


# if not os.path.exists("./bge_onnx"):
#     OptimumEmbedding.create_and_save_optimum_model(
#         "BAAI/bge-base-en-v1.5", "./bge_onnx"
#     )

# embed_model = OptimumEmbedding(folder_name="./bge_onnx", embed_batch_size=100)


def chatallfiles_page():

    ##############################################################################################################
    ##### OpenAI Chatbot #####
    ##############################################################################################################

    from openai import OpenAI as OG_OpenAI

    def process_in_default_mode(user_question):
        main_full_response = ""
        client = OG_OpenAI()

        message_placeholder = st.empty()
        responses = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content":   """
                            Objective:
                            Think Carefully: Consider whether to respond concisely and directly for simple questions, or to provide a more detailed explanation for complex questions with Prompt Below:
                            ALWAYS PRIORITIZE CONCISE AND SIMPLE RESPONSE

                            Detailed Prompt:                        
                            As an experimental Homen's biography writer, you know everything about Homen's life and work. Your goal is to respond on behalf of Homen like a therapist.

                            Utilize a diverse array of knowledge spanning numerous fields including nutrition, sports science, finance, economics, globalization policies, accounting, technology management, high-frequency trading, machine learning, data science, human psychology, investor psychology, and principles from influential thinkers and top business schools. The goal is to simplify complex topics from any domain into an easily understandable format.

                            Rephrase for Clarity: Begin by rephrasing the query to confirm understanding, using the approach, "So, what I'm hearing is...". This ensures that the response is precisely tailored to the query's intent.

                            Concise Initial Response: Provide an immediate, succinct answer, blending insights from various areas of expertise. 

                            List of Key Topics: Identify the top three relevant topics, such as 'Technological Innovation', 'Economic and Market Analysis', or 'Scientific Principles in Everyday Applications'. This step will frame the subsequent detailed explanation.

                            Detailed Explanation and Insights: Expand on the initial response with insights from various fields, using metaphors and simple language, to make the information relatable and easy to understand for everyone.

                            -----------------

                            Response Format:

                                **Question Summary**
                                **Key Topics**
                                **Answer**
                                **Source of Evidence**
                                **Detailed Explanation and Insights**
                                **Confidence **

                            **Question Summary** Description: 
                                Restate the query for clarity and confirmation.
                            Example: 
                                "So, what I'm hearing is, you're asking about..."

                            **Key Topics** Description: 
                                List the top three relevant topics that will frame the detailed explanation.
                            Example: 
                                "1. Economic Impact, 2. Technological Advancements, 3. Strategic Decision-Making."

                            **Answer** Description: 
                                Provide a succinct, direct response and an introspecting question to the rephrased query. Speak about the reason why in terms of opportunity cost, fear and hope driven human psychology, percentage probabilities of desired outcomes, and question back to user to let them introspect. 
                            Example: 
                                "The immediate solution to this issue would be... Now here is a question that I have for you... The reason why I ask this question is because..." 

                            **Source of Evidence** Description:
                                Quote the most relevant part of the search result that answers the user's question.

                            **Detailed Explanation and Insights** Description: 
                                Expand on the Quick Respond with insights from various fields, using metaphors and simple language. List out specific people and examples.
                            Example: 
                                "Drawing from economic theory, particularly the concepts championed by Buffett and Munger, we can understand that..."

                            **Confidence ** Description:
                                Confidence  of the response, low medium high.
                            -----------------

                            Example Output:
                                For a query about investment strategies, the response would start with a rephrased question to ensure understanding, followed by a concise answer about fundamental investment principles. The key topics might include 'Market Analysis', 'Risk Management', and 'Long-term Investment Strategies'. The detailed explanation would weave in insights from finance, economics, and successful investors' strategies, all presented in an easy-to-grasp manner. If applicable, specific investment recommendations or insights would be provided, along with a rationale and confidence . The use of simple metaphors and analogies would make the information relatable and easy to understand for everyone.
                            """},
                {"role": "user", "content": f"User Input: {user_question}"}
            ],
            stream=True,
            seed=42,
        )
        
        for response in responses:
            if response.choices[0].delta.content:
                main_full_response += str(response.choices[0].delta.content).encode('utf-8').decode('utf-8').replace('$', '\$')
                message_placeholder.markdown(f"**One Time Response**: {main_full_response}")
        st.markdown("---")
        return main_full_response

    async def files_bm25_search(query):
        # BM25 search
        nodes = await information_retriever.aretrieve(query)
        output_data = []

        for node in nodes:
            output = {
                "Query": query,
                "Details": node.text,
                "Confidence ": node.score,
            }

            output_data.append(output)
        # ic(output_data)
        return output_data

    api_key = st.secrets["openai_api_key"]

    async def process_files_data(prompt, search_files_result):
        system_prompt = """
            Response Structure:
            Please use the appropriate JSON schema below for your response.

            json
            {
            "User Question": "string",
            "Key Words": "string",
            "Response": "string",
            "Source of Evidence": "string",
            }
            User Question: Quote the user's question, no changes.
            Key Words: Summarize the main topics that the answer covers with Keywords.
            Response: Provide answer, focus on the quantitative and qualitative aspects of the answer.
            Source of Evidence: Quote the most relevant part of the search result that answers the user's question.
        """

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

        async with aiohttp.ClientSession() as session:
            # Create a list to store the tasks
            tasks = []

            # Create a task for each highlighted_content
            for result in search_files_result:
                payload = {
                    "model": "gpt-3.5-turbo-0125",
                    "response_format": { "type": "json_object" },
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"User Needs Input: {prompt}, FILES search result: {result}"},
                            ]
                        },
                    ],
                    "seed": 42,
                }
                task = session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                tasks.append(task)

            responses = await asyncio.gather(*tasks)  # Await the tasks here

            output = []
            for response in responses:
                response_json = await response.json()
                output.append(response_json['choices'][0]['message']['content'])                              
            # ic(output)
            return output
                

    def process_in_files_mode(user_question):
        main_full_response = ""
        client = OG_OpenAI()

        # Generate user needs from the question
        user_needs = ""
        first_response_message_placeholder = st.empty()

        responses = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": f"This is process_in_files_mode. Rephrase the User Input. Extrapolate the user needs in a key topic list. Begin with 'Here is what I am understanding from your question...'"},
                {"role": "user", "content": f"User Input: {user_question}"}
            ],
            stream=True,
            seed=42,
        )
        for response in responses:
            # if not None
            if response.choices[0].delta.content:
                user_needs += str(response.choices[0].delta.content).replace('$', '\$')
                first_response_message_placeholder.markdown(f"**User Needs**: {user_needs}")
        st.markdown("---")
        st.session_state.main_conversation.append({"role": "Assistant", "content": f"""**User Needs**: 
                                                                                        {user_needs}"""})
        
        main_full_response += user_needs

        # Search in FILESs
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # st.toast(f"Searching in FILES")
            search_files_output_data_list = loop.run_until_complete(files_bm25_search(query=user_needs))
            ##### Dense: Cohere Rerank #####
            co = cohere.Client(st.secrets["COHERE_API_KEY"])
            rerank_search_files_output_data_list = [{'text': str({"result": result}).replace('$', '\$')} for result in search_files_output_data_list]
            rerank_search_files_output_data_list_results = co.rerank(model="rerank-english-v2.0",
                                    query=user_needs,
                                    documents=rerank_search_files_output_data_list,
                                    top_n=10)
            ##### Dense: Cohere Rerank #####
            processed_data = loop.run_until_complete(process_files_data(user_needs, rerank_search_files_output_data_list_results))
            main_full_response += "\n" + str(processed_data)
        finally:
            loop.close()

        # ic(main_full_response)

        # Final Response with GPT-4-0125-Preview

        final_response = ""
        second_response_message_placeholder = st.empty()
        responses = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "system", "content": f"Highly detailed **Answer**, rest of the response should be concise, unless user asks for more details. Make sure to rephrase the User Input. Extrapolate the user needs in a key topic list. Utilize FILES Search Result. Begin with '**Question Summary**, **Key Topics**, **Answer**, **Source of Evidence**, **Confidence ** (low medium high)...'"},
                {"role": "user", "content": f"User Input: {user_question}, User Needs: {user_needs}, FILES Search Result: {processed_data}, Main Full Response: {main_full_response}"}
            ],
            stream=True,
            seed=42,
        )
        for response in responses:
            # if not None
            if response.choices[0].delta.content:
                final_response += str(response.choices[0].delta.content).encode('utf-8').decode('utf-8').replace('$', '\$')
                second_response_message_placeholder.markdown(f"**Final Response**: {final_response}")
        st.markdown("---")
        return final_response

    ### Title and Description
    st.title("💬 Chat for All Files")
    st.toast("🌱 Hi! I’m Parsely and I’m here to assist your document analysis needs. Type below to get started.")

    ### FileUploader from utils.py
    @st.experimental_fragment    
    def file_uploader_fragment():
        uploader = FileUploader(SUPPORTED_EXTENSIONS)
        uploader.upload_files()

    file_uploader_fragment()
    st.toast(f'📁 {len(st.session_state["selected_files"])} files uploaded')

    ### Conversation Memory and Summary
    if "main_conversation" not in st.session_state:
        st.session_state["main_conversation"] = []

    if "main_conversation_memory" not in st.session_state:
        st.session_state["main_conversation_memory"] = []
        st.session_state["main_conversation_memory"].append("")

    ### Llama_Parse Mode
    llama_parse_mode = sac.switch(label='Llama_Parse Mode (Only Indexes File if On before Uploadfile)', align='start', size='md')
    st.session_state["llama_parse_mode"] = llama_parse_mode
    # # ic(st.session_state["llama_parse_mode"])

    ### Vectara Query Mode
    vectara_query_mode = sac.switch(label='Vectara Query Mode', align='start', size='md')
    st.session_state["vectara_query_mode"] = vectara_query_mode
    # # ic(st.session_state["vectara_query_mode"])

    ### Reset Conversation
    if st.button("Reset Conversation"):
        st.session_state["main_conversation"] = []
        st.session_state["main_conversation_memory"] = []
        st.session_state["main_conversation_memory"].append("")

    ### Display previous conversation messages
    if st.session_state["main_conversation"]:
        for message in st.session_state.main_conversation:
            if message["role"] == "User":
                with st.chat_message("User"):
                    st.markdown("User: ")
                    st.markdown(message["content"])
            elif message["role"] == "Assistant":
                with st.chat_message("Assistant"):
                    st.markdown("Assistant: ")
                    st.markdown(message["content"])

    ### Chat Interface
    user_question = st.chat_input("Ask a question about the selected PDF content:")

    if user_question:
        with st.chat_message("User"):
            st.markdown("User: ")
            st.markdown(user_question)
            st.session_state.main_conversation.append({"role": "User", "content": user_question})

        with st.chat_message("Assistant"):
            if not st.session_state['selected_files']:
                # st.toast(f"No files selected, chat in default mode")
                default_response_with_custom_prompt = process_in_default_mode(user_question)
                st.session_state.main_conversation.append({"role": "Assistant", "content": default_response_with_custom_prompt})
            else:
                # st.toast(f"Files selected, chat in files retrieval mode")
                for _ in range(3):  # Retry up to 3 times
                    try:
                        llama_index_node_documents = []
                        documents_referred = []
                        # page_1_st_session_state_selected_files = st.session_state['selected_files']
                        # # ic(page_1_st_session_state_selected_files)
                        for document_obj in st.session_state['selected_files']:
                            # Assuming 'selected_files' is directly storing Document objects
                            # if document_obj has items
                            if isinstance(document_obj, dict):
                                for id, doc in document_obj.items():
                                    # ic(doc)
                                    llama_index_node_documents.append(doc)  # No need to recreate Document(text=doc.text)
                                    if vectara_query_mode:
                                        vectara_index.insert_file(file_path= "", metadata= {id: doc})

                                    try:
                                        # ic(doc)
                                        parsed_doc = ast.literal_eval(doc.text)
                                        documents_referred.append(f"{parsed_doc['source_name']} index: {parsed_doc['index']}")
                                    except ValueError as e:
                                        st.toast(f"Error parsing document text: {e}")
                            else:
                                # just use the document_obj 
                                llama_index_node_documents.append(document_obj)
                                try:
                                    parsed_doc = ast.literal_eval(document_obj.text)
                                    documents_referred.append(f"{parsed_doc['source_name']} index: {parsed_doc['index']}")
                                except ValueError as e:
                                    st.toast(f"Error parsing document text: {e}")
                                    
                        # ic(documents_referred)
                        service_context = ServiceContext.from_defaults(embed_model=embed_model)
                        nodes = service_context.node_parser.get_nodes_from_documents(llama_index_node_documents)
                        similarity_top_k_value = len(llama_index_node_documents) // 2 // 2
                        similarity_top_k_value = max(10, similarity_top_k_value)
                        
                        information_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=similarity_top_k_value)

                        break
                    except ValueError as e:
                        if str(e) == "Please pass exactly one of index, nodes, or docstore.":
                            st.toast("Please wait for 5 seconds")
                            time.sleep(5)  # Wait for 5 seconds before retrying
                        else:
                            raise  # If the error is not the one we're expecting, re-raise it

                user_question += f". Use the documents referred: {documents_referred}"
                main_full_response = process_in_files_mode(user_question)
                st.session_state.main_conversation.append({"role": "Assistant", "content": main_full_response})
                if llama_parse_mode:
                    llama_parse_documents = LlamaParse(result_type="markdown").load_data('assets/files/uber_10q_march_2022.pdf')
                    node_parser = MarkdownElementNodeParser(llm=OpenAI(model="gpt-3.5-turbo-0125"), num_workers=8)
                    nodes = node_parser.get_nodes_from_documents(llama_parse_documents)
                    base_nodes, node_mapping = node_parser.get_base_nodes_and_mappings(nodes)
                    ctx = ServiceContext.from_defaults(
                        llm=OpenAI(model="gpt-4-0125-preview"), 
                        embed_model=OpenAIEmbedding(model="text-embedding-3-small"), 
                        chunk_size=512
                    )
                    recursive_index = VectorStoreIndex(nodes=base_nodes, service_context=ctx)
                    # raw_index = VectorStoreIndex.from_documents(llama_parse_documents, service_context=ctx)
                    retriever = RecursiveRetriever(
                        "vector", 
                        retriever_dict={
                            "vector": recursive_index.as_retriever(similarity_top_k=15)
                        },
                        node_dict=node_mapping,
                    )
                    reranker = SentenceTransformerRerank(top_n=5, model="BAAI/bge-reranker-large")
                    recursive_query_engine = RetrieverQueryEngine.from_args(retriever, node_postprocessors=[reranker], service_context=ctx)                                            
                    response_2 = recursive_query_engine.query(user_question)
                    # ic(response_2)
                    # st.session_state.main_conversation.append({"role": "Assistant", "content": "**Llama Parse Result: \n**"+str(response_2)})
                    user_question += f". Add on the Llama Parse Response Result: {str(response_2)}"
                    main_full_response_with_llama_parse = process_in_files_mode(user_question)
                    st.session_state.main_conversation.append({"role": "Assistant", "content": main_full_response_with_llama_parse})
                if vectara_query_mode:
                    query_engine = vectara_index.as_query_engine(similarity_top_k=5)
                    vectara_response = query_engine.query(user_question)
                    user_question += f". Add on the Vectara Query Response: {str(vectara_response)}"
                    main_full_response_with_vectara = process_in_files_mode(user_question)
                    st.session_state.main_conversation.append({"role": "Assistant", "content": main_full_response_with_vectara})
                # encode decoded string to utf-8
                # st.markdown("Assistant: ")
                # st.markdown(main_full_response)

# # ic(st.session_state['selected_files'])