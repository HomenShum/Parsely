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


from llama_index.llms.openai import OpenAI
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding


logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

azure_endpoint=st.secrets["AOAIEndpoint"]
api_key=st.secrets["AOAIKey"]
api_version="2024-02-15-preview"

azure_gpt_4o_llm = AzureOpenAI(
    engine="gpt-4o",
    model="gpt-4o",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

# You need to deploy your own embedding model as well as your own chat completion model
azure_openai_embed_model = AzureOpenAIEmbedding(
    model="text-embedding-3-small",
    deployment_name="text-embedding-3-small",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

# Settings.llm = llm
# Settings.embed_model = embed_model

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
        OG_OpenAI_client = OG_OpenAI()

        message_placeholder = st.empty()
        responses = OG_OpenAI_client.chat.completions.create(
            model="gpt-4o-mini",
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
        nodes = await bm25_retriever.aretrieve(query)
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
                # logging.info(f"async def process_files_data() - Processing result: {result}")
                payload = {
                    "model": "gpt-4o-mini",
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
        OG_OpenAI_client = OG_OpenAI()

        # Generate user needs from the question
        user_needs = ""
        first_response_message_placeholder = st.empty()

        responses = OG_OpenAI_client.chat.completions.create(
            model="gpt-4o-mini",
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
            logging.info(f"Search Files Output Data List: {search_files_output_data_list}")
            ##### Dense: Cohere Rerank #####
            co = cohere.Client(st.secrets["COHERE_API_KEY"])
            rerank_search_files_output_data_list = [{'text': str({"result": result}).replace('$', '\$')} for result in search_files_output_data_list]
            rerank_response = co.rerank(
                model="rerank-english-v3.0",
                query=user_needs,
                documents=rerank_search_files_output_data_list,
                top_n=10,
                return_documents=True
            )

            ##### Dense: Cohere Rerank #####
            # Extract all document texts from the rerank results
            if rerank_response and rerank_response.results:
                processed_data = loop.run_until_complete(process_files_data(user_needs, rerank_response.results))
                main_full_response += "\n" + str(processed_data)
            else:
                logging.error("No results from rerank")
                main_full_response += "\nNo relevant documents found."

        finally:
            loop.close()



        # ic(main_full_response)

        # Final Response with gpt-4o

        final_response = ""
        second_response_message_placeholder = st.empty()
        responses = OG_OpenAI_client.chat.completions.create(
            model="gpt-4o",
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

    """ TODO:
    Your current function implements a sophisticated pipeline for query processing, search, reranking, and answering user questions, using both sparse (BM25) and dense (Cohere Rerank) search techniques. The flow also includes integrating results from Llama Parse and Qdrant for hybrid vector search and a final response summarization. Here is a breakdown of the workflow and suggestions for further improvement:

    Summary of Steps:
    User Intent Understanding:

    The function uses OpenAI's GPT-4o model to rephrase the user's input and extract key needs/topics from the query.
    This rephrased intent serves as the primary input for further processing.
    Sparse Search (BM25):

    A BM25-based keyword search on files is conducted to find relevant documents based on the user’s needs.
    This is the first pass for retrieving relevant results using simple term matching.
    Dense Rerank (Cohere):

    The search results are reranked using Cohere’s dense vector model to prioritize the most relevant documents.
    The reranked results are processed asynchronously to ensure speed and efficiency.
    Llama Parse + Qdrant Hybrid Search:

    Once reranked results are obtained, the function attempts to perform a hybrid search using Llama Parse, Qdrant, and custom vector search.
    This hybrid search helps improve both sparse and dense query processing by combining the strengths of both approaches.
    Final GPT-based Response Summarization:

    A final summarization step occurs where OpenAI’s GPT-4o model consolidates the search results and presents a concise, user-friendly answer.
    The response also includes a breakdown of key topics, confidence, and evidence sources.
    Suggestions and Optimizations:

    TODO:
    1. Improving Error Handling:
    Since the function interacts with multiple external APIs and performs complex operations, the error handling can be further enhanced. Currently, only minimal logging and exception catching are in place.
    Improvement:

    Consider adding more robust exception handling for API failures, network issues, and invalid data in both sparse and dense search steps.
    You can also add retries for external API requests in case of timeouts or temporary failures.
    python
    Copy code
    try:
        response_2 = query_engine.query(user_question)
    except Exception as e:
        logging.error(f"Error querying engine: {str(e)}")
        main_full_response += "\nLlama-Parse query failed."

    TODO:
    2. Cohere Rerank Results Handling:
    The reranking section relies heavily on getting results from Cohere. However, there is a fallback to report “No relevant documents found” if nothing is returned.
    Improvement:

    Consider returning even partial or less relevant results in case the top-ranked documents are not highly relevant, rather than showing a message like "No relevant documents found."

    TODO:    
    3. Asynchronous Processing Optimization:
    The use of asyncio.gather for handling multiple tasks is well done. However, if performance is critical, you could also ensure that the tasks are not unnecessarily blocking.
    Improvement:

    You might want to include a timeout for the async operations or prioritize the most important tasks first to avoid waiting too long for less critical responses.
    python
    Copy code
    responses = await asyncio.gather(*tasks, return_exceptions=True)  # Allows graceful handling of failed tasks

    TODO:
    4. Llama Parse + Qdrant Hybrid Configuration:
    The configuration of the Llama Parse and Qdrant hybrid search seems solid, but you could further tweak the parameters (e.g., similarity_top_k, sparse_top_k) based on performance needs or experimentation with different datasets.
    Improvement:

    Consider running benchmarks with different values for similarity_top_k and sparse_top_k to optimize the balance between sparse and dense results.
    Additionally, you could implement caching for repeated queries to avoid redundant recalculations.

    TODO:
    5. Summarization and GPT-4o Response Generation:
    The final summarization step is clear and well-structured. However, depending on user feedback or needs, you could also allow more granular control over the length or detail of the response.
    Improvement:

    You could offer options for the user to request either more or less detail in the final response. This can be done by adding optional parameters or a UI toggle for "Detailed Response" vs "Concise Response."
    python
    Copy code
    concise = st.checkbox("Provide a concise response", value=True)
    if concise:
        # Provide a short answer
    else:
        # Provide a detailed answer

    TODO:
    6. Caching & Reducing Redundant API Calls:
    If the same user questions or file searches are repeated, you could cache the search results or even the Llama Parse queries to avoid repeating expensive API calls unnecessarily.
    Improvement:

    Implement a caching layer for search results and GPT-based responses to improve performance on repeated queries.
    Potential Challenges:
    Latency: Combining multiple API calls (OpenAI, Cohere, Qdrant) could introduce latency, especially for large inputs or search results. Consider asynchronous execution and optimizing time-sensitive parts of the pipeline.
    Error Propagation: If a step fails (e.g., Qdrant collection creation), subsequent steps may depend on its success. Ensure there are appropriate fallbacks for each critical stage.

    TODO:
    7. Multi-File Mode:
    The user query may contain questions about multiple files.
    How to handle this? Assume user query = 
    i. "Could you find answer on ABC topic?" - The ABC topic exists in multiple files.
    ii. "Could you find answer on ABC topic and DEF topic?" - The ABC and DEF topics exist in two separate files. 
    iii. "Could you give me a summary on all of the files?" - Each file need to be summarized and combined into a single response.
    iv. "Could you compare ABC topic in file 1 and ABC topic in file 2?" - The ABC topic exists in both files.
    v. "Could you find ABC topic in file 1 only?" - The ABC topic should be found only in file 1.
    """
    def process_in_files_mode_with_llama_parse_chat_response(user_question, documents_referred):
        """
        Process user question in files mode with Llama Parse Chat response.

        Args:
        user_question (str): The user's question.

        Returns:
        str: The final response to the user's question.
        """

        # Initialize variables
        main_full_response = ""
        OG_OpenAI_client = OG_OpenAI()

        # Generate user needs from the question
        user_needs = ""
        first_response_message_placeholder = st.empty()

        # Use OpenAI chat completions to generate user needs in a key topic list
        responses = OG_OpenAI_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"This is process_in_files_mode_with_llama_parse_chat_response. Rephrase the User Input. Extrapolate the user needs in a key topic list. Begin with 'Here is what I am understanding from your question... List of key insights I will search for in the '{documents_referred}' files... Readdress the user question: {user_question}'"},
                {"role": "user", "content": f"User Input: {user_question}"}
            ],
            stream=True,
            seed=42,
        )
        for response in responses:
            # If there is a response, add it to the user needs
            if response.choices[0].delta.content:
                user_needs += str(response.choices[0].delta.content).replace('$', '\$')
                first_response_message_placeholder.markdown(f"**User Needs**: {user_needs}")
        st.markdown("---")

        # Add the user needs to the main conversation
        st.session_state.main_conversation.append({"role": "Assistant", "content": f"""**User Needs**: 
                                                                                        {user_needs}"""})
        
        # Add the user needs to the main full response
        main_full_response += user_needs

        # Search in FILESs
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Use BM25 search to find relevant documents in FILES
            search_files_output_data_list = loop.run_until_complete(files_bm25_search(query=user_needs))
            logging.info(f"Search Files Output Data List: {search_files_output_data_list}")

            # Dense: Cohere Rerank
            # Use Cohere rerank to rerank the search results
            co = cohere.Client(st.secrets["COHERE_API_KEY"])
            rerank_search_files_output_data_list = [{'text': str({"result": result}).replace('$', '\$')} for result in search_files_output_data_list]
            rerank_response = co.rerank(
                model="rerank-english-v3.0",
                query=user_needs,
                documents=rerank_search_files_output_data_list,
                top_n=10,
                return_documents=True
            )

            # Process the reranked search results
            if rerank_response and rerank_response.results:
                # Process the reranked search results
                processed_data = loop.run_until_complete(process_files_data(user_needs, rerank_response.results))
                main_full_response += "\n" + str(processed_data)
            else:
                logging.error("No results from rerank")
                main_full_response += "\nNo relevant documents found."

        finally:
            loop.close()

        # Llama Parse Chat Integration
        llama_parse_mode = True  # Assuming a condition to check whether to use Llama Parse
        if llama_parse_mode:
            # Use Llama Parse to generate a chat response
            from qdrant_client import QdrantClient, AsyncQdrantClient
            from qdrant_client import models
            from llama_index.core import Settings

            from llama_index.vector_stores.qdrant import QdrantVectorStore
            from llama_index.core import VectorStoreIndex, StorageContext

            # Initialize Qdrant client and vector store
            # qdrant_client = QdrantClient(host="localhost", port=6333)
            # qdrant_aclient = AsyncQdrantClient(host="localhost", port=6333)
            qdrant_client = QdrantClient(location=":memory:")
            qdrant_aclient = AsyncQdrantClient(location=":memory:")
            

            # Delete the collection if it exists
            # if qdrant_client.collection_exists(collection_name=f"{documents_referred}_documents_referred".replace('\\', '_').replace(':', '_')):
            #     qdrant_client.delete_collection(collection_name=f"{documents_referred}_documents_referred".replace('\\', '_').replace(':', '_'))

            if not qdrant_client.collection_exists(collection_name=f"{documents_referred}_documents_referred".replace('\\', '_').replace(':', '_')):
                ### TODO:
                ## Create collection with collection name pointing to the document referred 
                ## If document does not exist, create it
                
                # Create a new collection
                qdrant_client.create_collection(
                    collection_name=f"{documents_referred}_documents_referred".replace('\\', '_').replace(':', '_'),
                    vectors_config={
                        "text-dense": models.VectorParams(
                            size=1536,  # OpenAI Embeddings
                            distance=models.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        "text-sparse": models.SparseVectorParams(
                            index=models.SparseIndexParams(
                                on_disk=False,
                            )
                        )
                    },
                )

            # Initialize the vector store
            vector_store = QdrantVectorStore(
                collection_name=f"{documents_referred}_documents_referred".replace('\\', '_').replace(':', '_'),
                client=qdrant_client,
                aclient=qdrant_aclient,
                enable_hybrid=True,
                batch_size=20,
                fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions",
            )
            # Initialize the storage context
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            Settings.chunk_size = 512
            # Initialize the hybrid index
            # ic(st.session_state['llama_parse_hybrid_index'])
            
            index = VectorStoreIndex.from_documents(
                documents=st.session_state['llama_parse_hybrid_index'],
                embed_model=azure_openai_embed_model,
                storage_context=storage_context,
            )

            # Initialize the query engine
            query_engine = index.as_query_engine(
                similarity_top_k=2, 
                sparse_top_k=12, 
                vector_store_query_mode="hybrid"
            )
            # ic(query_engine)
            try:
                response_2 = query_engine.query(user_question)
                main_full_response += "\nLlama-Parse query = " + str(response_2)
            except Exception as e:
                logging.error(f"Error querying engine: {str(e)}")
                main_full_response += "\nLlama-Parse query = null."


        # Final Response with gpt-4o to combine responses from FILES BM25+Cohere Rerank Response and Llama-Parse Hybrid Response
        final_response = ""
        second_response_message_placeholder = st.empty()
        responses = OG_OpenAI_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"Highly detailed **Answer**, rest of the response should be concise, unless user asks for more details. Make sure to rephrase the User Input. Extrapolate the user needs in a key topic list. Utilize FILES Search Result. Begin with '**Question Summary**, **Key Topics**, **Answer**, **Source of Evidence**, **Confidence ** (low medium high)...'"},
                {"role": "user", "content": f"User Input: {user_question}, User Needs: {user_needs}, FILES Search Result: {processed_data}, Main Full Response: {main_full_response}"}
            ],
            stream=True,
            seed=42,
        )
        for response in responses:
            if response.choices[0].delta.content:
                final_response += str(response.choices[0].delta.content).encode('utf-8').decode('utf-8').replace('$', '\$')
                second_response_message_placeholder.markdown(f"**Final Response**: {final_response}")
        st.markdown("---")
        return final_response


    ### Title and Description
    st.title("💬 Chat for All Files")
    st.toast("🌱 Hi! I’m Parsely and I’m here to assist your document analysis needs. Type below to get started.")

    ### FileUploader from utils.py
    uploader = FileUploader(SUPPORTED_EXTENSIONS)
    uploader.upload_files()

    st.toast(f'📁 {len(st.session_state["selected_files"])} files uploaded')

    ### Conversation Memory and Summary
    if "main_conversation" not in st.session_state:
        st.session_state["main_conversation"] = []

    if "main_conversation_memory" not in st.session_state:
        st.session_state["main_conversation_memory"] = []
        st.session_state["main_conversation_memory"].append("")

    ### Llama_Parse Mode
    llama_parse_mode = sac.switch(label='Llama_Parse Mode (Hybrid Response)', align='start', size='md')
    st.session_state["llama_parse_mode"] = llama_parse_mode

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

    # Function to extract 'source_name' and 'index' using regex from the text field
    def extract_source_name_and_index(text):
        """
        This function extracts the 'source_name' and 'index' from a document text field.
        """
        try:
            # Define regex patterns to extract source_name and index
            source_name_pattern = re.compile(r"'source_name':\s*'([^']+)'")
            index_pattern = re.compile(r"'index':\s*(\d+)")

            # Search for source_name and index in the text
            source_name_match = source_name_pattern.search(text)
            index_match = index_pattern.search(text)

            # Extract and return the matches or fallback to 'Unknown'
            source_name = source_name_match.group(1) if source_name_match else 'Unknown'
            index = index_match.group(1) if index_match else 'N/A'

            return source_name, index
        except Exception as e:
            st.toast(f"Error extracting source_name and index: {e}")
            return 'Unknown', 'N/A'



    ### Chat Interface
    user_question = st.chat_input("Ask a question about the selected PDF content:")

    if user_question:
        with st.chat_message("User"):
            st.markdown("User: ")
            st.markdown(user_question)
            st.session_state.main_conversation.append({"role": "User", "content": user_question})


        with st.chat_message("Assistant"):
            if not st.session_state['selected_files']:
                # No files selected, default mode
                default_response_with_custom_prompt = process_in_default_mode(user_question)
                st.session_state.main_conversation.append({"role": "Assistant", "content": default_response_with_custom_prompt})
            else:
                # Files selected, process them
                for _ in range(3):  # Retry up to 3 times
                    try:
                        llama_index_node_documents = []
                        documents_referred = []

                        # Loop through selected files
                        for document_obj in st.session_state['selected_files']:
                            with open("output_file.txt", "w", encoding="utf-8") as f:
                                f.write(str(document_obj))
                            
                            # Check if document_obj is a dictionary of documents
                            if isinstance(document_obj, dict):
                                for id, doc in document_obj.items():
                                    llama_index_node_documents.append(doc)

                                    # Parse the text field to extract source_name and index
                                    source_name, index = extract_source_name_and_index(doc.text)
                                    st.toast(f"Document {source_name} with index: {index}")
                                    documents_referred.append(f"{source_name} index: {index}")

                            # Handle non-dict document_obj
                            else:
                                llama_index_node_documents.append(document_obj)

                                # Parse the text field for non-dict objects
                                source_name, index = extract_source_name_and_index(document_obj.text)
                                st.toast(f"Document {source_name} with index: {index}")
                                documents_referred.append(f"{source_name} index: {index}")

                        # Processing documents
                        from llama_index.core.schema import Document
                        from llama_index.core.node_parser import SentenceSplitter
                        from llama_index.core.storage.docstore import SimpleDocumentStore
                        from llama_index.retrievers.bm25 import BM25Retriever
                        import Stemmer

                        # Create sentence splitter and parse documents into nodes
                        node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
                        nodes = node_parser.get_nodes_from_documents(documents=llama_index_node_documents)
                        num_nodes = len(nodes)
                        
                        # Adjust similarity_top_k_value based on the number of nodes
                        similarity_top_k_value = min(10, num_nodes)
                        
                        # Create document store and add nodes
                        docstore = SimpleDocumentStore()
                        docstore.add_documents(nodes)
                        
                        # Initialize BM25 retriever with adjusted k value
                        bm25_retriever = BM25Retriever.from_defaults(
                            docstore=docstore,
                            similarity_top_k=similarity_top_k_value,
                            stemmer=Stemmer.Stemmer("english"),
                            language="english",
                        )

                        break  # Break the retry loop if successful
                    except ValueError as e:
                        if str(e) == "Please pass exactly one of index, nodes, or docstore.":
                            st.toast("Please wait for 5 seconds")
                            time.sleep(5)
                        else:
                            raise  # Reraise unexpected errors

                # Modify user question with referred documents
                user_question += f". Use the documents referred: {documents_referred}"
                
                if llama_parse_mode:
                    main_full_response_with_llama_parse = process_in_files_mode_with_llama_parse_chat_response(user_question, documents_referred)
                    st.session_state.main_conversation.append({"role": "Assistant", "content": main_full_response_with_llama_parse})
                    st.toast("Llama-parse response generated", icon='🦙')
                else:
                    main_full_response = process_in_files_mode(user_question)
                    st.session_state.main_conversation.append({"role": "Assistant", "content": main_full_response})
