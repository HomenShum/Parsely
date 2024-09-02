def product_recommendation_builder():
    import os
    import sys
    import logging
    import time
    from typing import Any, List
    import numpy as np
    import pandas as pd
    import streamlit as st

    # Configure logging
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().handlers = []
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    # Import llama_index modules
    from llama_index.core import (
        ServiceContext,
        Document,
        Settings,
        VectorStoreIndex,
        SimpleDirectoryReader,
        QueryBundle,
    )
    from llama_index.retrievers.bm25 import BM25Retriever
    from llama_index.llms.openai import OpenAI
    from llama_index.llms.azure_openai import AzureOpenAI
    from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
    from llama_index.core.postprocessor import LLMRerank
    from llama_index.core.retrievers import VectorIndexRetriever
    from IPython.display import HTML, display
    

    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

    if "count" not in st.session_state:
        st.session_state.count = 3

    @st.cache_data
    def sparse_dense_retrieval(query, count):
        ##### Sparse: Llama Index BM25 #####
        start = time.time()

        OpenAI.api_key = st.secrets["OPENAI_API_KEY"]

        # Load the dataset
        df = pd.read_excel('click_car_product_dataset.xlsx', sheet_name='productos')
        df = df.replace({np.nan: None})  # Replace NaN values with None

        # Convert DataFrame to a list of dictionaries
        search_results = df.to_dict(orient='records')

        formatted_results = []

        for i, article in enumerate(search_results):
            formatted_article = {
                "Number": i + 1,
                "Name": article.get("NOMBRE ARTICULO", "N/A"),
                "Description": article.get("DEFINICION DEL ARTICULO", "N/A"),
                "Short Description": article.get("NOMBRE CORTO DEL ARTICULO", "N/A"),
                "Category": article.get("CATEGORIA (DEPARTAMENTO)", "N/A"),
                "Brand": article.get("MARCA DEL REPUESTO", "N/A"),
                "Vehicle Brand": article.get("MARCA VEHICULO (si hay varios, separados por coma)", "N/A"),
                "Vehicle Model": article.get("MODELO DE VEHICULO", "N/A"),
                "Engine": article.get("MOTOR (Cilindrada)", "N/A"),
                "SKU/OEM": article.get("OEM / SKU", "N/A"),
                "Technical Link": article.get("LINK FICHA TÉCNICA", "N/A"),
                "Warnings": article.get("Advertencias", "N/A"),
                "Recommendations": article.get("Recomendaciones.", "N/A"),
                "Provider": article.get("PROVEEDOR", "N/A"),
                "ClickCar ID": article.get("CLICKCAR ID", "N/A"),
                "ClickCar ID with Storage": article.get("CLICKCAR ID Con almacen", "N/A"),
                "Brand Code": article.get("CODIGO DE MARCA", "N/A")
            }
            formatted_results.append(str(formatted_article))

        # Convert each record to a Document object for Llama Index
        documents = []
        for record in formatted_results:
            text_for_bm25 = ''.join([str(value) for value in record])
            document = Document(text=text_for_bm25)
            documents.append(document)

        azure_endpoint = st.secrets["AOAIEndpoint"]
        api_key = st.secrets["AOAIKey"]
        api_version = "2024-02-15-preview"

        llm = AzureOpenAI(
            model="gpt-4o",
            deployment_name="gpt-4o",
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )

        embed_model = AzureOpenAIEmbedding(
            model="text-embedding-3-small",
            deployment_name="text-embedding-3-small",
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )

        Settings.llm = llm
        Settings.embed_model = embed_model
        service_context = ServiceContext.from_defaults(embed_model=embed_model)
        nodes = service_context.node_parser.get_nodes_from_documents(documents)

        bm25retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=15)
        retrieved_nodes = bm25retriever.retrieve(query)

        logging.info(f"Output data before LLM rerank: {retrieved_nodes}")

        ##### Dense: LLM Rerank #####
        reranker = LLMRerank(
            choice_batch_size=5,
            top_n=count,
        )

        query_bundle = QueryBundle(query)
        reranked_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle)

        formatted_results = []

        for node in reranked_nodes:
            formatted_results.append({
                "part_details": node.node.get_text(),
                "score": node.score,
            })

        return formatted_results



    from openai import OpenAI
    import openai
    import streamlit as st
    import pandas as pd
    import copy
    import requests
    from deep_translator import GoogleTranslator
    import json
    import re 
    import os
    #################### Settings ####################
    client = OpenAI()

    # If your .env file is in the same directory as your script

    url1 = st.secrets['url1']
    url3 = st.secrets['url3']

    def translate_to_esp(text):
        translated_text = GoogleTranslator(source='english', target='spanish').translate(text)
        return translated_text

    def translate_to_eng(text):
        translated_text = GoogleTranslator(source='spanish', target='english').translate(text)
        return translated_text

    @st.cache_data
    def memory_summary_agent(memory):
        summarization = client.chat.completions.create(
            model = st.session_state["openai_model"],
            messages=[
                {"role": "system", "content": 'Summarize the conversation so far:'},
                {"role": "user", "content": memory},
            ],
            stop=["None"],
            seed=42,   
        )
        text = summarization.choices[0].message.content.strip()
        text = re.sub("\s+", " ", text)
        return text

    @st.cache_data
    def summarize_all_messages(message):
        return memory_summary_agent(". ".join(st.session_state.memory) + ". " + message)

    @st.cache_data
    def retrieve_auto_parts_details(query):
        """
        Retrieve the most relevant auto part details based on a given query.

        :param url: The endpoint URL to make the request to.
        :param query: The search query to find auto parts.
        :return: A dictionary containing a list of auto parts with their details.
        """
        url = st.secrets['url1']
        # url = st.secrets['url1']
        try:
            response = requests.get(url, params={'q': query})
            response.raise_for_status()
            # We expect the response to be a JSON with a 'results' field containing the parts details
            part_details = response.json().get('results', [])
            st.session_state.df = pd.DataFrame(part_details)
            with st.expander("Auto Part Details 2"):
                st.dataframe(st.session_state.df)
            st.session_state.memory.append(memory_summary_agent(". New data pulled from database that may or may not match user needs: " + str(part_details) + ". Previous information regarding user needs: " + st.session_state.memory[-1]))
            return {'part_details': part_details}
        except requests.exceptions.RequestException as e:
            # This will capture any errors related to the request
            return {'error': str(e)}
    # Set the system and user prompts


    @st.cache_data
    def precision_retrieve_auto_parts_details(query):
        """
        Retrieve the most relevant auto part details based on a given query.

        :param url: The endpoint URL to make the request to.
        :param query: The search query to find auto parts.
        :return: A dictionary containing a list of auto parts with their details.
        """
        url = st.secrets['url3']
        try:
            response = requests.get(url, params={'q': query})
            response.raise_for_status()
            # We expect the response to be a JSON with a 'results' field containing the parts details
            part_details = response.json().get('results', [])
            st.session_state.df = pd.DataFrame(part_details)
            with st.expander("Auto Part Details"):
                st.dataframe(st.session_state.df)
            st.session_state.memory.append(memory_summary_agent(". New data pulled from database that may or may not match user needs: " + str(part_details) + ". Previous information regarding user needs: " + st.session_state.memory[-1]))
            return {'part_details': part_details}
        except requests.exceptions.RequestException as e:
            # This will capture any errors related to the request
            return {'error': str(e)}

    # Set the system and user prompts

    salesperson_system_prompt = (
        "Objective: Recommend Auto Parts from Inventory Database and Provide Link using Schema.\n"
        "Potential Response: For the Hyundai Tucson suspension system, I recommend that you change the entire set of shock absorbers, bushings and bearings, it is also recommended to check the engine bases. Annex recommended products:"
        "Potential Response: If your vehicle's battery is discharging, it could be that the battery is in poor condition, you should check at your nearest maintenance center, it could also be the alternator. Here we show you some related products."
        "Potential Response: When you notice white smoke coming out of your hood, it may have several reasons, the main reasons may be engine overheating, damage to the cooling system or water passing into the combustion chamber or injection system. For this you must check your cooling system, and the smoke coming out of the exhaust pipe. Here are some related products, with their possible failure."
        "Potential Response: Seeing a puddle of oil under your engine means that you have a considerably large oil leak, this can be due to damage to the oil cooler and its hoses, damage to gaskets, leaks in the oil filters or even damage to the engine block. engine. I attach some products related to its failure."
        "Potential Response: When changing gears and the shift is not executed correctly, this may be due to clutch wear or lack of oil in it, in addition to possible overheating and possible damage to the clutch fan. Annex some related products."
        "Potential Response: If your Corolla does not start, it may be because the automatic starter is damaged, or has some malfunction, for this we recommend the following products for your vehicle."
        "Potential Response: This failure may be due to a failure in the gasoline injection system, gasoline filters, or gasoline pump. I attach some products that can help you for this failure."
        "Potential Response: This failure may be due to the fact that your car's bearings are worn and require regreasing or replacement. Here I show you the bearings and grease."
        "Potential Response: In this case, your brake pads are worn and the safety hook is making contact with the brake disc, you should change your brake pads as soon as possible. Recommended spare parts annex."
        "Potential Response: This may be because your brake discs are scratched or damaged, for this you can take them to a mechanical workshop so they can evaluate the damage and tell you if they can be rectified or if they need to be changed. Annex products related to your problem."
        "Potential Response: I have a Mazda Bt50, and when I cross the car to the right it makes a clicking or clicking noise when I cross and move forward with the car, what could it be?"
        "Potential Response: This may be due to the fact that the electric fan bearing is expired and is resting on the safety protector and when it is turned on it hits it, for this it is recommended to adjust the bearing or change the electric fan. Annex products related to your failure."
        "Potential Response: When the coolant liquid changes color or viscosity, this is usually due to contamination, either because the engine oil is passing into the cooling system or external agents. For this, it is recommended to check the radiator and hoses, in addition to the oil coolers, hoses and case coils. Annex rescaled spare parts with their failure."
        "Potential Response: This noise may be due to a lack of grease in the bushings and bearings. For this reason, we recommend that you re-grease the steering system or change the bushings and bearings of your Jeep Grand Cherokee. Annex spare parts for your failure."
        "Potential Response: This failure may be due to damage to the radiator, hoses or water pump, it is recommended to check these auto parts. Annex spare parts for your failure."
        """
        Schema / Response Format for when the user needs are clear and relevant to the user prompt:
        -----------------------------
        \nRecommended products with links:	

        \nProblem and Solution:

        \nClickCar Store Link: https://www.clickcar.store/products
        
        \n\nLink 1. https://www.clickcar.store/products/product-1-Technical-Link
            \nProduct Name: 
            \nConfidence Score:
            \nRationale:
        \n\nLink 2. https://www.clickcar.store/products/product-2-Techincal-Link
            \nProduct Name:
            \nConfidence Score:
            \nRationale:
        \n\nLink 3. https://www.clickcar.store/products/product-3-Technical-Link
            \nProduct Name:
            \nConfidence Score:
            \nRationale:

        -----------------------------
        Definitions:
        Problem and Solution: Explain the problem given the user need and user prompt and why the recommended auto parts are relevant to the user need and user prompt.
        Links: Provide links to the recommended auto parts from inventory database. If no link then provide default to https://www.clickcar.store/products
        Product Name: Provide the name of the auto part
        Confidence Score: Provide a confidence score for each recommendation "Low | Medium | High"
        Rationale: Explain the problem given the user need and user prompt and why the recommended auto parts are relevant to the user need and user prompt.

        Edge Cases: User need is not clear, user need is not relevant to the user prompt, user need is not relevant to the recommended auto parts, user need is not relevant to the retrieved auto part information from inventory database.

        If user need is not clear, ask user to clarify the user need. If user need is not relevant to the user prompt, ask user to clarify the user need. If user need is not relevant to the recommended auto parts, ask user to clarify the user need. If user need is not relevant to the retrieved auto part information from inventory database, ask user to clarify the user need.
        """
    )


    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4o"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "memory" not in st.session_state:
        st.session_state.memory = []

    if "auto_part_criteria" not in st.session_state:
        st.session_state.auto_part_criteria = []

    if "auto_part_details" not in st.session_state:
        st.session_state.auto_part_details = []

    if "summary" not in st.session_state:
        st.session_state.summary = []

    if "auto_part_image_link" not in st.session_state:
        st.session_state.auto_part_image_link = []

    st.session_state.memory.append("")

    st.session_state["openai_model"] = "gpt-4o-mini"
    st.session_state["openai_model_context_window"] = "16K Tokens"
    st.session_state["openai_model_training_data"] = "Up to Sep 2021"




    #################### Main ####################
    ### OpenAI API Key Management ###
    selected_key = None
    with st.expander("OpenAI API Key Management"):
        st.header("Input your OpenAI API Key")
        selected_key = st.text_input("API Key", type="password")
        if re.match(r"sk-\S+", selected_key):
            st.success("Valid API Key")
            st.session_state["openai_api_key"] = selected_key

        st.divider()

        st.subheader("Translate to Espanol or English")
        
        # if st.button("Translate to Espanol"):
        #     for i in st.session_state.messages:
        #         i["content"] = translate_to_esp(i["content"])
        #     st.session_state.auto_part_criteria.append(translate_to_esp(st.session_state.auto_part_criteria[-1]))
        #     st.session_state.summary.append(translate_to_esp(st.session_state.summary[-1]))
        #     st.rerun()

        # if st.button("Translate to English"):
        #     for i in st.session_state.messages:
        #         i["content"] = translate_to_eng(i["content"])
        #     st.session_state.auto_part_criteria.append(translate_to_eng(st.session_state.auto_part_criteria[-1]))
        #     st.session_state.summary.append(translate_to_eng(st.session_state.summary[-1]))
        #     st.rerun()
        
        if st.button("Translate to Espanol"):
            for i in st.session_state.messages:
                i["content"] = translate_to_esp(i["content"])
            if st.session_state.auto_part_criteria:
                st.session_state.auto_part_criteria.append(translate_to_esp(st.session_state.auto_part_criteria[-1]))
            else:
                st.toast("No auto part criteria yet.")
            if st.session_state.summary:
                st.session_state.summary.append(translate_to_esp(st.session_state.summary[-1]))
            else:
                st.toast("No summary yet.")
            # st.rerun()

        if st.button("Translate to English"):
            for i in st.session_state.messages:
                i["content"] = translate_to_eng(i["content"])
            if st.session_state.auto_part_criteria:
                st.session_state.auto_part_criteria.append(translate_to_eng(st.session_state.auto_part_criteria[-1]))
            else:
                st.toast("No auto part criteria yet.")
            if st.session_state.summary:
                st.session_state.summary.append(translate_to_eng(st.session_state.summary[-1]))
            else:
                st.toast("No summary yet.")
            # st.rerun()

        st.divider()

        st.session_state.count = st.number_input('Insert desired number of retrieval', value=3, max_value=10, min_value=1, step=1)
        # #print("\n\nst.session_state.count:", st.session_state.count)
        
        st.divider()

    ### Streamlit UI ###
    st.title("ClickCar Chat Agents")
    st.subheader("What auto parts are you looking for from ClickCar store?\n")
    st.subheader(translate_to_esp("What auto parts are you looking for from ClickCar store?"))

    if st.button("Clear Messages"):
        st.session_state.messages = []
        st.session_state.memory = []
        st.session_state.auto_part_criteria = []
        st.session_state.auto_part_details = []
        st.session_state.summary = []
        st.session_state.auto_part_image_link = []
        st.rerun()
        st.stop()
    ###
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    ###


    if prompt := st.chat_input("Any auto part that you are looking for in ClickCar?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # this is the auto part criteria extracted from user prompt

        auto_part_criteria_response = ""
        for response in client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[{"role": "system", "content": 'List most confident potential auto part needs using only list of key words, stay concise.'}, 
                        {"role": "user", "content": 'User input:' + prompt}],
            stream=True,
            seed = 42,
        ):
            auto_part_criteria_response += str(response.choices[0].delta.content)
        st.session_state.auto_part_criteria.append(auto_part_criteria_response)
        #print ("\nauto_part_criteria_response:\n", auto_part_criteria_response)


        # #print("\nCount sanity check:\n", st.session_state.count)
        sparse_dense_retrieval_response = sparse_dense_retrieval(auto_part_criteria_response, st.session_state.count)
        if "error" in sparse_dense_retrieval_response:
            st.session_state.auto_part_details.append(f"Based on user input: {prompt}. No auto part found.")
            st.stop()
        else:
            st.session_state.auto_part_details.append(str(sparse_dense_retrieval_response))


        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            full_spanish_response = ""
            for response in client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": "system", "content": salesperson_system_prompt},
                    {"role": "user", "content": ". \nPrior Conversation:\n"+ st.session_state.memory[-1] + ". \nIgnore this information if it is not relevant to recommended auto parts. \n"},
                    {"role": "user", "content": ". \nUser prompt: \n"+ prompt + ". \n Recommended auto parts: \n" + auto_part_criteria_response},
                    {"role": "user", "content": ". \nRetrieved auto part information from inventory database: \n"+ str(st.session_state.auto_part_details[-1])},
                    # {"role": "user", "content": f". \n1. Diagnose automotive issue from customer's description.\n 2. Suggest potential causes and solutions.\n 3. Provide {count} auto part recommendation, description, rationale, confidence, and links given information from user needs: \n {auto_part_criteria_response} \nShow Confidence: Low | Medium | High"},
                ],
                stop=["None"],
                stream=True,
                seed=42,
            ):
                full_response += str(response.choices[0].delta.content)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response + "\n")
            links = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', full_response)
            #print ('\nlinks\n', links)
            st.session_state.auto_part_image_link.append(links)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.memory.append(memory_summary_agent(". New information from Assistant about user needs: " 
                                                                + auto_part_criteria_response 
                                                                + ". Response to user:" 
                                                                + full_response 
                                                                + ". Previous information: " 
                                                                + st.session_state.memory[-1]
                                                                + " User Prompt: "
                                                                + prompt
                                                                ))
            #summary
            st.session_state.summary.append(summarize_all_messages(prompt))

    # #print("\ndebug prompt: ", prompt, "\n")

    with st.expander("Auto Part Details Inside"):
        
        st.header("Auto Part Criteria")
        if not st.session_state.auto_part_criteria:
            st.write("No auto part criteria yet.")
        else:
            st.write(st.session_state.auto_part_criteria[-1])
            
        st.divider()
        
        st.header("Summary")
        if not st.session_state.summary:
            st.write("No summary yet.")
        else:
            st.write(st.session_state.summary[-1])

        st.divider()

        # Test Image Feature
        st.header("Auto Part Image (Test)")
        if not st.session_state.auto_part_image_link:
            st.write("No auto part images available yet.")
            st.session_state.auto_part_image_link.append("https://www.clickcar.store/cdn/shop/products/G2225011R-G2225011R-MASTER_KING_1_1220x_crop_center.png?v=1696430675")
            st.write("Test Auto Part Link: https://www.clickcar.store/products/amortiguadores-hyundai-tucson-derecho-trasero-marca-master-king?_pos=1&_sid=66f9b566e&_ss=r")
            st.image(st.session_state.auto_part_image_link[-1], caption="Test Auto Part Image", width=300)
        else:
            st.info("This feature would work if dataset contains image links. Right now dataset doesn't have the image links")
            if st.session_state.auto_part_image_link:
                for i in range(1, min(st.session_state.count + 1, len(st.session_state.auto_part_image_link) + 1)):
                    links = st.session_state.auto_part_image_link[-i]
                    for link in links:
                        # prevent single string such as h in https from being iterated
                        if len(link) > 10:                        
                            st.write("Auto Part Link: ", link)
                            st.image(link, caption=f"Auto Part Image{i}", width=300)
