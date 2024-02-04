import streamlit as st
import os

os.environ["LLAMA_CLOUD_API_KEY"] = 'llx-rnSkOSSGdKz0Tbbpf21AQkHOJfSe3J36OiyNX7oL79hgjTaY'
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

import time
startall = time.time()

from llama_parse import LlamaParse
from llama_index.node_parser import MarkdownElementNodeParser
from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.embeddings import OpenAIEmbedding
from llama_index.retrievers import RecursiveRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SentenceTransformerRerank

# assets\files\uber_10q_march_2022.pdf
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
# raw_query_engine = raw_index.as_query_engine(similarity_top_k=15, node_postprocessors=[reranker], service_context=ctx)

# print(f"Time taken for all operations: {time.time() - startall}")

# query = "how much is the Cash paid for 'Income taxes, net of refunds'?"

# start_raw_query = time.time()
# response_1 = raw_query_engine.query(query)
# print("\n***********New LlamaParse+ Basic Query Engine***********")
# print(response_1)
# print(f"Time taken for raw query: {time.time() - start_raw_query}")

# start_recursive_query = time.time()
# response_2 = recursive_query_engine.query(query)
# print("\n***********New LlamaParse+ Recursive Retriever Query Engine***********")
# print(response_2)
# print(f"Time taken for recursive query: {time.time() - start_recursive_query}")

# query = "what is the change of free cash flow and what is the rate?"

# response_1 = raw_query_engine.query(query)
# print("\n***********New LlamaParse+ Basic Query Engine***********")
# print(response_1)

# response_2 = recursive_query_engine.query(query)
# print("\n***********New LlamaParse+ Recursive Retriever Query Engine***********")
# print(response_2)

# query = "what is Net loss attributable to Uber compared to last year"

# response_1 = raw_query_engine.query(query)
# print("\n***********New LlamaParse+ Basic Query Engine***********")
# print(response_1)

# response_2 = recursive_query_engine.query(query)
# print("\n***********New LlamaParse+ Recursive Retriever Query Engine***********")
# print(response_2)

# query = "what is Cash flows from investing activities"

# response_1 = raw_query_engine.query(query)
# print("\n***********New LlamaParse+ Basic Query Engine***********")
# print(response_1)

# response_2 = recursive_query_engine.query(query)
# print("\n***********New LlamaParse+ Recursive Retriever Query Engine***********")
# print(response_2)

## Save the index and node_mapping
# import json

# recursive_index.storage_context.persist(persist_dir="./storage")

# node_mapping_json = {k: v.dict() for k, v in node_mapping.items()}
# with open("./node_mapping.json", "w") as f:
#     json.dump(node_mapping_json, f)

"""
Started parsing the file under job_id e0c29545-e5d0-413f-be56-081c6d7eef46
Embeddings have been explicitly disabled. Using MockEmbedding.
52it [00:00, 48413.72it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 52/52 [00:08<00:00,  6.43it/s] 
Time taken for all operations: 29.907794952392578

***********New LlamaParse+ Basic Query Engine***********
The provided context does not include specific information regarding the cash paid for income taxes, net of refunds.
Time taken for raw query: 15.045743227005005

***********New LlamaParse+ Recursive Retriever Query Engine***********
The cash paid for income taxes, net of refunds, is $22 million.
Time taken for recursive query: 15.21908950805664
"""