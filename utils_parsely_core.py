from PIL import Image
from torchvision import transforms
from transformers import AutoModelForObjectDetection
import torch
import openai
import os
import streamlit as st
import time
import aiohttp

start_time = time.time()

device = "cuda" if torch.cuda.is_available() else "cpu"

# openai.api_key = st.secrets["openai_api_key"]

##############################################################################################################
##### Auto Pydantic Model Generation + FILES Content Processing #####
##############################################################################################################

from pydantic import BaseModel, Field, HttpUrl
import asyncio
from openai import AsyncOpenAI
import instructor
from typing import List, Optional, Dict
import json
from icecream import ic
import logging
from tenacity import retry, stop_after_attempt, wait_fixed

retry_decorator = retry(stop=stop_after_attempt(3), wait=wait_fixed(1))


class AutoPartRecommendation(BaseModel):
    Link: HttpUrl
    ProductName: Optional[str] = None
    ConfidenceScore: Optional[str] = None  # "Low" | "Medium" | "High"
    Rationale: Optional[str] = None

class SalesPersonResponse(BaseModel):
    RecommendedProducts: Optional[List[AutoPartRecommendation]] = None
    ProblemAndSolution: Optional[str] = None
    ClickCarStoreLink: Optional[HttpUrl] = "https://www.clickcar.store/products"

class FinanceInfo(BaseModel):
    Keywords: str
    Summary: str
    Company: Optional[str] = None
    Comments: Optional[str] = None
    Overview: Optional[str] = None
    Patents: Optional[str] = None
    LastFinancing: Optional[str] = None
    InvestmentStage: Optional[str] = None
    LastDealDetails: Optional[dict] = None
    PostValuation: Optional[str] = None
    TotalRaisedToDate: Optional[str] = None
    MarketCap: Optional[str] = None
    EV: Optional[str] = None
    TTMTotalRevenue: Optional[str] = None
    TTMEBITDA: Optional[str] = None
    Investors: Optional[List[str]] = None  # This could be a list
    BoardOfDirectors: Optional[List[str]] = None  # This could be a list
    Website: Optional[str] = None
    CSuite: Optional[List[str]] = None  # This could be a list
    Email: Optional[str] = None
    Phone: Optional[str] = None
    StreetAddress: Optional[str] = None
    City: Optional[str] = None
    State: Optional[str] = None
    Zip: Optional[str] = None
    NewsAlert: Optional[str] = None
    ProspectingTopicList: Optional[str] = None
    EmailTemplateForProspecting: Optional[str] = None
    SectorTags: Optional[List[str]] = None  # This could be a list
    GeneralTags: Optional[List[str]] = None  # This could be a list
    Questions: Optional[List[str]] = None
    

class AcademicResearchInfo(BaseModel):
    Keywords: str
    Summary: str
    Title: str
    Abstract: str
    Authors: List[str] = Field(default_factory=list)  # Already a list
    Methods: Optional[str] = None
    Results: Optional[str] = None
    Discussion: Optional[str] = None
    References: Optional[List[str]] = None  # This could be a list
    Questions: Optional[List[str]] = None
    

class BiologyResearchInfo(BaseModel):
    Keywords: str
    Summary: str
    StudyTitle: Optional[str] = None
    Species: Optional[str] = None
    GenomicData: Optional[str] = None
    ExperimentalSetup: Optional[str] = None
    Findings: Optional[str] = None
    BiologicalImplications: Optional[str] = None
    Charts: Optional[List[str]] = None  # This can be a list to accommodate multiple charts
    Questions: Optional[List[str]] = None
    

class MedicalResearchInfo(BaseModel):
    Keywords: Optional[str] = None
    Summary: Optional[str] = None
    ResearchTitle: Optional[str] = None
    ClinicalTrialPhase: Optional[str] = None
    PatientData: Optional[str] = None
    Treatment: Optional[str] = None
    Outcomes: Optional[str] = None
    MedicalConclusions: Optional[str] = None
    EthicalConsiderations: Optional[str] = None
    Questions: Optional[List[str]] = None
    

class TechnologyResearchInfo(BaseModel):
    Keywords: str
    Summary: str
    ArticleTitle: str
    TechKeywords: Optional[str] = None
    DevelopmentTools: Optional[str] = None
    Implementation: Optional[str] = None
    PerformanceMetrics: Optional[str] = None
    FutureScope: Optional[str] = None
    CodeSnippets: Optional[str] = None
    UpdateImpact: Optional[str] = None
    TechnologyEvolution: Optional[str] = None
    Questions: Optional[List[str]] = None
    

class LegalDocumentInfo(BaseModel):
    Keywords: Optional[str] = None
    Summary: Optional[str] = None
    DocumentTitle: Optional[str] = None
    PartiesInvolved: List[str] = Field(default_factory=list)
    AgreementType: Optional[str] = None
    KeyTerms: Optional[str] = None
    Jurisdiction: Optional[str] = None
    EffectiveDate: Optional[str] = None
    ExpirationDate: Optional[str] = None
    Signatories: List[str] = Field(default_factory=list)
    Questions: Optional[List[str]] = None
    

class TechnologyPatentInfo(BaseModel):
    Keywords: str
    Summary: str
    PatentTitle: str
    Inventors: List[str] = Field(default_factory=list)  # Already a list
    PatentNumber: str
    FilingDate: str
    PublicationDate: str
    PatentAbstract: str
    Claims: Optional[List[str]] = None  # This could be a list of different claims
    CPCClassification: Optional[str] = None
    Questions: Optional[List[str]] = None
    

class EducationalMaterialInfo(BaseModel):
    Keywords: str
    Summary: str
    BookTitle: str
    Chapters: List[str] = Field(default_factory=list)
    Authors: List[str] = Field(default_factory=list)
    Publisher: Optional[str] = None
    PublicationYear: Optional[str] = None
    ISBN: Optional[str] = None
    KeyTopics: Optional[List[str]] = None  # This could be a list
    Questions: Optional[List[str]] = None
    

class ClinicalGuidelinesInfo(BaseModel):
    Keywords: str
    Summary: str
    GuidelineTitle: str
    IssuingOrganization: str
    TargetPopulation: str
    ClinicalRecommendations: str
    EvidenceLevel: str
    LastUpdated: str
    References: Optional[List[str]] = None  # Adjusted to a list
    Questions: Optional[List[str]] = None
    

class RulesRegulationsInfo(BaseModel):
    Keywords: Optional[str] = None
    Summary: Optional[str] = None
    RegulationTitle: Optional[str] = None
    IssuingAuthority: Optional[str] = None
    Scope: Optional[str] = None
    KeyRegulations: Optional[List[str]] = None  # This could be a list
    ComplianceRequirements: Optional[List[str]] = None  # This could be a list
    EffectiveDate: Optional[str] = None
    RevisionHistory: Optional[List[str]] = None  # This could be a list
    Questions: Optional[List[str]] = None
    

class MarketingTrendInfo(BaseModel):
    Keywords: str
    Summary: str
    ReportTitle: str
    MarketOverview: str
    KeyTrends: Optional[List[str]] = None  # This could be a list
    ConsumerBehavior: Optional[str] = None
    CompetitiveLandscape: Optional[str] = None
    Predictions: Optional[List[str]] = None  # This could be a list
    DataSources: Optional[List[str]] = None  # This could be a list
    Questions: Optional[List[str]] = None
    

class ProductInfo(BaseModel):
    Keywords: str
    Summary: str
    ProductName: str
    Manufacturer: str
    Specifications: str
    UsageInstructions: Optional[str] = None
    SafetyInformation: Optional[str] = None
    CustomerReviews: Optional[List[str]] = None  # This could be a list
    PriceRange: Optional[str] = None
    ReleaseDate: Optional[str] = None
    PerformanceMetrics: Optional[str] = None
    UpdateDetails: Optional[str] = None
    RelatedTechnologies: Optional[List[str]] = None  # This could be a list
    Questions: Optional[List[str]] = None
    

aclient = instructor.apatch(AsyncOpenAI(api_key = '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'))

@retry_decorator
async def process_chunk_content_instructor(chunk_content: str, sem: asyncio.Semaphore, domain: str):
    async with sem:
        response_class = {
            "finance": FinanceInfo, "academic": AcademicResearchInfo, 
            "biology": BiologyResearchInfo, "medical": MedicalResearchInfo,
            "technology": TechnologyResearchInfo, "legal": LegalDocumentInfo, 
            "patent": TechnologyPatentInfo, "education": EducationalMaterialInfo,
            "clinical": ClinicalGuidelinesInfo, "rules": RulesRegulationsInfo,
            "marketing": MarketingTrendInfo, "product": ProductInfo
        }.get(domain, None)

        # if response_class is not None: then make a variable to string all of the class attributes together
        if response_class is not None:
            response_class_attributes = response_class.model_fields.keys()
            response_class_attributes_string = "\n".join(response_class_attributes)
            print(response_class, ": \n",response_class_attributes_string, "\n\n")
            
        processed_response = await aclient.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_model=response_class,
            messages=[
                {"role": "system", "content": """
                    Given the domain and its response class: Identify keywords from the chunk_content and summarize in detail. If the response class is not specified, then adapt to the context of the information and provide a summary that captures the main points of the chunk_content. If it is about instructions, then provide a step-by-step guide. If you cannot identify anything, then NA or leave blank.
                """},
                {"role": "user", "content": f"domain: {domain}. response_class_attributes: {str(response_class_attributes_string)}. chunk_content: {chunk_content}"}
            ],
            seed=42,
        )
        return processed_response

@retry_decorator
async def process_chunk_content_instructor_hyde(chunk_content: str, sem: asyncio.Semaphore, domain: str, questions: List[str]):
    async with sem:
        response_class = {
            "finance": FinanceInfo, "academic": AcademicResearchInfo, 
            "biology": BiologyResearchInfo, "medical": MedicalResearchInfo,
            "technology": TechnologyResearchInfo, "legal": LegalDocumentInfo, 
            "patent": TechnologyPatentInfo, "education": EducationalMaterialInfo,
            "clinical": ClinicalGuidelinesInfo, "rules": RulesRegulationsInfo,
            "marketing": MarketingTrendInfo, "product": ProductInfo
        }.get(domain, None)

        # if response_class is not None: then make a variable to string all of the class attributes together
        if response_class is not None:
            response_class_attributes = response_class.model_fields.keys()
            response_class_attributes_string = "\n".join(response_class_attributes)
            print(response_class, ": \n",response_class_attributes_string, "\n\n")

        ic(questions)
            
        processed_response = await aclient.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_model=response_class,
            messages=[
                {"role": "system", "content": f"""
                    Given the domain: {domain}. questions: {str(questions)} response_class_attributes: {str(response_class_attributes_string)}.: Identify keywords from the chunk_content and summarize in detail. Make sure to answer the questions. If the response class is not specified, then adapt to the context of the information and provide a summary that captures the main points of the chunk_content. If it is about instructions, then provide a step-by-step guide. If you cannot identify anything, then NA or leave blank.
                """},
                {"role": "user", "content": f"chunk_content: {chunk_content}"}
            ],
            seed=42,
        )
        return processed_response

class Domain(BaseModel):
    domain: List[str]

@retry_decorator
async def run_domain_classifier(chunk_content: str):
    messages = [
        {"role": "system", "content": "Identify the most suitable domain(s) for the chunk content. Possible domains include: finance, academic, biology, medical, technology, legal, patent, education, clinical, rules, marketing, product, or none. More than one domain may be applicable."},
        {"role": "user", "content": f"Identify suitable domain(s) for the following chunk content: {chunk_content}"}
    ]
    domain = await aclient.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_model=Domain,
        messages=messages,
        seed=42,
    )
    return domain.domain

class DomainHyde(BaseModel):
    domain: List[str]
    questions: List[str]

@retry_decorator
async def run_domain_classifier_with_hyde(chunk_content: str):
    messages = [
        {"role": "system", "content": "Identify the most suitable domain(s) for the chunk content. Possible domains include: finance, academic, biology, medical, technology, legal, patent, education, clinical, rules, marketing, product, or none. More than one domain may be applicable. Generate the top relevant questions list about the content."},
        {"role": "user", "content": f"Identify suitable domain(s) and top questions for the following chunk content: {chunk_content}"}
    ]
    domain = await aclient.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_model=DomainHyde,
        messages=messages,
        seed=42,
    )
    return domain 

# Test 
test_chunk_content = """
January 25, 2024
Authors
OpenAI
Announcements
, 
Product
We are releasing new models, reducing prices for GPT-3.5 Turbo, and introducing new ways for developers to manage API keys and understand API usage. The new models include:

Two new embedding models
An updated GPT-4 Turbo preview model 
An updated GPT-3.5 Turbo model
An updated text moderation model
By default, data sent to the OpenAI API will not be used to train or improve OpenAI models.

New embedding models with lower pricing
We are introducing two new embedding models: a smaller and highly efficient text-embedding-3-small model, and a larger and more powerful text-embedding-3-large model.

An embedding is a sequence of numbers that represents the concepts within content such as natural language or code. Embeddings make it easy for machine learning models and other algorithms to understand the relationships between content and to perform tasks like clustering or retrieval. They power applications like knowledge retrieval in both ChatGPT and the Assistants API, and many retrieval augmented generation (RAG) developer tools.


A new small text embedding model
text-embedding-3-small is our new highly efficient embedding model and provides a significant upgrade over its predecessor, the text-embedding-ada-002 model released in December 2022. 

Stronger performance. Comparing text-embedding-ada-002 to text-embedding-3-small, the average score on a commonly used benchmark for multi-language retrieval (MIRACL) has increased from 31.4% to 44.0%, while the average score on a commonly used benchmark for English tasks (MTEB) has increased from 61.0% to 62.3%.

Reduced price. text-embedding-3-small is also substantially more efficient than our previous generation text-embedding-ada-002 model. Pricing for text-embedding-3-small has therefore been reduced by 5X compared to text-embedding-ada-002, from a price per 1k tokens of $0.0001 to $0.00002.

We are not deprecating text-embedding-ada-002, so while we recommend the newer model, customers are welcome to continue using the previous generation model.

A new large text embedding model: text-embedding-3-large

text-embedding-3-large is our new next generation larger embedding model and creates embeddings with up to 3072 dimensions.

Stronger performance. text-embedding-3-large is our new best performing model. Comparing text-embedding-ada-002 to text-embedding-3-large: on MIRACL, the average score has increased from 31.4% to 54.9%, while on MTEB, the average score has increased from 61.0% to 64.6%.

Eval benchmark

ada v2

text-embedding-3-small

text-embedding-3-large

MIRACL average

31.4

44.0

54.9

MTEB average

61.0

62.3

64.6

text-embedding-3-large will be priced at $0.00013 / 1k tokens.

You can learn more about using the new embedding models in our Embeddings guide.

Native support for shortening embeddings
Using larger embeddings, for example storing them in a vector store for retrieval, generally costs more and consumes more compute, memory and storage than using smaller embeddings.

Both of our new embedding models were trained with a technique that allows developers to trade-off performance and cost of using embeddings. Specifically, developers can shorten embeddings (i.e. remove some numbers from the end of the sequence) without the embedding losing its concept-representing properties by passing in the dimensions API parameter. For example, on the MTEB benchmark, a text-embedding-3-large embedding can be shortened to a size of 256 while still outperforming an unshortened text-embedding-ada-002 embedding with a size of 1536.

ada v2	text-embedding-3-small	text-embedding-3-large
Embedding size	1536	512	1536	256	1024	3072
Average MTEB score	61.0	61.6	62.3	62.0	64.1	64.6
This enables very flexible usage. For example, when using a vector data store that only supports embeddings up to 1024 dimensions long, developers can now still use our best embedding model text-embedding-3-large and specify a value of 1024 for the dimensions API parameter, which will shorten the embedding down from 3072 dimensions, trading off some accuracy in exchange for the smaller vector size.

Other new models and lower pricing
Updated GPT-3.5 Turbo model and lower pricing
Next week we are introducing a new GPT-3.5 Turbo model, gpt-4o-mini, and for the third time in the past year, we will be decreasing prices on GPT-3.5 Turbo to help our customers scale. Input prices for the new model are reduced by 50% to $0.0005 /1K tokens and output prices are reduced by 25% to $0.0015 /1K tokens. This model will also have various improvements including higher accuracy at responding in requested formats and a fix for a bug which caused a text encoding issue for non-English language function calls.

Customers using the pinned gpt-3.5-turbo model alias will be automatically upgraded from gpt-3.5-turbo-0613 to gpt-4o-mini two weeks after this model launches.

Updated GPT-4 Turbo preview
Over 70% of requests from GPT-4 API customers have transitioned to GPT-4 Turbo since its release, as developers take advantage of its updated knowledge cutoff, larger 128k context windows, and lower prices. 

Today, we are releasing an updated GPT-4 Turbo preview model, gpt-4-0125-preview. This model completes tasks like code generation more thoroughly than the previous preview model and is intended to reduce cases of “laziness” where the model doesn’t complete a task. The new model also includes the fix for the bug impacting non-English UTF-8 generations.

For those who want to be automatically upgraded to new GPT-4 Turbo preview versions, we are also introducing a new gpt-4-turbo-preview model name alias, which will always point to our latest GPT-4 Turbo preview model. 

We plan to launch GPT-4 Turbo with vision in general availability in the coming months.

Updated moderation model
The free Moderation API allows developers to identify potentially harmful text. As part of our ongoing safety work, we are releasing text-moderation-007, our most robust moderation model to-date. The text-moderation-latest and text-moderation-stable aliases have been updated to point to it. You can learn more about building safe AI systems through our safety best practices guide.

New ways to understand API usage and manage API keys
We are launching two platform improvements to give developers both more visibility into their usage and control over API keys.

First, developers can now assign permissions to API keys from the API keys page. For example, a key could be assigned read-only access to power an internal tracking dashboard, or restricted to only access certain endpoints.

Second, the usage dashboard and usage export function now expose metrics on an API key level after turning on tracking. This makes it simple to view usage on a per feature, team, product, or project level, simply by having separate API keys for each.

"""

@retry_decorator
async def classify_and_process_content(chunk_content: str):
    sem = asyncio.Semaphore(100)
    
    # Run domain classifier
    domain_result = await run_domain_classifier(chunk_content)
    ic(domain_result)
    
    # Process each domain concurrently and collect results
    tasks = [asyncio.create_task(process_chunk_content_instructor(chunk_content, sem, domain)) for domain in domain_result]
    results = await asyncio.gather(*tasks)
    
    for result in results:
        ic(result)

@retry_decorator
async def classify_question_and_process_content(chunk_content: str):
    sem = asyncio.Semaphore(100)

    # Run domain and question classifier
    domain_hyde_result = await run_domain_classifier_with_hyde(chunk_content)
    ic(domain_hyde_result)
    questions_list = domain_hyde_result.questions
    domain_result = domain_hyde_result.domain

    # use process_chunk_content_instructor_hyde(chunk_content, sem, domain, questions)
    tasks = [asyncio.create_task(process_chunk_content_instructor_hyde(chunk_content, sem, domain, questions_list)) for domain in domain_result]
    results = await asyncio.gather(*tasks)

    for result in results:
        ic(result)

# Test
# print("Testing classify_and_process_content() function")
# start_time = time.time()
# metadata_using_pydantic = asyncio.run(classify_and_process_content(test_chunk_content))
# print(f'time taken: {time.time() - start_time}')
"""
Time Taken: 
1. time taken: 24.39721703529358
2. time taken: 9.80324673652649
3. time taken: 8.409319162368774
4. time taken: 8.669508934020996
5. time taken: 13.20159387588501
6. time taken: 9.463521480560303
7. time taken: 15.349002838134766
8. time taken: 12.394219636917114
9. time taken: 7.461678981781006
"""

# print("Testing classify_and_process_content() function with Hyde")
# start_time = time.time()
# metadata_using_pydantic = asyncio.run(classify_question_and_process_content(test_chunk_content))
# print(f'time taken: {time.time() - start_time}')

"""
time taken: 14.440421104431152
"""

##############################################################################################################
##### OpenAI Chatbot #####
##############################################################################################################

from openai import OpenAI as OG_OpenAI

def process_in_default_mode(user_question):
    main_full_response = ""
    client = OG_OpenAI()

    message_placeholder = st.empty()
    responses = client.chat.completions.create(
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
            main_full_response += str(response.choices[0].delta.content)
            message_placeholder.markdown(f"**One Time Response**: {main_full_response}")
    st.markdown("---")
    return main_full_response
