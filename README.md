## Parsely Business Model 3.23.24
# If you are an Individual/Small Teams that are looking for the following services:
1. You have access to all of the exclusive features for $50 per month + token costs given your API key, catered to "RAG over all your uploaded documents", "Excel Tools", "Live Transcriptions and Insights", "Recommendation Bot Deployment", "Phone-Call Booking Agent via Retell.AI" and more below 🤖
2. You have opportunity to pay an additional $500 per month for a consulting session on a project of your choice (Limited to 1 hours per session, 5 sessions max)
3. You have opportunity to pay an additional $1500 for a draft 1 working prototype/demo in 1 month, your choices of LLM RAG Tool (ie. such as a feature to automate your database/excel to post automatically)
4. With the publication of your idea turned to prototype, and with > 200 likes, your project will be suitable for a profit sharing program. (Ironing out details)

# $20 Subscription:
1. You have full access to the top three tools: "RAG over all your uploaded documents", "Excel Tools", and "Live Transcriptions and Insights", you can also experiment with the "Recommendation Bot Deployment", "Phone-Call Booking Agent via Retell.AI" on playground, deployment costs $50 per model/pipeline. Requires your OpenAI API key for token costs, rate limit and speed is based on your API Key Tiering.

# Free Tier:
1. You have limited access to top three tools: "RAG over all your uploaded documents", "Excel Tools", and "Live Transcriptions and Insights". Requires your OpenAI API key for token costs, rate limit and speed is based on your API Key Tiering.

# rag-a-thon with Llama-Index

2.6.24 Update:

Parsely supports 20+ file types, easily answering multiple questions over a wide diversity of file contents accurately. 

Data is stored in two ways: 
(1) User session based, which uses the session state to store all data. (Finished) 
(2) User database based, which uses Cloud Vector DBs to help customers deploy their personal product recommendation and booking agency and personal smart data retrieval vault. (WIP)

Looking for Cloud VDB's credit sponsorships: The credits will be used to help host the alpha test users' data for a feature called BYOB. A unique link will be generated using the user's cloud api key and collection name to allow the test user to deploy their shop sales consultant and booking agent using their product database and business document.

People are likely going to want to have control over which database they want to use (ie Qdrant vs Vectara vs Astra vs so many others) so that they own their own contents. Parsely make professionals who don't know how to code to have a leverage over people who could code their own personal assistance over all of their data. - Plus, they get to save their money on hundred thousands/millions of dollars over their own tech teams for the less "important agenda" projects when it comes to data transformation and retrieval and augmentation.

So far, our use cases have saved at least $12k to $50k per project, for 4 different projects. The cost was trivial in comparison, less than $200 cost throughout a three months scope. 

 
Use Cases Explored:
1. All Files Async Parallelized Processing
2. Our own querying method that involved sparse dense retrieval + parallel request merge responses + instructor pydantic class usage for raw text data transformation and transmission
3. Llama_index llama parser
4. Vectara Query
5. APIs deployed for fast and accurate parsing for SUPPORTED_EXTENSIONS =
[
    ".docx", ".doc", ".odt", ".pptx", ".ppt", ".xlsx", ".csv", ".tsv", ".eml", ".msg",
    ".rtf", ".epub", ".html", ".xml", ".pdf", ".png", ".jpg", ".jpeg", ".txt"
]

It works well for travel itinerary retrieva, meeting note retrieval, customer solution and product recommendation
The speed is faster than that of llama parser when scaled to dozens and hundreds of documents while maintaining precision in responses
Only key issue is complex table, which llama parser integration helped to ease on when the llama parser recursive retrieval result is integrated along with the input prompt.

Vectara integration was consuing at first with the three IDs, but eventually figured out.
BentoML integration was under going deployment, easy to use, love the interface.
