from pymongo import MongoClient
import pandas as pd

from langchain.document_loaders import CSVLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

model_name = 'llama3.1:latest'

embeddings_model_name = 'all-MiniLM-L6-v2'

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def retriever_():
    collection = MongoClient('mongodb+srv://meynikaratest:devtest123@cluster0.ny2uk.mongodb.net/').imaginx.chats
    lst=[]
    for data in collection.find({}, {'_id':0}):
        lst.append(data)
    df = pd.DataFrame(lst)
    df[df.columns[0:4]].to_csv(r'reports/reports_data.csv')

    file_path = r"reports/reports_data.csv"
    loader = CSVLoader(file_path, encoding="windows-1252")
    documents = loader.load()

    db = Chroma.from_documents(documents, embedding_function)

    return db.as_retriever()

def create_chain():
    template = """
                You are an intelligent assistant designed to generate comprehensive reports based on the provided context.

                ### Instructions:
                - If the user input is "generate the report", create a detailed report that includes the following sections:
                    1. **Frequently Asked Questions (FAQ)**: Summarize the most common questions asked by users.
                    2. **Identified Difficulties**: Highlight the topics or areas where users commonly face challenges or difficulties.
                    3. **Topic Coverage**: Provide an overview of the distribution of topics discussed by users.
                    4. **Sub Topic Coverage**: Provide an overview of the distribution of sub topics discussed by users.
                    5. **User Engagement**: Analyze user interaction patterns, such as the number of active users and frequency of interactions.
                    6. **Trend Analysis**: Identify how user questions and interactions evolve over time, revealing emerging topics or shifting interests.
                

                - For any other input, answer the question solely based on the provided context.

                ### Context:
                {context}

                ### User Input:
                {question}

                ### Response:
                """
    prompt = ChatPromptTemplate.from_template(template)


    retriever = retriever_()
    model = Ollama(model = model_name)
    # Set up the chain
    chain = (
                {
                    "context": retriever,
                    "question": RunnablePassthrough()
                }
                | prompt
                | model
                | StrOutputParser()
            )
    return chain
chain = create_chain()
print(chain.invoke("generate the report"))