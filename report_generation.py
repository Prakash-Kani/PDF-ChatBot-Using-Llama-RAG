from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough



model_name = 'llama3.1:latest'

embeddings_model_name = 'all-MiniLM-L6-v2'

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def create_retriever(course_name):
    from report_data_loader import data_loader
    data_loader(course_name)

    file_path = r"Reports/reports_data.csv"
    loader = CSVLoader(file_path, encoding="windows-1252")
    documents = loader.load()

    db = Chroma.from_documents(documents, embedding_function)

    retriever = db.as_retriever()
    return retriever

def create_chain(course_name):
    if course_name.lower() == 'all' or 'all' in course_name.lower():
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
    else:
        template = """
                    You are an intelligent assistant designed to generate comprehensive reports based on the provided context.

                    ### Instructions:
                    - If the user input is "generate the report", create a detailed report that includes the following sections:
                        1. **Frequently Asked Questions (FAQ)**: Summarize the most common questions asked by users.
                        2. **Identified Difficulties**: Highlight the topics or areas where users commonly face challenges or difficulties.   
                        3. **Sub Topic Coverage**: Provide an overview of the distribution of sub topics discussed by users.
                        4. **User Engagement**: Analyze user interaction patterns, such as the number of active users and frequency of interactions.
                        5. **Trend Analysis**: Identify how user questions and interactions evolve over time, revealing emerging topics or shifting interests.
                    

                    - For any other input, answer the question solely based on the provided context.

                    ### Context:
                    {context}

                    ### User Input:
                    {question}

                    ### Response:
                    """        
    prompt = ChatPromptTemplate.from_template(template)


    retriever = create_retriever(course_name=course_name)
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
