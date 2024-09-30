from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

model_name = 'llama3.1:latest'
embeddings_model_name =  "all-MiniLM-L6-v2"

model = Ollama(model = model_name)


store = {}

def History_Chain(retriever):
    contextualize_q_system_prompt = (
                                        "Given a chat history and the latest user question "
                                        "which might reference context in the chat history, "
                                        "formulate a standalone question which can be understood "
                                        "without the chat history. Do NOT answer the question, "
                                        "just reformulate it if needed and otherwise return it as is."
                                    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
                                                                [   
                                                                    ("system", contextualize_q_system_prompt),
                                                                    MessagesPlaceholder("chat_history"),
                                                                    ("human", "{input}"),
                                                                ]
                                                            )
    history_aware_retriever = create_history_aware_retriever(model,
                                                             retriever, 
                                                             contextualize_q_prompt
                                                            )
    return history_aware_retriever


def Question_Answer_Chain():


    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "If the context does not contain the answer, respond with 'The information is not available in the provided context.'"
        # "The user is greeting you. Respond appropriately as a friendly assistant."
        "\n\n"
        "{context}"
    )   
    qa_prompt = ChatPromptTemplate.from_messages(
                                                    [
                                                        ("system", system_prompt),
                                                        MessagesPlaceholder("chat_history"),
                                                        ("human", "{input}"),
                                                    ]
                                                )
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    return question_answer_chain

def RAG_Chain(retriever):
    history_aware_retriever = History_Chain(retriever)
    question_answer_chain = Question_Answer_Chain()
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
        # store[session_id].add_user_message("What is the title of the given context?")
        store[session_id].add_ai_message("Let's dive into chemical bonding and molecular structure. What specific topic or question can I help you with today?")
    return store[session_id]

def Conversational_Chain():
    rag_chain = RAG_Chain()
    return RunnableWithMessageHistory(
                                        rag_chain,
                                        get_session_history,
                                        input_messages_key="input",
                                        history_messages_key="chat_history",
                                        output_messages_key="answer",
                                    )


