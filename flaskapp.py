from flask import Flask, request, jsonify
from Chat_Bot import *
from doc_loader import ingest
from report_generation import create_chain
from datetime import datetime as dt
import os
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


embeddings_model_name =  "all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
store ={}

def get_session_history1(session_id: str, ai_message) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
        # store[session_id].add_user_message("What is the title of the given context?")
        store[session_id].add_ai_message(ai_message)
    return store[session_id]
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
        # store[session_id].add_user_message("What is the title of the given context?")
        # store[session_id].add_ai_message(ai_message)
    return store[session_id]
def Conversational_Chain(filename):

    db = Chroma(persist_directory= filename, embedding_function=embeddings)

    retriever = db.as_retriever()
    rag_chain = RAG_Chain(retriever)
    return RunnableWithMessageHistory(
                                        rag_chain,
                                        get_session_history,
                                        input_messages_key="input",
                                        history_messages_key="chat_history",
                                        output_messages_key="answer",
                                    )


app = Flask(__name__)


UPLOAD_FOLDER = 'Uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


DB_FOLDER = 'Databases'
if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)
app.config['DB_FOLDER'] = DB_FOLDER



@app.route('/invoke', methods=['POST'])
def invoke_conversational_rag_chain():
    # Get the input data from the request
    data = request.get_json()

    # Ensure the input is provided
    if 'input' not in data:
        return jsonify({'error': 'No input provided'}), 400
    
    if 'filename' not in data:
        return jsonify({'error': 'No filename provided'}), 400

    # Set up the configuration, if any
    config = data.get('config', {})
    filename = data['filename']

    session_Id = config['session_id']
    text_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{filename}.txt')
    with open(text_file_path, 'r') as text_file:
        ai_message = text_file.read()

    if session_Id and ai_message:
        get_session_history1(session_Id, ai_message)


    db_path = os.path.join(app.config['DB_FOLDER'], filename)
    llm = Conversational_Chain(filename=db_path)

    # # Invoke the conversational RAG chain
    result = llm.invoke(
        {"input": data['input']},
        config=config
    )

    # Extract the answer from the result
    response = {'answer': result.get('answer'), 'time_stamp': dt.now()}

    # Return the response as JSON
    return jsonify(response)



@app.route('/ingest', methods=['POST'])
def ingest_pdf():
    # Ensure that 'filename' and 'pdf' are part of the form data
    if 'filename' not in request.form or 'pdf' not in request.files:
        return jsonify({'error': 'Filename or PDF file is missing'}), 400

    # Get the filename and PDF file from the request
    filename = request.form['filename']
    pdf_file = request.files['pdf']
    topic = request.form['topic']
    history = f"Let's dive into {topic}. What specific topic or question can I help you with today?"

    # Check if the file is a valid PDF
    if pdf_file.filename == '' or not pdf_file.filename.endswith('.pdf'):
        return jsonify({'error': 'Invalid file format. Please upload a PDF.'}), 400

    # Save the PDF file to the server
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf_file.save(file_path)

    if topic:
        text_file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{filename}.txt')
        with open(text_file_path, 'w') as text_file:
            text_file.write(history)


    persist_directory = os.path.join(app.config['DB_FOLDER'], filename)
    ingest(file_path=file_path, persist_directory = persist_directory)


    # Return success response
    return jsonify({'message': 'File uploaded and processed successfully'}), 200



@app.route('/report', methods=['POST'])
def report_generator():
    # Get the input data from the request
    data = request.get_json()

    # Ensure the input is provided
    if 'course_name' not in data:
        return jsonify({'error': 'No course_name provided'}), 400

    filename = data['course_name']
    if filename:
        chain = create_chain(course_name=filename)

        result = chain.invoke("generate the report")

        response = {'report': result, 'time_stamp': dt.now()}
        # Return the response as JSON
        return jsonify(response)
    else:
        return jsonify({'error': 'Invalid Course Name'}), 400

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000, debug=True)
