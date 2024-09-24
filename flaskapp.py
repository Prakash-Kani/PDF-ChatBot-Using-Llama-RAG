from flask import Flask, request, jsonify
from chatbot import Conversational_Chain
import os

llm  = Conversational_Chain()
app = Flask(__name__)


UPLOAD_FOLDER = './Uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




@app.route('/invoke', methods=['POST'])
def invoke_conversational_rag_chain():
    # Get the input data from the request
    data = request.get_json()

    # Ensure the input is provided
    if 'input' not in data:
        return jsonify({'error': 'No input provided'}), 400

    # Set up the configuration, if any
    config = data.get('config', {})

    # Invoke the conversational RAG chain
    result = llm.invoke(
        {"input": data['input']},
        config=config
    )

    # Extract the answer from the result
    response = {'answer': result.get('answer')}

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

    # Check if the file is a valid PDF
    if pdf_file.filename == '' or not pdf_file.filename.endswith('.pdf'):
        return jsonify({'error': 'Invalid file format. Please upload a PDF.'}), 400

    # Save the PDF file to the server
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf_file.save(file_path)
    print(file_path, filename)

    # Process the PDF (if you need to pass the file to the chatbot)
    # Example: llm.ingest(file_path)  # If needed for RAG model ingestion

    # Return success response
    return jsonify({'message': 'File uploaded and processed successfully'}), 200

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000, debug=True)
