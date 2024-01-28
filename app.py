import os
import google.generativeai as genai
import google.ai.generativelanguage as glm
from langchain.vectorstores import Chroma
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st

st.title("DocsGPT")

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

rag = glm.Tool(
    function_declarations=[
      glm.FunctionDeclaration(
        name='vector_search',
        description="Returns the content of the document user attached. Make sure that your not passing query as a question use like keywords instead. Use this function to search for contents in the user attached or uploaded documents to you.",
        parameters=glm.Schema(
            type=glm.Type.OBJECT,
            properties={
                'query': glm.Schema(type=glm.Type.STRING),
            },
            required=['query']
        )
      )
    ]
)

gemini = genai.GenerativeModel('gemini-pro', tools=[rag])

class rawkn:
    def __init__(self, text):
        self.text = text
    def get_relevant_documents(self, query):
        return self.text

def loader_data(files):
    file_type = files[0].type
    total_content = ''
    for file in files:
        if file_type == "application/pdf":
            pdf_reader = PdfReader(file)
            content = ''
            for page in pdf_reader.pages:
                content += page.extract_text()
        if file_type == "text/plain":
            content = file.read()
            content = content.decode("utf-8")
        total_content += content
        print(total_content)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    texts = text_splitter.split_text(total_content)
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        vector_store = Chroma.from_texts(texts, embeddings).as_retriever()
        st.session_state.knowledge = vector_store
    except:
        st.session_state.knowledge = rawkn(total_content)

if "history" not in st.session_state:
    st.session_state.history = []
if "knowledge" not in st.session_state:
    st.session_state.knowledge = None
if "chat" not in st.session_state:
    st.session_state.chat = gemini.start_chat(history=[])

for history in st.session_state.history:
    with st.chat_message(history["role"]):
        st.markdown(history["text"])

with st.sidebar:
    st.title("Knowledge")
    st.markdown("""### Tips to use DocsGPT:
- Upload your documents [pdf, txt] to DocsGPT and make sure to click on the process button.
- wait for a second and then start chatting with DocsGPT.
- While asking questions to DocsGPT about your uploaded files, please refer your uploaded files as *Document*, *Docs*, *attached or uploaded docs*, so the model can easily understands what you are referring to.""")
    files = st.file_uploader("Upload a file", accept_multiple_files=True, type=["pdf", "txt"])
    process = st.button("Process")
    if process and files:
        st.session_state.knowledge = loader_data(files)
        with st.spinner('loading your file. This may take a while...'):
            loader_data(files)
    
            
if prompt := st.chat_input("Enter your message..."):
    st.session_state.history.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = st.session_state.chat.send_message(prompt)
        if response.candidates[0].content.parts[0].text == '':
            args = response.candidates[0].content.parts[0].function_call.args['query']
            if st.session_state.knowledge is not None:
                related_docs = str(st.session_state.knowledge.get_relevant_documents(args))
            else:
                related_docs = 'No knowledge documents loaded'
            response = st.session_state.chat.send_message(
                glm.Content(
                    parts=[glm.Part(
                        function_response = glm.FunctionResponse(
                        name='vector_search',
                        response={'rag': related_docs},
                        )
                    )]
                )
            ).text
        else:
            response = response.candidates[0].content.parts[0].text
        print(st.session_state.chat.history)
        message_placeholder.markdown(response)
    st.session_state.history.append({"role": "assistant", "text": response})
