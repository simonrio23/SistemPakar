import os
from flask import Flask, request, render_template, jsonify, redirect, url_for
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Konfigurasi keamanan
app.config['SECRET_KEY'] = os.urandom(24)

# Fungsi inisialisasi model dan dokumen
def initialize_chatbot():
    try:
        # Load dan proses PDF
        loader = PyPDFLoader("Dataset.pdf")
        data = loader.load()
        
        # Pisah dokumen menjadi chunk
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(data)

        # Buat vector store
        vectorstore = Chroma.from_documents(
            documents=docs, 
            embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        )

        # Buat retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 5}
        )

        # Inisialisasi LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0.3, 
            max_tokens=None
        )

        # Template sistem
        system_template = (
            "Anda adalah asisten AI khusus untuk mendiagnosa penyakit pada bauh mangga. "
            "Gunakan konteks yang diberikan untuk menjawab pertanyaan dengan akurat, singkat dan detail. "
            "Berikan solusi praktis dan tidak terlalu menggunakan penjelasan yang panjang dan aman untuk setiap permasalahan. "
            "Jika informasi tidak tersedia, sampaikan dengan jelas dan tidak terlalu panjang. "
            "Fokus pada keselamatan dan solusi penyembuhan buah mangga. "
            "\n\nKonteks: {context}"
        )

        # Buat prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        # Buat document chain
        document_chain = create_stuff_documents_chain(llm, prompt)

        # Buat retrieval chain
        retrieval_chain = create_retrieval_chain(
            retriever, 
            document_chain
        )

        return retrieval_chain

    except Exception as e:
        print(f"Kesalahan inisialisasi: {e}")
        return None

# Inisialisasi chatbot global
CHATBOT = initialize_chatbot()

# Memori percakapan
conversation_memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/clear', methods=['GET', 'POST'])
def clear_chat():
    conversation_memory.chat_memory.messages = []
    return redirect(url_for('chatbot'))

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        try:
            # Ambil query dari form
            query = request.form.get('query', '')

            if not query:
                return jsonify({
                    'status': 'error', 
                    'message': 'Query tidak boleh kosong'
                })

            # Proses dengan retrieval chain
            result = CHATBOT.invoke({
                "input": query,
                "chat_history": conversation_memory.chat_memory.messages
            })

            # Simpan ke memori percakapan
            conversation_memory.save_context(
                {"input": query}, 
                {"output": result['answer']}
            )

            return render_template('chatbot.html', 
                query=query, 
                answer=result['answer'],
                chat_history=conversation_memory.chat_memory.messages
            )

        except Exception as e:
            print(f"Kesalahan dalam chatbot: {e}")
            return render_template('chatbot.html', 
                query=query, 
                answer="Maaf, terjadi kesalahan dalam memproses pertanyaan."
            )
    
    return render_template('chatbot.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(
        host='0.0.0.0', 
        port=5000, 
        debug=True
    )