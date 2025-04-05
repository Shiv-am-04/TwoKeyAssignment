import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from groq import Groq
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader,Docx2txtLoader,TextLoader
from langchain.document_loaders import UnstructuredExcelLoader,UnstructuredPowerPointLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.schema import HumanMessage,AIMessage
import tempfile
import base64


load_dotenv()


groq_api_key = os.getenv('GROQ_API_KEY')

llm = ChatGroq(model='deepseek-r1-distill-llama-70b',api_key=groq_api_key,temprature=0.6)


st.set_page_config(page_title='AskMe')


def transcribe_audio(file_path):
    client = Groq()

    prompt = '''
    You are a highly skilled transcriptionist with over 15 years of experience in converting audio into text.
    Make sure all the words in the audio should be retained in the text provided.
    '''

    with open(file_path,'rb') as f:
        try:
            output = client.audio.transcriptions.create(
                file=(file_path,f.read()),
                model = 'whisper-large-v3-turbo',
                prompt = prompt,
                response_format='text',
                temperature=0.8
            )
        except Exception:
            return False
    
    return output


prompt_template = """
    Summarize the following document while preserving its key ideas, main insights, and important data points. 
    Ensure that the summary maintains the original context and tone.
    Document:{text}
    """

prompt = PromptTemplate.from_template(prompt_template)

refine_template = (
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
)

refine_prompt = PromptTemplate.from_template(refine_template)

summarize_chain = load_summarize_chain(
    llm=llm,
    chain_type="refine",
    question_prompt=prompt,
    refine_prompt=refine_prompt,
    return_intermediate_steps=True,
    input_key="input_document",
    output_key="output_text",
)


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def summarize_image(image_path,extension):
    encoded_image = encode_image(image_path)
    client = Groq()
   
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "analyze image and provide a concise summary that captures its main part in the image."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{extension};base64,{encoded_image}",
                            },
                        },
                    ],
                }
            ],
            model='llama-3.2-11b-vision-preview')
    except Exception:
        return False
    
    summary = chat_completion.choices[0].message.content

    return summary

def process_file(file_path,file_extension):
    
    # Load content based on file type
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension == ".docx":
        loader = Docx2txtLoader(file_path)
    elif file_extension == ".txt":
        loader = TextLoader(file_path)
    elif file_extension == ".xlsx":
        loader = UnstructuredExcelLoader(file_path)
    elif file_extension == ".pptx":
        loader = UnstructuredPowerPointLoader(file_path)
    else:
        return "Unsupported file format!"
    
    # Load documents from the file
    documents = loader.load()

    return documents



audio_extensions = ['.mp3','.wav','.m4a','.aac','.flac']
visual_extension = ['.jpg','.png','.jpeg','.svg','.gif','.webp','.tif','.tiff']

file = st.file_uploader(label='upload file')

if file:
    extension = os.path.splitext(file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name  
        print(temp_file_path)

        if extension in audio_extensions:
            output = transcribe_audio(temp_file_path)
            summary = summarize_chain.invoke({'input_document':[Document(page_content=output)]})
            if summary == False:
                st.error("something went wrong")
            else:    
                st.success(summary['output_text'])
        elif extension in visual_extension:
            summary = summarize_image(temp_file_path,extension)
            if summary == False:
                st.error("something went wrong")
            else:    
                st.success(summary)
        else:
            doc = process_file(temp_file_path,extension)
            try:
                summary = summarize_chain.invoke({'input_document':doc})
                st.success(summary['output_text'])
            except Exception:
                st.error("something went wrong")


#### creating chat interface ####

chat_prompt = ChatPromptTemplate.from_template(
    '''
    Using the provided context {doc}, please formulate a response that addresses the user's question. 
    While taking {doc} into account, ensure that your response is not solely reliant on it; supplement your answer with your own knowledge as needed. 
    The answer should be concise—neither too long nor too short—aiming for clarity and relevance. It should be less than 200 words
    '''
)

chain = chat_prompt|llm

msg = StreamlitChatMessageHistory(key='messages')

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id : msg,
    input_messages_key='question',
    history_messages_key='history'
)


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {'role':'ai','content':'Ask Query Related to Uploaded file'}
    ]


for message in st.session_state.messages:
    if isinstance(message,dict):
        role = message.get('role')
        res = message.get('content','No Content')
    elif isinstance(message,HumanMessage):
        role = 'user'
        res = message.content
    elif isinstance(message,AIMessage):
        role = 'ai'
        res = message.content
    with st.chat_message(role):
        st.write(res)

query = st.chat_input(placeholder='Ask Query Related to Uploaded file')

if query:
    st.session_state.messages.append({'role':'user','content':query})
    st.chat_message('user').write(query)

    with st.chat_message('ai'):
        config = {'configurable':{'session_id':'a'}}
        try:
            response = chain_with_history.invoke({'question':query,'doc':summary},config=config)
            st.session_state.messages.append({'role':'ai','content':response.content})
            st.write(response.content)
        except Exception:
            st.error("something went wrong")
