from langchain.text_splitter import RecursiveCharacterTextSplitter
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
def text_splitter(data, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks    
def watsonx_embedding():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }
    
    watsonx_embedding = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params,
    )
    return watsonx_embedding
loader = TextLoader("new_Policies.txt")
txt_data = loader.load()
chunks_txt = text_splitter(txt_data, 200, 20)
vectordb = Chroma.from_documents(chunks_txt, watsonx_embedding())
query = "Email policy"
retriever = vectordb.as_retriever(search_kwargs={"k": 2})
docs = retriever.invoke(query)
print(docs)