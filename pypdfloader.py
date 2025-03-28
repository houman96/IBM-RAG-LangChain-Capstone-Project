from pprint import pprint
import json
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredFileLoader

pdf_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/WgM1DaUn2SYPcCg_It57tA/A-Comprehensive-Review-of-Low-Rank-Adaptation-in-Large-Language-Models-for-Efficient-Parameter-Tuning-1.pdf"

loader = PyPDFLoader(pdf_url)

pages = loader.load_and_split()

print(pages[0])