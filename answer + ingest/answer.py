import os
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from pydantic import SecretStr
