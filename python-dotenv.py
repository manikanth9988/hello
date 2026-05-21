from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
load_dotenv()
ChatGroq.api_key = os.getenv("API")
