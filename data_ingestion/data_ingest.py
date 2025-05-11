from langchain_astradb import AstraDBVectorStore
from dotenv import load_dotenv
import os 
import pandas as pd 
from data_ingestion.data_transform import data_converter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_KEYSTAPCE = os.getenv("ASTRA_DB_KEYSPACE")

os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
os.environ['ASTRA_DB_API_ENDPOINT'] = ASTRA_DB_API_ENDPOINT
os.environ['ASTRA_DB_APPLICATION_TOKEN'] = ASTRA_DB_APPLICATION_TOKEN
os.environ['ASTRA_DB_KEYSTAPCE'] = ASTRA_DB_KEYSTAPCE

class ingest_data:
    def __init__(self):
        print("data ingestion class has init..") 
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.data_converter = data_converter()
    
    def data_ingestion(self, status):
        vstore = AstraDBVectorStore(
            embedding=self.embeddings,
            collection_name="chatbotecomm",
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN,
            namespace=ASTRA_DB_KEYSTAPCE
        )
        
        storage = status
        
        if storage == None:
            docs = self.data_converter.data_transformation()
            inserted_ids = vstore.from_documents(documents=docs, collection_name="chatbotecomm", embedding=self.embeddings)
            print(inserted_ids)
        else:
            return vstore
        
        return vstore
    
    
if __name__ == "__main__":
    ingest_data = ingest_data()
    # vstore, inserted_id = ingest_data.data_ingestion(status=None)
    vstore = ingest_data.data_ingestion(status="Not none")
    
    
    # print(f"\nInserted id : {inserted_id}, \nVstore: {vstore}")
    
    results = vstore.similarity_search("can you tell me the low budget headphone")
    for result in results:
        print(f"{result.page_content}, {result.metadata}")