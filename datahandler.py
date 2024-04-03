import os
import json
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
import docx2txt



ArdTemperatureValue = 5
ArdphValue = 5
Water_Level = 5
ArdConductivityValue = 5
brightness = 5
 
# Define the application prompt #utiliser les valeurs actuelles : temperature:34, Ph_value:12, water_level:30, conductivity:20, brightness:10, donner des recommandations spécifiques à chaque mesure

combined_prompt = """
Utilisez les valeurs actuelles :
- Température : {ArdTemperatureValue} degrés Celsius
- Ph : {ArdphValue}
- Niveau d'eau : {Water_Level}
- Conductivité : {ArdConductivityValue}
- brightnss: {brightness}
,donner des recommandations spécifiques à chaque mesure.
"""
user_input = f"temperature={ArdTemperatureValue};Ph_value={ArdphValue};water_level={Water_Level};conductivity={ArdConductivityValue};brightness={brightness}"

prompt = combined_prompt.format(
    ArdTemperatureValue=ArdTemperatureValue,
    ArdphValue=ArdphValue,
    Water_Level=Water_Level,
    ArdConductivityValue=ArdConductivityValue,
    brightness=brightness
)
   
class DataHandler:
    def __init__(self):
        self.docs_dir = "./handbook/"
        self.persist_dir = "./handbook_faiss"
 
        self.embedding =SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
        self.received_data = None
        self.qa_chain = None
        
        
        
       # Load or build the FAISS index
        self.load_or_build_faiss()

        # Initialize the LLM model and conversation retrieval chain
        self.llm = ChatOpenAI(
            api_key='',
            temperature=0.6
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.memory.load_memory_variables({})
        self.load_or_build_faiss()
        self.initialize_qa_chain()

    def load_or_build_faiss(self):
        if os.path.exists(self.persist_dir):
            print(f"Loading FAISS index from {self.persist_dir}")
            self.vectorstore = FAISS.load_local(self.persist_dir, self.embedding,
                                                allow_dangerous_deserialization=True)
            print("Done.")
        else:
            print(f"Building FAISS index from documents in {self.docs_dir}")
            # Code for building FAISS index goes here
            pass

    def initialize_qa_chain(self):
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            memory=self.memory,
            retriever=self.vectorstore.as_retriever()
        )

    def process_received_data(self, data):
        self.received_data = data
        print("Received data:", self.received_data)
       
        recommendations = self.generate_recommendations(data)
        return  recommendations

   

    def generate_recommendations(self, data):
        # Créer le modèle de prompt spécifique à chaque mesure
        if "temperature" in data:
            temperature_prompt = self.combined_prompt.format(ArdTemperatureValue=data["temperature"],
                                                              ArdphValue=ArdphValue,
                                                              Water_Level=Water_Level,
                                                              ArdConductivityValue=ArdConductivityValue,
                                                              brightness=brightness)
            prompt = PromptTemplate(input_variables=["user_input"], temparature=temperature_prompt)
        elif "Ph_value" in data:
            ph_prompt = self.combined_prompt.format(ArdTemperatureValue=ArdTemperatureValue,
                                                    ArdphValue=data["Ph_value"],
                                                    Water_Level=Water_Level,
                                                    ArdConductivityValue=ArdConductivityValue,
                                                    brightness=brightness)
            prompt = PromptTemplate(input_variables=["user_input"], ph=ph_prompt)
        elif "water_level"in data: 
            water_prompt = self.combined_prompt.format(ArdTemperatureValue=ArdTemperatureValue,
                                                    ArdphValue=data,
                                                    Water_Level=Water_Level["water_level"],
                                                    ArdConductivityValue=ArdConductivityValue,
                                                    brightness=brightness)
            prompt = PromptTemplate(input_variables=["user_input"], water_level=water_prompt)
        elif "conductivity"in data:
            conductivity_prompt=self.combined_prompt.format(ArdTemperatureValue=ArdTemperatureValue,
                                                    ArdphValue=data,
                                                    Water_Level=Water_Level,
                                                    ArdConductivityValue=ArdConductivityValue["conductivity"],
                                                    brightness=brightness)
            prompt = PromptTemplate(input_variables=["user_input"], conductivity=conductivity_prompt)
        elif "brightness"in data:
            self.combined_prompt.format(ArdTemperatureValue=ArdTemperatureValue,
                                                    ArdphValue=data,
                                                    Water_Level=Water_Level,
                                                    ArdConductivityValue=ArdConductivityValue,
                                                    brightness=brightness["brightness"])
            brightness_prompt = PromptTemplate(input_variables=["user_input"], conductivity=brightness_prompt)
        else:
            prompt = PromptTemplate(input_variables=["user_input"], ph="")
            

        # Utiliser le modèle de prompt pour générer la requête
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"user_input": user_input})["output"]
        return response
    

    def execute(self):
        while True:
            user_input = input("Question:.\n>")
            if user_input == "exit":
                break
            else:
                result = self.qa_chain.invoke({"question": user_input})
                response = result["answer"]
                print("Recommendations:", response)

# Main entry point
if __name__ == "__main__":
    main = DataHandler()
    main.execute()
