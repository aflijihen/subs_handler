import paho.mqtt.client as mqtt
import json
import time
from langchain.chat_models.openai import ChatOpenAI
from datahandler import DataHandler
from langchain_community.document_loaders import DirectoryLoader,Docx2txtLoader
import docx2txt

class Subscriber:
    def __init__(self, broker_address, topic,data_handler, llm_model):
        self.broker_address = broker_address
        self.topic = topic
        
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.data_handler = DataHandler()
        self.llm_model = llm_model
        self.processing_message = False 
        
      

    def start(self):
        self.client.connect(self.broker_address)
       
        self.client.loop_start()

    def on_connect(self, client, userdata, flags, rc):
        print("Connected to MQTT broker with result code " + str(rc))
        self.client.subscribe(self.topic)

    def on_message(self, client, userdata, msg):
         if not self.processing_message:  # Vérifier si le message est déjà en cours de traitement
            self.processing_message = True 
            # Marquer le message comme étant traité
            data = json.loads(msg.payload.decode())
            print("Received data:", data) # Process the received data using DataHandler
            data_handler_instance.execute()
           
                                        
           
            recommendations = self.data_handler.process_received_data(data)
           
            print("Recommendations:", recommendations)
            # Convertir les données JSON en texte
           
            response = self.llm_model.execute(data, recommendations)
            print("Response from LLM model:", response)
            self.send_data(response)
         
                            
                

    def send_data(self, data):
        
        # Send data to MQTT topic
        self.client.publish(self.topic, json.dumps(data))
        
class MyLLM(ChatOpenAI):
    def __init__(self, api_key, temperature=0.6):
        super().__init__(api_key=api_key, temperature=temperature)

    def execute(self, data, recommendations):

        response_data = {
            "recommendations": recommendations,
            "processed_data": data,
            "additional_info": "Response from LLM model"
        }
         
        
        return response_data

# Exemple d'utilisation
if __name__ == "__main__":
    data_handler_instance = DataHandler()
    llm_model_instance=MyLLM(api_key="",temperature=0.6)
    subscriber = Subscriber(broker_address="mqtt.eclipseprojects.io", topic="Spirulina_Edge",
                            data_handler=data_handler_instance,llm_model=llm_model_instance )
    subscriber.client.on_connect=subscriber.on_connect
    subscriber.client.on_message = subscriber.on_message
    subscriber.start()
    
   
    try:
        # Keep the subscriber running for 2 hours (7200 seconds)
        time.sleep(60)
    except KeyboardInterrupt:
        # Stop the subscriber if interrupted by keyboard
        subscriber.stop()


   
