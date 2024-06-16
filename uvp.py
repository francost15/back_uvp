import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import re
from dotenv import load_dotenv

# Cargar variables de entorno desde un archivo .env
load_dotenv()

# Leer las claves API y otros parámetros sensibles desde las variables de entorno
api_key = os.getenv("OPENAI_API_KEY")
assistant_id = os.getenv("OPENAI_ASSISTANT_ID")
vector_store_id = os.getenv("OPENAI_VECTOR_STORE_ID")

# Inicializar el cliente de OpenAI
client = openai.Client(api_key=api_key)

# Actualizar el Assistant para usar la base de datos vectorizada
assistant = client.beta.assistants.update(
    assistant_id=assistant_id,
    tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
)

class EventHandler(openai.AssistantEventHandler):
    def __init__(self):
        super().__init__()
        self.response = None

    def on_message_done(self, message):
        message_content = message.content[0].text.value
        # Eliminar las citas del texto generado usando una expresión regular
        cleaned_message = re.sub(r"【\d+:\d+†source】", "", message_content)
        self.response = cleaned_message

class Query(BaseModel):
    query: str

app = FastAPI()

@app.post("/uvp/")
async def interact(query: Query):
    try:
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": query.query
                }
            ],
            tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
        )
        print(thread)  # Imprimir el objeto thread en la consola para depuración

        event_handler = EventHandler()
        with client.beta.threads.runs.stream(
            thread_id=thread.id,
            assistant_id=assistant_id,
            event_handler=event_handler,
        ) as stream:
            stream.until_done()
        
        if event_handler.response is None:
            raise HTTPException(status_code=500, detail="No response from assistant")
        
        return {"response": event_handler.response}
    except openai.OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "API de Asistente UVP en funcionamiento"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
