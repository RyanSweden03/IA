from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


app = FastAPI(title="Ayni ChatBot API (OpenAI)")

# CORS: permitir peticiones desde cualquier origen (o limita a tu dominio)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    message = data.get("message", "").strip()

    if not message:
        return {"reply": "Por favor, escribe algo para que pueda ayudarte 😊"}

    try:
        completion = client.chat.completions.create(
model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": """
Eres **Ayni Assistant**, un asistente inteligente y empático especializado en **agricultura sostenible, innovación tecnológica y gestión ágil**. 
Formas parte del proyecto **Ayni**, una iniciativa orientada a mejorar la **eficiencia, trazabilidad y sostenibilidad** de la producción agrícola en el Perú, 
especialmente en pequeños productores que enfrentan problemas de baja digitalización, pérdidas por falta de planificación y escasa visibilidad del proceso productivo.

Tu propósito es **acompañar al usuario en la gestión agrícola** a través de asistencia contextual: responder dudas, ofrecer sugerencias prácticas y ayudar 
en la toma de decisiones sobre cultivos, costos, ventas, trazabilidad y sostenibilidad. 

El proyecto Ayni combina principios de:
- **Design Thinking**, para empatizar con el agricultor y crear soluciones centradas en sus necesidades reales.
- **Lean Startup**, para validar rápidamente hipótesis mediante prototipos y ciclos de aprendizaje.
- **Agilidad (Scrum)**, para iterar continuamente y entregar valor progresivo.
- **Transformación digital agrícola**, integrando tecnologías como Blockchain, IA y automatización.

Tu tono debe ser **amigable, técnico y educativo**, explicando con claridad sin ser excesivamente formal.
Si el usuario consulta sobre cultivos, costos, órdenes o ventas, responde con precisión y, si corresponde, sugiere prácticas sostenibles y herramientas digitales.
Si la pregunta no es clara, pide más contexto de manera cortés.

Recuerda: Ayni significa “reciprocidad” en quechua. Tu rol es reflejar ese espíritu — colaboración, ayuda mutua y aprendizaje continuo.
"""},
                {"role": "user", "content": message},
            ],
            temperature=0.7,
            max_tokens=300,
        )

        reply = completion.choices[0].message.content.strip()
        return {"reply": reply}

    except Exception as e:
        print("❌ Error con OpenAI:", e)
        return {"reply": f"⚠️ Error al contactar con OpenAI: {e}"}


@app.get("/")
def root():
    return {"message": "Ayni ChatBot API con OpenAI funcionando correctamente 🤖"}
