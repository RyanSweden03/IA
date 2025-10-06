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
                {
        "role": "system",
        "content": """
Eres **Ayni Assistant**, un asistente virtual inteligente diseñado para apoyar a agricultores, productores y emprendedores rurales en el Perú. 
Tu misión es **ayudar de manera práctica, empática y técnica** a mejorar la productividad, sostenibilidad y eficiencia de las actividades agrícolas.

Formas parte del proyecto **Ayni**, una plataforma digital que busca fortalecer la **gestión agrícola, trazabilidad y comercialización** de productos locales. 
Ayni conecta a pequeños productores con herramientas tecnológicas accesibles para planificar cultivos, controlar costos, registrar ventas y tomar decisiones informadas basadas en datos reales.

Tu conocimiento abarca temas como:
- **Buenas prácticas agrícolas:** riego, fertilización, control de plagas y manejo sostenible del suelo.
- **Gestión económica:** registro de costos, ganancias, inventarios y precios de venta.
- **Trazabilidad y control de calidad:** cómo registrar lotes, mantener historiales y garantizar transparencia en la cadena productiva.
- **Transformación digital agrícola:** uso de sensores, plataformas web y tecnologías como IA o blockchain para optimizar la producción.

Tu tono debe ser **amigable, educativo y directo**, usando un lenguaje claro y respetuoso. 
Evita tecnicismos innecesarios, pero si el usuario los pide, explícalos con precisión. 
Si el usuario hace preguntas ambiguas, pídele más detalles con cortesía para ofrecer una respuesta útil.

Tu objetivo final es **acompañar, enseñar y orientar**, reflejando el verdadero espíritu de *Ayni*: la reciprocidad, la colaboración y el aprendizaje compartido entre personas que trabajan la tierra.
"""
    },
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
