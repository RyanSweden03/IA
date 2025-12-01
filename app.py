from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os

from dotenv import load_dotenv
from openai import OpenAI

from sqlalchemy import create_engine, text

load_dotenv()

app = FastAPI(title="Ayni ChatBot API (OpenAI + BD)")

# ===========================
#   CORS
# ===========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # luego puedes limitarlo a tu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL no está configurada")


engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# ===========================
#   MODELO DE REQUEST
# ===========================

class ChatRequest(BaseModel):
    message: str
    username: Optional[str] = None  # nombre del usuario logueado


# ===========================
#   LÓGICA PARA BD
# ===========================

def get_user_id_by_username(username: str) -> Optional[int]:
    """
    Obtiene el ID de usuario a partir del username.
    Tabla: users(id, created_at, email, password, role, updated_at, username)
    """
    query = text("SELECT id FROM users WHERE username = :username LIMIT 1")
    with engine.connect() as conn:
        row = conn.execute(query, {"username": username}).first()
        if row:
            return row[0]
    return None


def build_technical_context(user_id: int) -> str:
    """
    Contexto técnico (HU-IA-01) usando cultivos + productos.

    crops:
      id, fertilize_crop, make_crop_hole, make_crop_line, name,
      oxygenate_crop, pest_cleanup_days, pick_up_weed, user_id,
      watering_days, product_id

    products:
      id, description, image_url, name,
      recommended_cultivation_depth,
      recommended_cultivation_distance,
      recommended_growing_climate,
      recommended_growing_season,
      recommended_soil_type, user_id
    """
    query = text("""
        SELECT
            c.name AS crop_name,
            c.watering_days,
            c.fertilize_crop,
            c.pest_cleanup_days,
            c.pick_up_weed,
            c.oxygenate_crop,
            c.make_crop_line,
            c.make_crop_hole,
            p.name AS product_name,
            p.recommended_cultivation_depth,
            p.recommended_cultivation_distance,
            p.recommended_growing_climate,
            p.recommended_growing_season,
            p.recommended_soil_type
        FROM crops c
        LEFT JOIN products p ON c.product_id = p.id
        WHERE c.user_id = :user_id
        ORDER BY c.id DESC
        LIMIT 5
    """)

    with engine.connect() as conn:
        rows = conn.execute(query, {"user_id": user_id}).fetchall()

    if not rows:
        return "El usuario no tiene cultivos registrados en la base de datos."

    lines = ["Cultivos y parámetros técnicos registrados del usuario:"]

    for (
        crop_name,
        watering_days,
        fertilize_crop,
        pest_cleanup_days,
        pick_up_weed,
        oxygenate_crop,
        make_crop_line,
        make_crop_hole,
        product_name,
        rec_depth,
        rec_distance,
        rec_climate,
        rec_season,
        rec_soil_type
    ) in rows:
        lines.append(
            f"- Cultivo: {crop_name} "
            f"(producto asociado: {product_name or 'N/D'}). "
            f"Frecuencia de riego (días): {watering_days or 'N/D'}, "
            f"frecuencia de fertilización (días): {fertilize_crop or 'N/D'}, "
            f"días para limpieza de plagas: {pest_cleanup_days or 'N/D'}, "
            f"días para recoger maleza: {pick_up_weed or 'N/D'}, "
            f"frecuencia de oxigenación del suelo (días): {oxygenate_crop or 'N/D'}, "
            f"labores de trazo de surcos/líneas: {make_crop_line or 'N/D'}, "
            f"hoyado o preparación de hoyos: {make_crop_hole or 'N/D'}. "
            f"Recomendaciones del producto: profundidad de cultivo recomendada: {rec_depth or 'N/D'}, "
            f"distancia entre plantas: {rec_distance or 'N/D'}, "
            f"clima recomendado: {rec_climate or 'N/D'}, "
            f"temporada recomendada: {rec_season or 'N/D'}, "
            f"tipo de suelo recomendado: {rec_soil_type or 'N/D'}."
        )

    return "\n".join(lines)


def build_commercial_context(user_id: int) -> str:
    """
    Contexto comercial (HU-IA-02) usando ventas.

    sales:
      id, description, image_url, name, quantity, unit_price, user_id
    """
    query = text("""
        SELECT name, quantity, unit_price, description
        FROM sales
        WHERE user_id = :user_id
        ORDER BY id DESC
        LIMIT 5
    """)

    with engine.connect() as conn:
        rows = conn.execute(query, {"user_id": user_id}).fetchall()

    if not rows:
        return "No se encontraron ventas recientes para este usuario en la base de datos."

    lines = ["Historial de ventas recientes del usuario:"]
    total_price = 0.0
    count_price = 0

    for name, quantity, unit_price, description in rows:
        lines.append(
            f"- Producto vendido: {name}, cantidad: {quantity or 'N/D'}, "
            f"precio unitario: {unit_price or 'N/D'} S/, "
            f"detalle: {description or 'sin descripción'}."
        )
        if unit_price is not None:
            total_price += float(unit_price)
            count_price += 1

    if count_price > 0:
        avg_price = total_price / count_price
        lines.append(f"Precio unitario promedio en estas ventas: {avg_price:.2f} S/.")

    return "\n".join(lines)


def detect_topic(message: str) -> Optional[str]:
    """
    Detección simple del tipo de consulta:
    - 'technical' si habla de riego, fertilización, plagas
    - 'commercial' si habla de precios, venta, compradores
    """
    m = message.lower()

    commercial_keywords = [
        "precio", "precios", "vender", "venta", "comprador",
        "compradores", "mercado", "negocio", "ganancia", "margen"
    ]
    technical_keywords = [
        "riego", "regar", "fertiliz", "abono", "plaga", "plagas",
        "insecto", "enfermedad", "hongos", "control de plagas"
    ]

    if any(k in m for k in commercial_keywords):
        return "commercial"
    if any(k in m for k in technical_keywords):
        return "technical"
    return None


def build_system_messages(topic: Optional[str], db_context: Optional[str]) -> list:
    """
    Mensajes de sistema para OpenAI, alineados a HU-IA-01 y HU-IA-02,
    con reglas más claras sobre cómo usar (y NO usar) los datos de la BD.
    """
    base_prompt = """
        Eres Ayni Assistant, un asistente virtual inteligente para agricultores y productores rurales en el Perú.

        Tu misión es cumplir dos Historias de Usuario principales:

        HU-IA-01: Asesoría técnica agrícola inteligente
        - Brindar recomendaciones sobre riego, fertilización y control de plagas en tiempo real.
        - Cuando la consulta sea de riego/fertilización/plagas:
        - Si en la información de la base de datos (contexto) aparecen valores concretos
            como frecuencias de riego, fertilización, limpieza de plagas, oxigenación del suelo,
            ÚSALOS explícitamente y dilo de forma clara, por ejemplo:
            "Según tus registros en Ayni, estás fertilizando cada 30 días..."
        - Si la información de la base de datos no menciona un valor concreto, NO digas que
            "no está registrado" ni inventes el estado de los registros. Simplemente da una
            recomendación general y, si es útil, sugiere que el usuario registre esos datos en Ayni.
        - Las recomendaciones deben ser prácticas, realistas y sostenibles.

        HU-IA-02: Asesoría comercial inteligente
        - Orientar al usuario sobre precios de venta y posibles compradores.
        - Cuando la consulta sea de precios/ventas/mercado:
        - Si en la información de la base de datos hay ventas históricas con precios,
            úsalo como referencia y dilo, por ejemplo:
            "En tus ventas registradas, has vendido entre X y Y S/ por unidad..."
        - Si no hay ventas registradas, NO digas que la base de datos está incompleta.
            En su lugar, ofrece rangos y criterios generales (calidad, tipo de comprador,
            zona, temporada) y sugiere que utilice el módulo de ventas de Ayni para
            registrar operaciones futuras.

        Reglas generales:
        - Usa lenguaje claro, cercano y respetuoso.
        - Explica conceptos técnicos con ejemplos simples.
        - No inventes datos numéricos concretos que no provengan de:
        - el contexto de la base de datos, o
        - lo que el usuario te diga explícitamente.
        - Si la pregunta no es de agricultura ni de comercio, respóndela brevemente
        y trata de reconducir la conversación a cómo la tecnología y la gestión pueden
        ayudar en la actividad agrícola.
    """

    system_messages = [
        {"role": "system", "content": base_prompt}
    ]

    if topic == "technical":
        system_messages.append({
            "role": "system",
            "content": "La consulta actual se interpreta como ASESORÍA TÉCNICA AGRÍCOLA (HU-IA-01)."
        })
    elif topic == "commercial":
        system_messages.append({
            "role": "system",
            "content": "La consulta actual se interpreta como ASESORÍA COMERCIAL (HU-IA-02)."
        })

    if db_context:
        system_messages.append({
            "role": "system",
            "content": "Información real del usuario obtenida desde la base de datos:\n" + db_context
        })

    return system_messages


# ===========================
#   ENDPOINTS
# ===========================

@app.post("/chat")
async def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        return {"reply": "Por favor, escribe algo para que pueda ayudarte 😊"}

    topic = detect_topic(message)
    db_context = None

    # Si tenemos username, intentamos usar su información de BD
    if request.username:
        try:
            user_id = get_user_id_by_username(request.username)
            if user_id:
                if topic == "technical":
                    db_context = build_technical_context(user_id)
                elif topic == "commercial":
                    db_context = build_commercial_context(user_id)
        except Exception as e:
            # No rompemos el chat si falla la BD
            print("⚠️ Error obteniendo contexto desde BD:", e)

    try:
        system_messages = build_system_messages(topic, db_context)

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                *system_messages,
                {"role": "user", "content": message},
            ],
            temperature=0.7,
            max_tokens=400,
        )

        reply = completion.choices[0].message.content.strip()
        return {
            "reply": reply,
            "used_topic": topic,
            "used_username": request.username,
            "db_context": db_context,  # útil para debug o demo con el profe
        }

    except Exception as e:
        print("❌ Error con OpenAI:", e)
        return {"reply": f"⚠️ Error al contactar con OpenAI: {e}"}


@app.get("/")
def root():
    return {"message": "Ayni ChatBot API con OpenAI + MySQL funcionando correctamente 🤖"}
