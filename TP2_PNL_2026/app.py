"""RAG sobre protección de datos personales (Ley 25.326 y guías de la AAIP).

Mismos pasos que rag_integrador.ipynb, con interfaz Gradio.
Ejecutar desde esta carpeta: python app.py
"""

import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

RAIZ = Path(__file__).resolve().parent
load_dotenv(RAIZ / ".env")
MODELO_GENERADOR = os.getenv("OLLAMA_MODEL", "granite4:latest")
MODELO_EMBEDDINGS = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")

# Corte elegido en la auditoría (corrida 2) y colección de esa corrida.
TAMANO_FRAGMENTO = 1100
SOLAPAMIENTO = 180
K = 3
TAMANO_LOTE = 50
CARPETA_INDICE = "chroma_db_integrador"
NOMBRE_COLECCION = "integrador_1100_180_bge_m3"

# Frase fija de abstención: la misma que pide el prompt.
FRASE_ABSTENCION = "No encuentro esa información en los documentos."

# Cada documento: (ruta, página inicial, página final) con numeración impresa.
DOCUMENTOS = [
    (RAIZ / "datos" / "guia_ia_responsable_aaip_2024.pdf", 6, 49),
    (RAIZ / "datos" / "lineamientos_plan_datos_cft_aaip_2023.pdf", 9, 30),
    (RAIZ / "datos" / "ley_25326_texto_actualizado.pdf", 1, 20),
]


def cargar_documentos() -> list[Document]:
    """Carga los 3 PDF y conserva solo el recorte de páginas de cada uno.

    Returns:
        Páginas del recorte con sus metadatos de origen.
    """
    paginas_del_recorte = []

    for ruta, pagina_inicial, pagina_final in DOCUMENTOS:
        paginas_del_pdf = PyPDFLoader(str(ruta)).load()

        for pagina in paginas_del_pdf:
            pagina_impresa = pagina.metadata["page"] + 1

            if pagina_inicial <= pagina_impresa <= pagina_final:
                paginas_del_recorte.append(pagina)

    return paginas_del_recorte


def preparar_indice() -> Chroma:
    """Abre el índice persistente e indexa los fragmentos si está vacío.

    Returns:
        Colección Chroma de la corrida 2 (1100/180, bge-m3).
    """
    embeddings = OllamaEmbeddings(model=MODELO_EMBEDDINGS)
    indice = Chroma(
        collection_name=NOMBRE_COLECCION,
        embedding_function=embeddings,
        persist_directory=str(RAIZ / CARPETA_INDICE),
    )

    # Si la colección ya tiene datos se usa tal cual: no se duplican fragmentos.
    if indice.get(limit=1)["ids"]:
        return indice

    documentos = cargar_documentos()
    divisor = RecursiveCharacterTextSplitter(
        chunk_size=TAMANO_FRAGMENTO,
        chunk_overlap=SOLAPAMIENTO,
    )
    fragmentos = divisor.split_documents(documentos)
    identificadores = []

    for numero in range(len(fragmentos)):
        identificadores.append(f"fragmento_{numero}")

    # Se indexa por lotes para no sobrecargar a Ollama, como en la sección 6.
    for inicio in range(0, len(fragmentos), TAMANO_LOTE):
        fin = inicio + TAMANO_LOTE
        indice.add_documents(
            documents=fragmentos[inicio:fin],
            ids=identificadores[inicio:fin],
        )
        print("Indexados hasta el fragmento", min(fin, len(fragmentos)))

    return indice


INDICE: Chroma | None = None


def consultar(pregunta: str) -> tuple[str, str]:
    """Recupera evidencia y devuelve respuesta y fragmentos por separado.

    Args:
        pregunta: Consulta de la persona usuaria.

    Returns:
        Respuesta generada y fragmentos efectivamente enviados al modelo.
    """
    if not pregunta.strip():
        return "Escribí una pregunta.", "Todavía no se consultaron fuentes."

    if INDICE is None:
        raise RuntimeError("El índice debe prepararse antes de abrir la interfaz.")

    hallazgos = INDICE.similarity_search_with_score(query=pregunta, k=K)
    contexto = ""
    fuentes = ""

    for numero in range(len(hallazgos)):
        documento, distancia = hallazgos[numero]
        nombre = Path(str(documento.metadata.get("source", "sin fuente"))).name
        pagina = documento.metadata["page"] + 1
        referencia = f"{nombre}, página {pagina}"

        contexto += f"[{numero + 1}] {referencia}\n{documento.page_content}\n\n"
        fuentes += f"**[{numero + 1}] {referencia}** (distancia {distancia:.3f})\n\n"
        fuentes += f"```text\n{documento.page_content}\n```\n\n"

    # Prompt de la sección 8 del cuaderno.
    pedido = ChatPromptTemplate.from_messages([
        (
            "system",
            "Respondé en español con una oración completa y sólo con la evidencia "
            "del contexto. Citá [1], [2] o [3] junto a cada afirmación. Si el dato "
            "no está en el contexto, respondé exactamente: No encuentro esa "
            "información en los documentos.",
        ),
        ("human", "Contexto:\n{contexto}\n\nPregunta: {pregunta}"),
    ])
    modelo = ChatOllama(model=MODELO_GENERADOR, temperature=0)
    # Primero recuperamos documentos a la vista; esta cadena sólo redacta.
    cadena = pedido | modelo | StrOutputParser()
    respuesta = cadena.invoke({"contexto": contexto, "pregunta": pregunta})

    # La lista muestra qué se envió al modelo; no certifica su interpretación.
    texto_respuesta = respuesta.strip()

    if not texto_respuesta or texto_respuesta in {"[1]", "[2]", "[3]"}:
        texto_respuesta = (
            "No obtuve una respuesta verificable. "
            "Revisá los fragmentos recuperados."
        )

    # Una abstención no debe venir acompañada de citas que aparenten respaldar un dato.
    if texto_respuesta.startswith(FRASE_ABSTENCION):
        texto_respuesta = FRASE_ABSTENCION

    return texto_respuesta, fuentes


if __name__ == "__main__":
    INDICE = preparar_indice()
    interfaz = gr.Interface(
        fn=consultar,
        inputs=gr.Textbox(label="Pregunta sobre protección de datos personales"),
        outputs=[
            gr.Textbox(label="Respuesta"),
            gr.Markdown(label="Fragmentos recuperados y enviados al modelo"),
        ],
        title="Consulta sobre protección de datos personales",
        description=(
            "Responde con la Ley 25.326 y dos guías oficiales de la AAIP (Guía para "
            "una IA responsable y Lineamientos para un Plan de Protección de Datos). "
            "La Ley es obligatoria; las guías son recomendaciones."
        ),
        flagging_mode="never",
    )
    interfaz.launch()
