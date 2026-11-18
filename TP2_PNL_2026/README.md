# TP Integrador 2 — RAG local sobre protección de datos personales

Procesamiento de Lenguaje Natural — Tecnicatura Superior en Ciencias de Datos e Inteligencia Artificial (IFTS N.º 24).

Sistema RAG local que responde preguntas sobre la Ley 25.326 y dos guías oficiales de la AAIP, busca evidencia antes de generar y muestra los fragmentos que recibió el modelo. Todo corre en la computadora con Ollama: no se usan servicios pagos ni claves de API.

---

## 1. Problema y destinatario

Este sistema permite que una persona responsable de un proyecto de datos, en un organismo público o en una empresa, consulte qué debe cumplir y qué se recomienda cuando su proyecto trata datos personales o usa IA (por ejemplo, cuándo necesita el consentimiento de los titulares, en qué casos conviene una evaluación de impacto o qué exigirle a un proveedor de nube), en la Ley 25.326 y en dos guías oficiales de la AAIP, para diseñar sus proyectos cumpliendo la normativa antes de ponerlos en marcha.

La Ley 25.326 es obligatoria; la Guía de la AAIP (versión preliminar, 2024) y los Lineamientos (CFT/AAIP, 2023) son recomendaciones.

**Puede responder** preguntas cuya respuesta esté escrita en el texto de los documentos: cuándo hace falta el consentimiento y sus excepciones, datos sensibles, transferencia internacional y derechos de los titulares (Ley); evaluación de impacto, niveles de riesgo y recomendaciones por etapa del ciclo de vida de un sistema de IA (Guía); qué incluir en un plan de protección de datos y qué exigir al alojar datos en la nube (Lineamientos).

**No puede responder** normativa de otros países, jurisprudencia, reformas posteriores a la versión descargada, plazos o costos de trámites que los documentos no mencionan, ni el contenido de figuras e imágenes (ver sección 7). No reemplaza el asesoramiento legal.

**Origen del proyecto.** La idea surge de la materia Trabajo, Tecnología y Sociedad, que curso en paralelo en la misma tecnicatura. Allí trabajamos sobre protección de datos personales, inteligencia artificial y el estado de las leyes, guías y regulaciones que existen hoy. Al mismo tiempo, las noticias recientes sobre la seguridad de los datos personales muestran una preocupación creciente, y es esperable que las organizaciones tengan que justificar cada vez más cómo sus productos cumplen la normativa. Este sistema toma esa pregunta y la convierte en un prototipo: una herramienta mínima que permite consultar la ley y las guías oficiales con la fuente a la vista. No pretende reemplazar el asesoramiento legal; es un primer paso, pensado como si fuera el producto mínimo de una herramienta real.

---

## 2. Corpus

Los tres archivos están en [`datos/`](datos/) y se publican para que el índice pueda reconstruirse desde ellos. El detalle de procedencia está en [`datos/LEEME.md`](datos/LEEME.md).

| Archivo | Documento | Autor u organismo | Origen | Versión | Condiciones de uso | Páginas indexadas |
|---|---|---|---|---|---|---|
| `guia_ia_responsable_aaip_2024.pdf` | Guía para entidades públicas y privadas en materia de Transparencia y Protección de Datos Personales para una IA responsable | Agencia de Acceso a la Información Pública (AAIP) | [argentina.gob.ar](https://argentina.gob.ar/sites/default/files/aaip-argentina-guia_para_usar_la_ia_de_manera_responsable.pdf) | Versión preliminar, junio de 2024 | CC BY 4.0 (términos de argentina.gob.ar) | 6 a 49 |
| `lineamientos_plan_datos_cft_aaip_2023.pdf` | Lineamientos para la formulación de un Plan de Protección de Datos Personales | Consejo Federal para la Transparencia (CFT), editado por la AAIP | [argentina.gob.ar](https://www.argentina.gob.ar/sites/default/files/documento_datos_2023.pdf) | 2023 | CC BY 4.0 (términos de argentina.gob.ar) | 9 a 30 |
| `ley_25326_texto_actualizado.pdf` | Ley 25.326 de Protección de los Datos Personales, texto actualizado | Congreso de la Nación Argentina | [argentina.gob.ar](https://www.argentina.gob.ar/normativa/nacional/ley-25326-64790/actualizacion) | Texto actualizado (incluye la Ley 26.343, BO 9/1/2008) | Norma pública; CC BY 4.0 (términos de argentina.gob.ar) | 1 a 20 |

**Recorte.** De cada documento se indexa solo el cuerpo (86 páginas en total). Se excluyen portadas, autoridades, índices, páginas en blanco, acrónimos, referencias y anexos, porque no aportan contenido para responder y agregarían ruido a la búsqueda.

**Transformaciones.** La Guía y los Lineamientos son los PDF originales, sin cambios. La Ley es la página de argentina.gob.ar guardada como PDF desde el navegador (el PDF nativo de la Cámara de Diputados no estaba disponible); conserva el encabezado y el pie del navegador en cada página.

---

## 3. Arquitectura

```text
3 PDF en datos/ (versionados en GitHub)
  → PyPDFLoader + recorte de páginas (archivo y página en los metadatos)
  → RecursiveCharacterTextSplitter (chunk_size=1100, chunk_overlap=180; elegido en la auditoría)
  → OllamaEmbeddings (bge-m3) → Chroma persistente (chroma_db_integrador/, no se publica)
pregunta → embedding con bge-m3 → similarity_search_with_score (k=3)
  → contexto «[n] archivo, página» → ChatPromptTemplate → ChatOllama (granite4:latest, temperature=0)
  → Gradio: respuesta + fragmentos enviados al modelo
```

| Pieza | Elección |
|---|---|
| Lenguaje y entorno | Python 3.12, `uv` |
| Librerías | las de [`requirements.txt`](requirements.txt) (LangChain, Chroma, Ollama, Gradio, pandas) |
| Embeddings | `bge-m3` en Ollama, el mismo para indexar y consultar |
| Modelo generador | `granite4:latest` en Ollama, `temperature=0` |
| Base de clase | cuadernos 04, 06 y 07 y `app.py` de la unidad 014 |

**Archivos del proyecto**

| Archivo | Qué es |
|---|---|
| `rag_integrador.ipynb` | El recorrido completo y verificable: carga, revisión del texto, fragmentación, índice, recuperación, generación y auditoría |
| `app.py` | La interfaz Gradio sobre el mismo pipeline (mismo recorte, corte 1.100/180, K=3 y prompt) |
| `utils_visualizacion.py` | Copia sin cambios del archivo de clase; solo dibuja gráficos |
| `pruebas/auditoria.csv` | Registro de las corridas de prueba (6 casos por corrida, 20 columnas) |
| `datos/` | Los tres PDF y su procedencia |
| `.env.example` | Nombres de los dos modelos de Ollama; se copia como `.env` |

---

## 4. Instalación local

### 4.1 Requisitos previos

| Requisito | Para qué | Cómo obtenerlo |
|---|---|---|
| Python 3.12 | Ejecutar el cuaderno y la app | Lo puede instalar `uv` al crear el entorno (`uv venv --python 3.12`) |
| [`uv`](https://docs.astral.sh/uv/) | Crear el entorno virtual e instalar las librerías | Ver la documentación de `uv` |
| [Ollama](https://ollama.com/download) | Correr en local el modelo de embeddings y el generador | Ver abajo |

**Instalar Ollama** (según [ollama.com/download](https://ollama.com/download)):

- **Windows:** abrir PowerShell y ejecutar
  ```powershell
  irm https://ollama.com/install.ps1 | iex
  ```
- **macOS:** descargar e instalar [Ollama.dmg](https://ollama.com/download/Ollama.dmg), o en la terminal:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```
- **Linux:**
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

Para comprobar la instalación: `ollama --version`. Ollama tiene que estar abierto mientras se usa el cuaderno o la app.

### 4.2 Entorno, librerías y modelos

Desde la carpeta `TP2_PNL_2026/`:

```powershell
uv venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
ollama pull granite4:latest
ollama pull bge-m3
copy .env.example .env
```

Para comprobar que los modelos están descargados: `ollama list` tiene que mostrar `bge-m3` y `granite4:latest`.

**Qué instala `requirements.txt`**

| Librería | Para qué se usa |
|---|---|
| `langchain-community` | `PyPDFLoader`: carga los PDF página por página |
| `pypdf` | Lectura de los PDF que usa `PyPDFLoader` |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter`: corta el texto en fragmentos |
| `langchain-ollama`, `ollama` | `OllamaEmbeddings` (bge-m3) y `ChatOllama` (granite4) |
| `langchain-chroma`, `chromadb` | Índice vectorial Chroma |
| `langchain-core` | `ChatPromptTemplate` y `StrOutputParser` |
| `gradio` | Interfaz web local (`app.py`) |
| `pandas` | Tablas de resultados y `pruebas/auditoria.csv` |
| `matplotlib` | Gráficos del cuaderno |
| `python-dotenv` | Lee los nombres de los modelos desde `.env` |
| `ipykernel` | Usar `.venv` como kernel del cuaderno en VS Code |

En un proyecto con `requirements.txt` se instala con `uv pip install -r requirements.txt` (`uv sync` es para proyectos con `pyproject.toml`).

`.env.example` solo define los dos modelos (`OLLAMA_MODEL` y `OLLAMA_EMBEDDING_MODEL`); no hace falta ninguna clave.

### 4.3 Usar la interfaz (paso a paso)

La interfaz es una página web local: corre en tu computadora y no necesita internet una vez descargados los modelos.

1. **Abrí Ollama.** Tiene que estar funcionando antes de empezar (en Windows, buscalo en el menú Inicio). Para comprobarlo, en una terminal: `ollama list`.
2. **Abrí una terminal en la carpeta `TP2_PNL_2026/`.** En VS Code: menú *Terminal → Nueva terminal*.
3. **Ejecutá la app:**
   ```powershell
   .venv\Scripts\python.exe app.py
   ```
4. **Esperá el mensaje con la dirección.** La primera vez la app arma el índice y muestra el avance («Indexados hasta el fragmento 50, 100…»); puede tardar unos minutos. Cuando esté lista aparece:
   ```text
   * Running on local URL:  http://127.0.0.1:7860
   ```
   Las veces siguientes usa el índice ya guardado y abre enseguida.
5. **Abrí esa dirección en el navegador** (Ctrl+clic sobre el enlace en la terminal, o copiala).
6. **Escribí una pregunta y presioná *Submit*.** Por ejemplo: *¿Qué debo tener en cuenta al contratar un proveedor de nube para alojar datos personales?*
7. **Leé las dos partes del resultado:**
   - **Respuesta:** lo que redactó el modelo, con referencias [1], [2] o [3]. Si el dato no está en los documentos, responde «No encuentro esa información en los documentos.».
   - **Fragmentos recuperados:** los tres textos que recibió el modelo, con archivo, página y distancia (menor distancia = más parecido a la pregunta). Sirven para comprobar la respuesta abriendo el PDF en esa página.
8. **Para otra pregunta**, reemplazá el texto y volvé a presionar *Submit* (*Clear* borra todo).
9. **Para cerrar la app**, volvé a la terminal y presioná **Ctrl+C**. Mientras la app está abierta, esa terminal queda ocupada; si necesitás otra, abrila con el botón **+**.

**Problemas frecuentes**

| Lo que ves | Qué significa | Qué hacer |
|---|---|---|
| El navegador muestra `ERR_CONNECTION_REFUSED` | La app no está corriendo (se cerró con Ctrl+C o todavía no terminó de arrancar) | Volvé a ejecutar el paso 3 y esperá el mensaje del paso 4 |
| Error con `.../tokenize` o «connection refused» en la terminal | Ollama se cerró o se cortó su proceso interno | Cerrá Ollama del todo, abrilo de nuevo, comprobá con `ollama list` y volvé a ejecutar |
| Aviso `DeprecationWarning: langchain-community…` | Aviso de la librería que provee `PyPDFLoader`; no es un error | Nada: la app funciona igual |

### 4.4 Ejecutar el cuaderno

Abrí la carpeta en VS Code, elegí `.venv` como kernel y abrí `rag_integrador.ipynb`. Con Ollama abierto, ejecutá las celdas en orden. El cuaderno muestra todo el recorrido: el texto extraído, los fragmentos, el índice, lo que se recupera para cada pregunta, las respuestas y la auditoría.

Ejecutar el cuaderno no es necesario para usar la app. Tené en cuenta que al ejecutarlo completo se vuelven a generar las corridas de la auditoría y se reescribe `pruebas/auditoria.csv`: los fragmentos recuperados son siempre los mismos, pero las respuestas del modelo pueden variar (ver sección 6).

---

## 5. Reindexación

- El índice se guarda en `chroma_db_integrador/`, que Git ignora: se reconstruye desde `datos/`.
- El nombre de cada colección incluye el corte y el modelo de embeddings: `integrador_500_80_bge_m3` (corrida 1) e `integrador_1100_180_bge_m3` (corrida 2 y app). Si cambian el corpus, el recorte, el corte o el modelo, se crea **otra colección**: no se mezclan vectores de configuraciones distintas.
- En el cuaderno, la celda que construye cada índice borra y recrea su propia colección cada vez que se ejecuta, así no quedan fragmentos de una configuración anterior. `app.py` usa la colección si ya tiene datos y, si está vacía, la construye.
- Para reconstruir todo desde cero: cerrar la app, borrar la carpeta `chroma_db_integrador/` y volver a ejecutar el cuaderno o la app.
- Los fragmentos se indexan **en lotes de 50**. Al enviar cientos de fragmentos en una sola llamada, el proceso de `bge-m3` en Ollama se cortaba a mitad de camino (error de conexión en `/tokenize`); con lotes se obtiene el mismo índice y se ve el avance. El lote no cambia los fragmentos: el corte ya está hecho y cada vector depende solo del texto de su fragmento.
- No se publican `.venv/`, `.env` ni `chroma_db*/`.

---

## 6. Pruebas

Casos escritos **antes** de evaluar: tres respondibles (uno por documento), dos sin respuesta en el corpus y uno de limitación. La tabla muestra la **corrida 1** (corte base de clase 500/80, K=3).

| Pregunta | Respuesta / fuente esperada | Fragmento recuperado (archivo y página) | Respuesta del modelo | ¿Qué observé? |
|---|---|---|---|---|
| **R1.** ¿En qué casos mi organización debería hacer una evaluación de impacto en protección de datos antes de desarrollar un sistema de IA? | Organizaciones que tratan grandes volúmenes de datos, datos sensibles o hacen una indagación sistemática — Guía, p. 29 | Guía p. 36, p. 29 y p. 30 | Tres viñetas sobre privacidad por diseño y los riesgos que detecta la evaluación; no dice en qué casos corresponde | Llega la p. 29, pero el fragmento del título; el de los criterios quedó en el puesto 12. El modelo respondió sin la evidencia en lugar de abstenerse |
| **R2.** ¿Qué debo tener en cuenta al contratar un proveedor de nube para alojar datos personales? | Altos estándares de seguridad, cumplimiento de las regulaciones y acuerdo con el encargado — Lineamientos, p. 27 | Lineamientos p. 27 (los tres) | Correcta: estándares de seguridad, proveedores extranjeros y contenido del acuerdo | Funciona en todas las corridas. Agrupa las citas al final en lugar de ponerlas junto a cada afirmación |
| **R3.** ¿Necesito el consentimiento de las personas para usar sus datos personales en mi proyecto? ¿Hay excepciones? | Sí, libre, expreso e informado; no es necesario, entre otros casos, con fuentes de acceso público irrestricto o funciones propias del Estado — Ley 25.326, art. 5, p. 4 | Guía p. 20, Ley p. 12 y p. 6 | «No encuentro esa información en los documentos.» | No llega la Ley p. 4 (el fragmento con las excepciones quedó en el puesto 8). Llega la regla sin excepciones y el modelo se abstiene |
| **S1.** ¿Cuánto tarda la AAIP en aprobar una evaluación de impacto? | No figura en el corpus: debería abstenerse | Guía p. 30, p. 46 y p. 29 | «No encuentro esa información en los documentos.» | Correcto. Son las distancias más altas de los seis casos (0,91 a 1,00) |
| **S2.** ¿Cuánto cuesta inscribir una base de datos en el Registro Nacional de Bases de Datos? | No figura en el corpus: debería abstenerse | Lineamientos p. 18, Guía p. 33 (dos) | «No encuentro esa información en los documentos.» | Correcto, aunque el primer fragmento habla de la inscripción en el Registro a una distancia tan baja como una respondible (0,649): la distancia sola no detecta que no hay respuesta |
| **L1.** ¿Qué valor de riesgo corresponde a una probabilidad alta con un impacto crítico? | El dato está en la Figura 3 de la Guía (p. 30), que es una imagen: debería abstenerse | Guía p. 29 (dos) y p. 31 | Describió el «riesgo alto» del reglamento europeo (Guía p. 31, otro tema) y lo citó como [1][2] | Limitación de extracción: la matriz no se indexa. En esta corrida el modelo no se abstuvo; en otras ejecuciones sí |

**Cómo se comprueba la evidencia.** Para R1–R3 se definió una frase textual de la fuente esperada y se busca en los fragmentos recuperados (texto normalizado en minúsculas y espacios). Las primeras frases elegidas para R1 y R3 aparecían también en otras páginas y daban falsos positivos; se reemplazaron por frases que solo existen en la fuente esperada («son las que tratan grandes volúmenes de datos» y «no será necesario el consentimiento»). La abstención se cuenta solo si la respuesta es exactamente la frase fija del prompt. Cuando la evidencia no entra, el cuaderno busca en qué fragmento está y en qué puesto quedó, para distinguir un problema de corte de uno de recuperación.

**Corrida 2: un solo cambio, el corte a 1.100/180.**

| Caso | Evidencia en el top-3 (corrida 1 → 2) | Respuesta (corrida 1 → 2) |
|---|---|---|
| R1 | No → **Sí** (Guía p. 29 primera) | Respondió sin la evidencia → se abstuvo |
| R2 | Sí → Sí | Correcta → correcta |
| R3 | No → **Sí** (Ley p. 4 tercera) | Se abstuvo → se abstuvo |
| S1, S2 | — | Se abstuvieron en las dos |
| L1 | — | Respondió → se abstuvo |

El corte amplio resolvió la recuperación: en R1 y R3 la evidencia ahora llega al contexto. Pero el modelo se abstuvo en los dos casos, así que el problema pasó de la búsqueda a la generación.

**Interfaz (`app.consultar`, misma configuración que la corrida 2).** Cinco casos coinciden con la corrida 2. R1 varió: en una ejecución respondió exactamente con la evidencia de la Guía p. 29 y la citó bien; al volver a ejecutarla se abstuvo. Con los mismos fragmentos y `temperature=0`, el modelo a veces responde y a veces se abstiene.

**Limitaciones observadas y qué cambiaría.**
- **Figuras como imagen (L1).** El dato de la matriz de riesgo no se extrae del PDF. Cambiaría la carga para incluir OCR documentado o transcribiría la figura a texto.
- **La generación no es estable.** La búsqueda devuelve siempre los mismos fragmentos, pero la respuesta puede cambiar entre ejecuciones (R1 y L1). Por eso el sistema muestra los fragmentos: la respuesta se tiene que verificar contra ellos.
- **Abstención con evidencia disponible (R1 y R3, corrida 2).** Como siguiente prueba cambiaría solo el prompt (por ejemplo, permitir varias oraciones) y compararía con la corrida 2.

El registro completo de cada corrida está en [`pruebas/auditoria.csv`](pruebas/auditoria.csv). Este conjunto de pruebas muestra el mecanismo, no mide la precisión general del sistema.

---

## 7. Decisiones y límites

- **Fragmentación.** `chunk_size` y `chunk_overlap` se miden en caracteres. Una página típica del corpus tiene unos 2.300 caracteres: 500 equivale a un cuarto de página y 1.100 a media página. Se compararon 500/80 (483 fragmentos) y 1.100/180 (239 fragmentos) con R3: el corte breve no trajo la Ley p. 4 ni las excepciones del art. 5; el amplio trajo la regla y las excepciones juntas. La corrida 1 usa el corte base de clase (500/80) y la corrida 2 cambia solo el corte. **Elijo 1.100/180** porque es el que trae al contexto la evidencia de las tres respondibles. Descarto 500/80: con ese corte la búsqueda falla en dos de las tres (R1 y R3 quedan fuera del top-3), y su única ventaja, fragmentos más cortos y enfocados, no compensa que la evidencia no llegue. El riesgo del corte amplio es que mezcla temas en un mismo fragmento.
- **k = 3.** Es la cantidad de fragmentos que entran al contexto. La comparación de cortes también se hizo con K=3 (y no K=1 como en el laboratorio de clase), para ver lo mismo que recibe el modelo: con tres documentos que hablan del consentimiento, la fuente esperada no siempre queda primera.
- **Embeddings.** `bge-m3`, el mismo para indexar y consultar; el cuaderno comprueba que el vector guardado y el de la pregunta tienen la misma dimensión (1024).
- **Prompt.** Se usa textual el prompt de clase: responder solo con el contexto, citar [1], [2] o [3] y, si falta el dato, responder exactamente «No encuentro esa información en los documentos.». La abstención se pide, pero no está garantizada: S1 y S2 se abstuvieron siempre; L1 y R1 cambiaron de comportamiento entre ejecuciones con los mismos fragmentos.
- **Figuras e imágenes.** `PyPDFLoader` solo lee la capa de texto. La matriz de riesgo de la Guía (p. 30, Figura 3) es una imagen y su contenido no se indexa; lo mismo pasa con los gráficos de los Lineamientos (p. 28): de la torta quedan las etiquetas y los porcentajes sueltos, y del gráfico de barras solo las marcas del eje, no el valor de cada barra. Ese fragmento llega al contexto en R2; el modelo no usó esos números, pero podría confundir una marca del eje con un dato. Contar caracteres por página detecta páginas vacías, pero no contenido perdido dentro de una página que además tiene texto: eso se detectó comparando a mano el texto extraído con el PDF. Para incluirlas haría falta OCR documentado o transcribir la figura a texto.
- **Ruido del corpus.** La Guía repite un encabezado en cada página y tiene palabras cortadas con guion; la Ley trae fecha, título y URL del navegador. No se limpió: se declara como límite.
- **Guía preliminar.** Las respuestas reflejan la versión preliminar de 2024.
- **Dependencias.** `langchain-community` (de donde viene `PyPDFLoader`) muestra un aviso de que dejará de mantenerse; se usa porque es la ruta de clase y funciona.
- **Citas.** El sistema muestra los fragmentos que recibió el modelo; eso permite volver al documento, pero no garantiza que el modelo los haya interpretado bien.

---

## 8. Uso de IA

Usé IA como asistente para planificar, escribir código y revisar. Las decisiones, la ejecución y las verificaciones las hice yo. Los modelos locales (`bge-m3` y `granite4`) son parte del sistema, no ayuda para construirlo.

| Momento | Etapa | Herramienta y modelo | Para qué | Qué verifiqué personalmente |
|---|---|---|---|---|
| 12/11 | Planificación | Claude (claude.ai), Claude Opus 5.5 | Resumir la consigna y el material de clase para armar el plan de trabajo | Leí la consigna completa y el cuaderno 07 |
| 12–13/11 | Corpus | Claude (claude.ai), Claude Opus 5.5 | Buscar documentos oficiales con licencia reutilizable y definir los recortes | Descargué los PDF y comprobé organismo, fecha, licencia, texto extraíble y rangos de páginas |
| 13–14/11 | Pruebas e instrucciones | Claude (claude.ai), Claude Opus 5.5 | Proponer los casos de prueba, el método de auditoría y el archivo de instrucciones para el asistente de código | Busqué cada respuesta esperada en el PDF |
| 14–17/11 | Cuaderno | Claude Code en VS Code (modo plan), claude-sonnet-5-5 | Adaptar los cuadernos 07, 06 y 04 a tres documentos: carga, fragmentación, índice, recuperación, generación y auditoría con CSV | Aprobé cada plan, ejecuté cada celda, comparé el texto extraído con el PDF, corregí errores de sintaxis del código generado y escribí las observaciones y los diagnósticos |
| 16–17/11 | Interpretación | Claude (claude.ai), Claude Opus 5.5 | Discutir resultados, diagnosticar fallas (Ollama, índices duplicados) y redactar borradores de observaciones | Comprobé cada resultado en la salida del cuaderno; detecté y corregí frases de evidencia que daban falsos positivos y un chequeo de abstención que contaba mal una respuesta |
| 17/11 | Interfaz | Claude Code, claude-sonnet-5-5 | Adaptar `app.py` de clase al mismo pipeline del cuaderno | Pasé los 6 casos por `app.consultar` y probé la interfaz en el navegador |
| 17–18/11 | README | Claude (claude.ai), Claude Opus 5.5 | Borrador desde mis notas y el cuaderno | Revisé cada sección y seguí la instalación desde una copia nueva del repositorio |

### Anexo: pedidos principales a Claude Code

Trabajé por etapas, con modo plan: Claude Code proponía un plan, yo lo revisaba y recién después escribía el código. Las instrucciones generales (stack de clase, qué no usar, preguntas de prueba y método de auditoría) estaban en un archivo `CLAUDE.md` que no se publica.

| Momento | Pedido (resumido) |
|---|---|
| 13–14/11 | Etapa 1: adaptar el cuaderno 07 a los tres PDF, con el recorte de páginas, la carga y la revisión del texto extraído contra el PDF. |
| 16–17/11 | Sección 5: comparar los cortes 500/80 y 1.100/180 con la pregunta R3, como en el cuaderno 07, pero con K=3. |
| 17/11 | Secciones 6 y 7: índice persistente y recuperación, siguiendo la tabla de decisiones de la consigna (modelo de embeddings y k); nombre de colección con corte y modelo; tabla resumen de los 6 casos. |
| 17/11 | Sección 8: generación con el prompt de clase textual, sin modificarlo, y chequeo de abstención. |
| 17/11 | Sección 9: auditoría con una variable por vez y registro en `pruebas/auditoria.csv`; corrida 1 (500/80) y corrida 2 (1.100/180) con tabla comparativa. |
| 17/11 | `app.py` y sección 10: la interfaz Gradio con el mismo recorte, corte, K y prompt del cuaderno, probada con los 6 casos. |

**Cómo usé la IA para resolver un error: la indexación por lotes.** Al construir el índice, Ollama cortaba el proceso de `bge-m3` a mitad de camino (error de conexión en `/tokenize`). En lugar de pedirle a Claude Code que «lo arreglara», analicé el error con Claude (claude.ai): leímos el mensaje para entender qué parte fallaba (el proceso del modelo, no el servidor de Ollama), descartamos la memoria del equipo como causa y evaluamos opciones. Elegí enviar los fragmentos en lotes de 50 y, antes de aplicarlo, verifiqué que no alterara el corte: el lote solo agrupa fragmentos ya cortados para el envío, y cada vector depende únicamente del texto de su fragmento. Lo comprobé en el cuaderno: el índice quedó con la misma cantidad de fragmentos y las búsquedas devolvieron las mismas distancias que antes del cambio. Dejé comentada la versión original para que se vea el antes y el después.

Con el mismo criterio (leer el resultado, entender la causa y recién después cambiar el código) decidí y pedí los otros ajustes: cambiar las frases de evidencia de R1 y R3, que daban falsos positivos; usar igualdad exacta para la abstención; separar las celdas que generan respuestas de las que solo muestran tablas, y corregir errores de sintaxis del código generado.
