# Asistente Conversacional de PDFs con RAG y Groq

Este proyecto es una aplicación web desarrollada en Streamlit que permite interactuar con documentos PDF mediante chat, utilizando capacidades de Recuperación Aumentada por Recuperación (RAG) y el modelo Groq API. El sistema permite cargar, analizar y consultar múltiples PDFs, generando respuestas basadas en el contenido real de los documentos.

## Características principales

- Carga y gestión de múltiples archivos PDF.
- Extracción robusta de texto desde PDFs.
- Chat conversacional con contexto real de los PDFs cargados.
- Búsqueda semántica: el sistema utiliza embeddings y un vector store para encontrar los fragmentos más relevantes de todos los PDFs cargados.
- Siempre incluye al menos un fragmento de cada PDF en las respuestas generales, garantizando cobertura de todos los documentos.
- Análisis automático: resumen, comparación y clasificación temática de los documentos.
- Exportación del historial de chat en formato Markdown con un solo clic.
- Interfaz profesional, limpia y sin iconos ni emojis.

## Instalación
### Variables de entorno y claves API

El archivo `.env` (si lo usas) nunca se sube a GitHub por seguridad. Si necesitas que la app funcione en otro equipo, debes crear el archivo `.env` manualmente y agregar tu clave API de Groq, o ingresarla directamente en la interfaz de la app si está habilitado.

Ejemplo de `.env`:

```
GROQ_API_KEY=tu_clave_api_aqui
```

Comparte tu clave solo de forma privada y nunca la subas al repositorio.

1. Clona este repositorio:

	```sh
	git clone https://github.com/dchevesich/copiloto-pdf-rag.git
	cd copiloto-pdf-rag
	```


2. Instala las dependencias necesarias:

	```sh
	pip install -r requirements.txt
	pip install chromadb sentence-transformers
	```

3. Configura tus variables de entorno si es necesario (por ejemplo, claves de API para Groq).

4. Ejecuta la aplicación:

	```sh
	streamlit run app.py
	```

## Uso

1. Sube uno o varios archivos PDF desde la interfaz.
2. Interactúa con el asistente haciendo preguntas sobre el contenido de los PDFs.
3. Utiliza las funciones de análisis automático para obtener resúmenes, comparaciones y clasificaciones temáticas.
4. Exporta el historial del chat en formato Markdown con un solo clic.

## Estructura del proyecto

- `app.py`: Archivo principal de la aplicación Streamlit.
- `requirements.txt`: Lista de dependencias necesarias para el proyecto.
- `README.md`: Este archivo.

## Dependencias principales

El proyecto utiliza las siguientes librerías clave:

- `streamlit`: Interfaz web interactiva.
- `pypdf`: Extracción de texto de PDFs.
- `groq`: Acceso a la API de Groq para generación de respuestas.
- `sentence-transformers`: Generación de embeddings para búsqueda semántica.
- `numpy`: Procesamiento numérico y cálculo de similitud.
- `langchain`, `langchain-groq`: Orquestación de flujos RAG y manejo de contexto (opcional).
- `pandas`: Procesamiento y análisis de datos.
- `requests`: Llamadas HTTP a APIs externas.

Consulta `requirements.txt` para la lista completa y versiones exactas.

## Notas

- Asegúrate de tener configuradas las claves de API necesarias para el funcionamiento del modelo Groq.
- El sistema está diseñado para funcionar en entornos Windows, pero puede adaptarse a otros sistemas operativos.
- Si tienes problemas con la extracción de PDFs, revisa la integridad de los archivos y el soporte de la librería `pypdf`.

## Licencia

Este proyecto se distribuye bajo la licencia MIT.

---

## Ejecución con Docker

Puedes levantar el entorno completo usando Docker y docker-compose. Ya se incluyen los archivos `Dockerfile` y `docker-compose.yml` en el proyecto.

1. Asegúrate de tener Docker y docker-compose instalados.
2. Desde la raíz del proyecto, ejecuta:

	```sh
	docker-compose up --build
	```

Esto construirá la imagen y levantará el servicio en el puerto 8501 (por defecto para Streamlit). Accede a la aplicación en `http://localhost:8501`.

## Arquitectura del sistema

El sistema está compuesto por los siguientes módulos principales:

- **Frontend**: Interfaz web desarrollada en Streamlit para la carga de PDFs, interacción conversacional y visualización de análisis.
- **Backend/Orquestador**: Lógica de negocio en Python que gestiona la extracción de texto, vectorización, almacenamiento temporal y orquestación de prompts.
- **Modelo LLM**: Utiliza la API de Groq para la generación de respuestas contextuales.
- **Vector Store**: Vectorización y almacenamiento temporal en memoria para búsquedas semánticas rápidas usando `sentence-transformers` y `numpy`. Siempre incluye fragmentos de todos los PDFs cargados.

## Justificación de elecciones técnicas

- **Streamlit**: Permite construir prototipos de interfaces web de forma rápida y profesional, ideal para flujos conversacionales y visualización de resultados.
- **Groq API**: Proporciona acceso a modelos LLM de alto rendimiento, facilitando respuestas contextuales y extensibles.
- **LangChain**: Facilita la orquestación de flujos RAG y la integración de múltiples fuentes de datos y modelos.
- **pypdf**: Robusta para la extracción de texto de PDFs.
- **Python**: Ecosistema maduro para IA, NLP y prototipado rápido.

## Flujo conversacional y orquestación

1. El usuario sube hasta 5 archivos PDF.
2. El sistema extrae y divide el texto de cada PDF en fragmentos.
3. Se generan embeddings y se almacena todo en un vector store en memoria.
4. El usuario realiza preguntas en lenguaje natural.
5. El sistema busca los fragmentos más relevantes en los PDFs mediante similitud semántica y siempre incluye al menos un fragmento de cada PDF en preguntas generales.
6. El prompt se envía al modelo Groq, que responde considerando el contexto real de los documentos.
7. El usuario puede solicitar análisis automáticos (resumen, comparación, clasificación temática) y exportar el chat.

## Limitaciones actuales

- El almacenamiento vectorial es temporal y en memoria; no persiste tras reiniciar la app.
- El sistema está optimizado para PDFs en español y textos no escaneados.
- No incluye autenticación ni control de acceso.
- No soporta aún integración directa con bases de datos vectoriales externas.
- El análisis automático depende de la calidad del texto extraído.
- El sistema prioriza los fragmentos más relevantes, pero en preguntas generales siempre incluye todos los PDFs para mayor cobertura.

## Roadmap y mejoras futuras

- Integrar almacenamiento vectorial persistente (ChromaDB, Qdrant, etc.).
- Añadir autenticación de usuarios y gestión de sesiones.
- Mejorar la extracción de texto para PDFs escaneados (OCR).
- Permitir selección de diferentes modelos LLM.
- Añadir soporte para otros formatos de documentos (Word, txt).
- Mejorar la visualización de análisis y resultados.
# copiloto-pdf-rag