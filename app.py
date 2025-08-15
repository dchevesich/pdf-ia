from dotenv import load_dotenv
load_dotenv()
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
# Vector store simple en memoria para fragmentos de PDF
class PDFVectorStore:
    """Almacena embeddings y fragmentos de texto para búsqueda semántica."""
    def __init__(self):
        self.texts = []
        self.embeddings = []
        # Usar modelo liviano compatible con HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    def _embed(self, texts):
        # Obtiene embeddings promedio de los tokens CLS
        with torch.no_grad():
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            outputs = self.model(**inputs)
            # Usar la media de los embeddings de la última capa oculta
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.cpu().numpy()

    def add_texts(self, texts):
        new_embs = self._embed(texts)
        self.texts.extend(texts)
        if len(self.embeddings) == 0:
            self.embeddings = list(new_embs)
        else:
            self.embeddings.extend(new_embs)

    def search(self, query, top_k=3):
        if not self.texts:
            return []
        query_emb = self._embed([query])[0]
        embs = np.vstack(self.embeddings)
        sims = np.dot(embs, query_emb) / (np.linalg.norm(embs, axis=1) * np.linalg.norm(query_emb) + 1e-8)
        idxs = np.argsort(sims)[::-1][:top_k]
        return [self.texts[i] for i in idxs]
import streamlit as st
import hashlib
import os
from typing import List, Dict, Set
import tempfile
import requests
import json
from datetime import datetime

class GroqChatManager:
    """Gestor de chat con Groq API"""
    
    def __init__(self, api_key: str = None):
        # API key debe ser proporcionada por variable de entorno o interfaz
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.3-70b-versatile"  # Modelo recomendado y gratuito en Groq

        # Inicializar chat history en session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    
    def is_configured(self) -> bool:
        """Verifica si Groq está configurado correctamente"""
        return self.api_key is not None and self.api_key.strip() != ""
    
    def call_groq_api(self, messages: List[Dict], temperature: float = 0.7) -> str:
        """Llama a la API de Groq"""
        if not self.is_configured():
            return " Error: API key de Groq no configurada"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1000,
            "stream": False
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError:
                # Mostrar el error detallado de la API si está disponible
                try:
                    error_detail = response.json()
                    return f"Error {response.status_code}: {error_detail}"
                except Exception:
                    return f"Error {response.status_code}: {response.text}"

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            return f" Error de conexión: {str(e)}"
        except KeyError as e:
            return f" Error en respuesta de API: {str(e)}"
        except Exception as e:
            return f" Error inesperado: {str(e)}"
    
    def build_context_from_pdfs(self, query: str) -> str:
        """Construye contexto usando búsqueda semántica y asegura al menos un fragmento de cada PDF cargado."""
        vector_store = st.session_state.get('vector_store')
        uploaded_pdfs = st.session_state.get('uploaded_pdfs')
        if vector_store and uploaded_pdfs:
            # Fragmentos más relevantes por similitud
            top_chunks = vector_store.search(query, top_k=5)
            # Asegurar al menos un fragmento de cada PDF
            extra_chunks = []
            for pdf in uploaded_pdfs.values():
                text = pdf.get('text', '')
                if text:
                    chunk = text[:500]
                    if chunk not in top_chunks:
                        extra_chunks.append(chunk)
            # Unir y deduplicar
            all_chunks = top_chunks + [c for c in extra_chunks if c not in top_chunks]
            context = "\n---\n".join(all_chunks)
            return f"Contexto relevante extraído de los PDFs:\n{context}"
        else:
            return "No hay documentos PDF cargados para consultar."
    
    def format_file_size(self, size_bytes: int) -> str:
        """Convierte bytes a formato legible"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        else:
            return f"{size_bytes/(1024**2):.1f} MB"
    
    def add_message_to_history(self, role: str, content: str):
        """Añade mensaje al historial de chat"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.chat_history.append({
            "role": role,
            "content": content,
            "timestamp": timestamp
        })
    
    def get_conversation_context(self, max_messages: int = 10) -> List[Dict]:
        """Obtiene el contexto de conversación para la API"""
        messages = []
        
        # Mensaje del sistema
        system_message = {
            "role": "system",
            "content": """Eres un asistente conversacional especializado en análisis de documentos PDF. 
            Puedes responder preguntas generales y también consultar documentos cuando estén disponibles.
            Siempre sé claro sobre qué documentos tienes disponibles y sus limitaciones actuales."""
        }
        messages.append(system_message)
        
        # Añadir historial reciente (solo contenido, sin timestamps para la API)
        recent_history = st.session_state.chat_history[-max_messages:] if st.session_state.chat_history else []
        for msg in recent_history:
            if msg["role"] in ["user", "assistant"]:
                messages.append({
                    "role": msg["role"], 
                    "content": msg["content"]
                })
        
        return messages
    
    def render_chat_interface(self):
        """Renderiza la interfaz de chat sin iconos ni emojis"""
    # Encabezado eliminado por solicitud del usuario

        # Configuración de API Key
        if not self.is_configured():
            st.error("API Key de Groq requerida")
            with st.expander("Configurar Groq API", expanded=True):
                st.markdown("""
                **Pasos para obtener tu API Key gratuita:**
                1. Ve a [console.groq.com](https://console.groq.com)
                2. Crea una cuenta gratuita
                3. Ve a 'API Keys' y crea una nueva key
                4. Pega tu key abajo

                **Límites gratuitos**: 6,000 tokens/minuto, muy generoso para pruebas.
        # Poblar vector store con fragmentos de texto de los PDFs cargados
        if st.session_state.get('uploaded_pdfs'):
            all_texts = []
            for pdf in st.session_state['uploaded_pdfs'].values():
                text = pdf.get('text', '')
                if text:
                    # Dividir en fragmentos de 500 caracteres
                    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                    all_texts.extend(chunks)
            if all_texts:
                st.session_state['vector_store'].add_texts(all_texts)
                """)
                api_key_input = st.text_input(
                    "Pega tu API Key de Groq:",
                    type="password",
                    help="Tu API key se mantendrá solo durante esta sesión"
                )
                if st.button("Guardar API Key"):
                    if api_key_input.strip():
                        st.session_state.groq_api_key = api_key_input.strip()
                        self.api_key = api_key_input.strip()
                        st.success("API Key guardada para esta sesión")
                        st.rerun()
                    else:
                        st.error("Por favor ingresa una API Key válida")
            return

    # Estado de documentos: mensajes eliminados por solicitud del usuario

        # Historial de chat
        st.subheader("Conversación")

        # Container para mensajes
        chat_container = st.container()
        with chat_container:
            if st.session_state.chat_history:
                for i, message in enumerate(st.session_state.chat_history):
                    timestamp = message.get("timestamp", "")
                    if message["role"] == "user":
                        st.chat_message("user").write(f"[{timestamp}] {message['content']}")
                    else:
                        st.chat_message("assistant").write(f"[{timestamp}] {message['content']}")
            else:
                st.info("Puedes preguntarme sobre los documentos que has subido o hacer preguntas generales.")

        # Input para nueva consulta
        st.subheader("Tu Consulta")

    # Expander de ejemplos de preguntas eliminado por solicitud del usuario

        # Área de input
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_area(
                "Escribe tu pregunta:",
                height=100,
                placeholder="Ej: ¿Qué documentos tienes disponibles? o ¿Puedes explicarme sobre IA?",
                key="user_input"
            )
        with col2:
            st.write("") # Spacer
            st.write("") # Spacer
            send_button = st.button(
                "Enviar",
                type="primary",
                use_container_width=True
            )
            clear_button = st.button(
                "Limpiar Chat",
                use_container_width=True
            )

        # Procesar input
        if send_button and user_input.strip():
            # Añadir pregunta del usuario al historial
            self.add_message_to_history("user", user_input.strip())
            # Mostrar loading
            with st.spinner("Pensando..."):
                # Construir contexto con PDFs
                context = self.build_context_from_pdfs(user_input.strip())
                # Preparar mensajes para la API
                messages = self.get_conversation_context()
                # Añadir contexto actual
                messages.append({
                    "role": "user",
                    "content": context
                })
                # Llamar a Groq API
                response = self.call_groq_api(messages)
                # Añadir respuesta al historial
                self.add_message_to_history("assistant", response)
            # Limpiar input y refrescar
            st.rerun()

        # Limpiar chat
        if clear_button:
            st.session_state.chat_history = []
            st.success("Chat limpiado")
            st.rerun()

    # ...existing code...

# Clase principal actualizada que combina PDFs + Chat
class PDFUploadManager:
    """Gestor de carga de PDFs con validación de duplicados y límite de archivos"""
    
    def __init__(self, max_files: int = 5):
        self.max_files = max_files
        
        # Inicializar session state
        if 'uploaded_pdfs' not in st.session_state:
            st.session_state.uploaded_pdfs = {}
        if 'pdf_hashes' not in st.session_state:
            st.session_state.pdf_hashes = set()
        if 'pdf_names' not in st.session_state:
            st.session_state.pdf_names = set()
        if 'upload_messages' not in st.session_state:
            st.session_state.upload_messages = []
        
        # Inicializar API key desde session state si existe
        if 'groq_api_key' in st.session_state:
            os.environ['GROQ_API_KEY'] = st.session_state.groq_api_key
    
    def calculate_file_hash(self, file_content: bytes) -> str:
        """Calcula el hash SHA-256 del contenido del archivo"""
        return hashlib.sha256(file_content).hexdigest()
    
    def is_duplicate(self, file_name: str, file_content: bytes) -> tuple[bool, str]:
        """
        Verifica si el archivo es duplicado
        Returns: (is_duplicate, reason)
        """
        file_hash = self.calculate_file_hash(file_content)
        
        # Verificar duplicado por nombre
        if file_name in st.session_state.pdf_names:
            return True, f"Ya existe un archivo con el nombre '{file_name}'"
        
        # Verificar duplicado por contenido
        if file_hash in st.session_state.pdf_hashes:
            return True, f"Ya existe un archivo con el mismo contenido que '{file_name}'"
        
        return False, ""
    
    def add_pdf(self, uploaded_file) -> bool:
        """
        Añade un PDF al sistema si pasa todas las validaciones
        Returns: True si se añadió exitosamente, False si no
        """
        if len(st.session_state.uploaded_pdfs) >= self.max_files:
            self.add_message("error", f"❌ Límite máximo alcanzado: {self.max_files} archivos")
            return False
        
        file_content = uploaded_file.read()
        uploaded_file.seek(0)  # Reset file pointer
        
        # Verificar si es duplicado
        is_dup, reason = self.is_duplicate(uploaded_file.name, file_content)
        if is_dup:
            self.add_message("warning", f" Archivo duplicado: {reason}")
            return False
        
        # Guardar archivo temporalmente y añadir a session state
        file_hash = self.calculate_file_hash(file_content)
        
        # Extraer texto del PDF
        try:
            import PyPDF2
            from io import BytesIO
            reader = PyPDF2.PdfReader(BytesIO(file_content))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            text = text.strip()
        except Exception as e:
            text = f"[Error al extraer texto: {e}]"

        st.session_state.uploaded_pdfs[uploaded_file.name] = {
            'content': file_content,
            'hash': file_hash,
            'size': len(file_content),
            'type': uploaded_file.type,
            'text': text
        }
        
        st.session_state.pdf_hashes.add(file_hash)
        st.session_state.pdf_names.add(uploaded_file.name)
        
        return True
    
    def remove_pdf(self, file_name: str):
        """Elimina un PDF del sistema"""
        if file_name in st.session_state.uploaded_pdfs:
            file_info = st.session_state.uploaded_pdfs[file_name]
            # Remover de todos los conjuntos de seguimiento
            st.session_state.pdf_hashes.discard(file_info['hash'])
            st.session_state.pdf_names.discard(file_name)
            del st.session_state.uploaded_pdfs[file_name]
            self.add_message("success", f"Archivo '{file_name}' eliminado")
            st.rerun()
    
    def add_message(self, msg_type: str, message: str):
        """Añade un mensaje persistente"""
        st.session_state.upload_messages.append({
            'type': msg_type,
            'message': message
        })
        # Mantener solo los últimos 5 mensajes
        if len(st.session_state.upload_messages) > 5:
            st.session_state.upload_messages = st.session_state.upload_messages[-5:]
    
    def display_messages(self):
        """Muestra los mensajes persistentes"""
        if st.session_state.upload_messages:
            for msg in st.session_state.upload_messages:
                if msg['type'] == 'success':
                    st.success(msg['message'])
                elif msg['type'] == 'warning':
                    st.warning(msg['message'])
                elif msg['type'] == 'error':
                    st.error(msg['message'])
            # Botón para limpiar mensajes
            if st.button("Limpiar Mensajes", key="clear_messages"):
                st.session_state.upload_messages = []
                st.rerun()
    
    def get_upload_status(self) -> Dict:
        """Retorna el estado actual de carga"""
        current_count = len(st.session_state.uploaded_pdfs)
        return {
            'current_count': current_count,
            'max_files': self.max_files,
            'is_full': current_count >= self.max_files,
            'remaining_slots': self.max_files - current_count
        }
    
    def format_file_size(self, size_bytes: int) -> str:
        """Convierte bytes a formato legible"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        else:
            return f"{size_bytes/(1024**2):.1f} MB"
    
    def render_upload_interface(self):
        """Renderiza la interfaz completa de carga"""
        st.header(" Cargador de Documentos PDF")
        
        # Mostrar mensajes persistentes
        self.display_messages()
        
        # Solo el botón de limpiar todo
        status = self.get_upload_status()
        if st.button("Limpiar Todo", disabled=status['current_count'] == 0):
            st.session_state.uploaded_pdfs = {}
            st.session_state.pdf_hashes = set()
            st.session_state.pdf_names = set()
            self.add_message("success", "Todos los archivos eliminados")
            st.rerun()
        
        # File uploader
        st.subheader("Cargar Archivos")
        
        # Usar key dinámica para limpiar el widget cuando sea necesario
        if 'uploader_key' not in st.session_state:
            st.session_state.uploader_key = 0
        
        # Deshabilitar uploader si está lleno
        if not status['is_full']:
            uploaded_files = st.file_uploader(
                label=f"Selecciona hasta {status['remaining_slots']} archivo(s) PDF",
                type=['pdf'],
                accept_multiple_files=True,
                key=f"pdf_uploader_{st.session_state.uploader_key}",
                help="Solo se aceptan archivos PDF. No se permiten duplicados."
            )
            
            # Procesar archivos subidos
            if uploaded_files:
                files_processed = False
                for uploaded_file in uploaded_files:
                    if len(st.session_state.uploaded_pdfs) < self.max_files:
                        result = self.add_pdf(uploaded_file)
                        if result:  # Solo marcar como procesado si se añadió exitosamente
                            files_processed = True
                    else:
                        self.add_message("warning", f"⚠️ Se alcanzó el límite máximo. '{uploaded_file.name}' no fue cargado.")
                        break
                
                # Limpiar el uploader después de procesar archivos
                if files_processed or any(uploaded_files):
                    # Forzar limpieza del widget creando una nueva key
                    if 'uploader_key' not in st.session_state:
                        st.session_state.uploader_key = 0
                    st.session_state.uploader_key += 1
                    st.rerun()
        else:
            st.warning(" No se pueden cargar más archivos. Elimina algunos para continuar.")
        
        # Lista de archivos cargados
        if st.session_state.uploaded_pdfs:
            st.subheader("Archivos Cargados")
            for i, (file_name, file_info) in enumerate(st.session_state.uploaded_pdfs.items(), 1):
                col1, col2, col3, col4 = st.columns([0.5, 3, 2, 1])
                with col1:
                    st.write(f"{i}.")
                with col2:
                    st.write(f"{file_name}")
                with col3:
                    st.write(f"{self.format_file_size(file_info['size'])}")
                with col4:
                    if st.button("Eliminar", key=f"remove_{file_name}", help=f"Eliminar {file_name}"):
                        self.remove_pdf(file_name)

            # Poblar vector store con fragmentos de texto de todos los PDFs cargados (sin duplicar)
            vector_store = st.session_state['vector_store']
            vector_store.texts = []
            vector_store.embeddings = []
            all_texts = []
            for pdf in st.session_state['uploaded_pdfs'].values():
                text = pdf.get('text', '')
                if text:
                    # Dividir en fragmentos de 500 caracteres
                    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                    all_texts.extend(chunks)
            if all_texts:
                vector_store.add_texts(all_texts)
        # Si no hay archivos cargados, no mostrar nada

def main():
    # Inicializar vector store global (en memoria) si no existe
    if 'vector_store' not in st.session_state:
        st.session_state['vector_store'] = PDFVectorStore()
    """Función principal"""
    st.set_page_config(
        page_title="PDF Chat Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("🤖 Asistente Conversacional de PDFs")
    st.markdown("---")
    # Crear instancias
    pdf_manager = PDFUploadManager(max_files=5)
    chat_manager = GroqChatManager()
    # Layout en columnas
    col1, col2 = st.columns([1, 1])
    with col1:
        pdf_manager.render_upload_interface()
    with col2:
        chat_manager.render_chat_interface()
        # Exportar chat como Markdown: descarga directa al hacer click
        import base64
        def get_download_link(data, filename, mime):
            b64 = base64.b64encode(data.encode()).decode()
            return f'<a href="data:{mime};base64,{b64}" download="{filename}">Descargar chat como Markdown</a>'

        if st.button("Exportar chat como Markdown", key="exportar_chat_md_btn2"):
            chat_md = "# Historial de Chat\n\n"
            for msg in st.session_state.get('chat_history', []):
                role = "Usuario" if msg['role'] == 'user' else "Asistente"
                chat_md += f"**{role} [{msg['timestamp']}]**:\n{msg['content']}\n\n"
            download_link = get_download_link(chat_md, "chat_completo.md", "text/markdown")
            st.markdown(download_link, unsafe_allow_html=True)

    # Sección de análisis automático
    st.markdown("---")
    st.subheader("Análisis Automático de PDFs")
    resumen_md = None
    if st.button("Resumen de PDFs"):
        if st.session_state.get('uploaded_pdfs'):
            resumen_prompt = "Haz un resumen claro y profesional de todos los documentos cargados."
            context = chat_manager.build_context_from_pdfs(resumen_prompt)
            messages = chat_manager.get_conversation_context()
            messages.append({"role": "user", "content": context})
            resumen = chat_manager.call_groq_api(messages)
            resumen_md = f"# Resumen de PDFs\n\n{resumen}"
            st.text_area("Resumen generado:", value=resumen_md, height=300)
        else:
            st.info("No hay PDFs cargados para resumir.")
    if resumen_md:
        st.download_button(
            label="Descargar resumen (.md)",
            data=resumen_md,
            file_name="resumen_pdfs.md",
            mime="text/markdown"
        )

    comparar_md = None
    if st.button("Comparar PDFs"):
        if st.session_state.get('uploaded_pdfs') and len(st.session_state.uploaded_pdfs) > 1:
            comparar_prompt = "Compara de forma clara y profesional los documentos cargados. Indica similitudes, diferencias y aspectos destacados de cada uno. No uses tablas."
            context = chat_manager.build_context_from_pdfs(comparar_prompt)
            messages = chat_manager.get_conversation_context()
            messages.append({"role": "user", "content": context})
            comparacion = chat_manager.call_groq_api(messages)
            comparar_md = f"# Comparación de PDFs\n\n{comparacion}"
            st.text_area("Comparación generada:", value=comparar_md, height=300)
        else:
            st.info("Debes cargar al menos 2 PDFs para comparar.")
    if comparar_md:
        st.download_button(
            label="Descargar comparación (.md)",
            data=comparar_md,
            file_name="comparacion_pdfs.md",
            mime="text/markdown"
        )

    clasificacion_md = None
    if st.button("Clasificar por temas"):
        if st.session_state.get('uploaded_pdfs'):
            clasif_prompt = "Analiza y clasifica los documentos cargados por temas principales y tópicos relevantes. Presenta los temas detectados para cada documento y los temas comunes entre ellos."
            context = chat_manager.build_context_from_pdfs(clasif_prompt)
            messages = chat_manager.get_conversation_context()
            messages.append({"role": "user", "content": context})
            clasificacion = chat_manager.call_groq_api(messages)
            clasificacion_md = f"# Clasificación por Temas\n\n{clasificacion}"
            st.text_area("Clasificación generada:", value=clasificacion_md, height=300)
        else:
            st.info("No hay PDFs cargados para clasificar.")
    if clasificacion_md:
        st.download_button(
            label="Descargar clasificación (.md)",
            data=clasificacion_md,
            file_name="clasificacion_pdfs.md",
            mime="text/markdown"
        )

if __name__ == "__main__":
    main()