from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import anthropic
from anthropic import Anthropic
import chromadb
import json
import uvicorn
from pydantic import BaseModel
from typing import List
import os
import hashlib
import PyPDF2
import io
import shutil

from dotenv import load_dotenv
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialisation de Chroma
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("cours")

# Charger le fichier .env
load_dotenv()

# Lire la passkey
api_key = os.getenv('ANTHROPIC_API_KEY')

# Initialisation de l'API Anthropic
anthropic_client = Anthropic(api_key=api_key)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class Document(BaseModel):
    id: str
    name: str
    content: str
    path: str

documents = []

def add_to_database(document: str, metadata: dict, doc_id: str):
    collection.add(
        documents=[document],
        metadatas=[metadata],
        ids=[doc_id]
    )

def retrieve_relevant_info(query: str, n_results: int = 2) -> list:
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results['documents'][0]

def process_course(course_content: str, mode: str = "resume") -> dict:
    try:
        relevant_info = retrieve_relevant_info(course_content)
        relevant_info_str = " ".join(relevant_info)
        
        if mode == "resume":
            prompt = f"""
            Contenu du cours : {course_content}
            
            Informations supplémentaires : {relevant_info_str}
            
            En te basant sur ces informations, génère :
            1. Un résumé concis du cours
            2. 3 questions importantes sur le contenu
            3. Une explication d'un concept clé
            
            Fournis la réponse au format JSON avec les clés : 'resume', 'questions', 'explication'.
            """
        elif mode == "quiz":
            prompt = f"""
            Contenu du cours : {course_content}
            
            Génère un quiz de 5 questions à choix multiples basées sur ce contenu.
            Chaque question doit avoir 4 options et une seule bonne réponse.
            
            Fournis la réponse au format JSON avec la clé 'quiz' contenant une liste de dictionnaires,
            chacun avec les clés 'question', 'options' (liste de 4 chaînes) et 'correct_answer' (index de la bonne réponse).
            """
        elif mode == "evaluation":
            prompt = f"""
            Contenu du cours : {course_content}
            
            Génère une évaluation complète basée sur ce contenu, comprenant :
            1. 3 questions à réponse courte
            2. 2 questions à développement
            3. 1 exercice pratique
            
            Fournis la réponse au format JSON avec les clés : 'questions_courtes', 'questions_developpement', 'exercice_pratique'.
            """
        elif mode == "apprentissage":
            prompt = f"""
            Contenu du cours : {course_content}
            
            Crée un plan d'apprentissage détaillé basé sur ce contenu, comprenant :
            1. Les objectifs d'apprentissage
            2. Une liste de ressources supplémentaires
            3. Des exercices pratiques
            4. Un calendrier d'étude suggéré
            
            Fournis la réponse au format JSON avec les clés : 'objectifs', 'ressources', 'exercices', 'calendrier'.
            """
        
        response = anthropic_client.messages.create(
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="claude-3-opus-20240229",
        )
        
        result = json.loads(response.content)
        return result
    
    except Exception as e:
        print(f"Erreur lors du traitement du cours : {str(e)}")
        return {}

def get_file_hash(file_content: bytes) -> str:
    return hashlib.md5(file_content).hexdigest()

def convert_pdf_to_text(file_content: bytes) -> str:
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@app.get("/")
async def home(request: Request):
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    return templates.TemplateResponse("index.html", {"request": request, "api_key": api_key})

@app.post("/set_api_key")
async def set_api_key(request: Request, api_key: str = Form(...)):
    os.environ["ANTHROPIC_API_KEY"] = api_key
    return templates.TemplateResponse("index.html", {"request": request, "api_key": api_key, "message": "Clé API mise à jour avec succès!"})


@app.get("/chat")
async def chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/documents")
async def get_documents(request: Request):
    return templates.TemplateResponse("documents.html", {"request": request, "documents": documents})
@app.post("/add_document")
async def add_document(request: Request, file: UploadFile = File(...)):
    if file.filename.endswith(('.pdf', '.txt')):
        content = await file.read()
        file_hash = get_file_hash(content)
        
        # Vérifier si le document existe déjà
        existing_doc = next((doc for doc in documents if doc.id == file_hash), None)
        if existing_doc:
            return templates.TemplateResponse("documents.html", {"request": request, "documents": documents, "message": "Ce document existe déjà!"})
        
        if file.filename.endswith('.pdf'):
            content_str = convert_pdf_to_text(content)
        else:
            try:
                content_str = content.decode('utf-8')
            except UnicodeDecodeError:
                content_str = content.decode('latin-1')
        
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(content)
        
        
        doc = Document(id=file_hash, name=file.filename, content=content_str, path=file_path)
        documents.append(doc)
        add_to_database(doc.content, {"name": doc.name}, doc.id)
        return templates.TemplateResponse("documents.html", {"request": request, "documents": documents, "message": "Document ajouté avec succès!"})
    else:
        return templates.TemplateResponse("documents.html", {"request": request, "documents": documents, "message": "Erreur : Seuls les fichiers PDF et TXT sont acceptés"})


@app.post("/delete_document/{doc_id}")
async def delete_document(request: Request, doc_id: str):
    global documents
    doc_to_delete = next((doc for doc in documents if doc.id == doc_id), None)
    
    if doc_to_delete:
        # Supprimer le fichier du répertoire static
        if os.path.exists(doc_to_delete.path):
            os.remove(doc_to_delete.path)
        
        # Supprimer le fichier du répertoire uploads si nécessaire
        upload_path = os.path.join(UPLOAD_DIR, doc_to_delete.name)
        if os.path.exists(upload_path):
            os.remove(upload_path)
        
        # Supprimer le document de la liste documents
        documents = [doc for doc in documents if doc.id != doc_id]
        
        # Supprimer le document de la base de données Chroma
        try:
            collection.delete(ids=[doc_id])
        except Exception as e:
            print(f"Erreur lors de la suppression du document de Chroma : {str(e)}")
        
        message = "Document supprimé avec succès!"
    else:
        message = "Document non trouvé."
    
    return templates.TemplateResponse("documents.html", {"request": request, "documents": documents, "message": message})
@app.get("/view_document/{doc_id}")
async def view_document(request: Request, doc_id: str):
    doc = next((doc for doc in documents if doc.id == doc_id), None)
    if doc:
        return RedirectResponse(url=f"/uploads/{doc.name}")
    return {"error": "Document not found"}

@app.post("/process_query")
async def process_query(request: Request, query: str = Form(...), mode: str = Form(...)):
    result = process_course(query, mode)
    return templates.TemplateResponse("chat.html", {"request": request, "result": result, "mode": mode, "messages": [{"role": "user", "content": query}, {"role": "assistant", "content": json.dumps(result, indent=2)}]})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)