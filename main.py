
from fastapi import FastAPI
from pydantic import BaseModel

from llama_index.core import VectorStoreIndex, Document, Settings,PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
import os

# -----------------------------------
# INIT
# -----------------------------------
load_dotenv()
app = FastAPI()

# -----------------------------------
# LLM + EMBEDDING CONFIG (LlamaIndex)
# -----------------------------------
Settings.llm = OpenAI(model="gpt-4.1-nano")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# -----------------------------------
# NEO4J VECTOR STORE
# -----------------------------------
vector_store = Neo4jVectorStore(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database="ace8866f",
    index_name="company_documents",
    node_label="Document",
    embedding_property="embedding",
    text_property="text",
    embedding_dimension=1536
)

qa_prompt = PromptTemplate(
    """
You are a strict enterprise HR assistant.

## RULES:
- Use ONLY the provided context to answer questions
- Never hallucinate or make up information
- If the answer is not found in the context, respond with: "I don't know"

## RESPONSE FORMAT:
- Return answer in proper markdown
- Each bullet point MUST be on its OWN separate line
- NEVER put multiple bullets on the same line
- Always add a newline character after each bullet point
- Use **bold** for key terms and amounts

## STRICT EXAMPLE — FOLLOW EXACTLY:
- First point goes here
- Second point goes here
- Third point goes here

NOT like this:
- First point - Second point - Third point  ← THIS IS WRONG

Context:
{context_str}

Question:
{query_str}

## ANSWER:
Write each bullet point on a separate line.
If answer not found, say "I don't know".
If not found, say "I don't know".
"""

)
# -----------------------------------
# INDEX (CONNECTS TO EXISTING DATA)
# -----------------------------------
index = VectorStoreIndex.from_vector_store(vector_store)

query_engine = index.as_query_engine( text_qa_template=qa_prompt,similarity_top_k=3)

# -----------------------------------
# REQUEST MODEL
# -----------------------------------
class QuestionRequest(BaseModel):
    question: str


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------
# API
# -----------------------------------
@app.post("/ask")
async def ask(data: QuestionRequest):

    response = query_engine.query(data.question)

    return {
        "question": data.question,
        "answer": str(response)
    }

