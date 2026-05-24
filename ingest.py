import os
import fitz

from dotenv import load_dotenv

from openai import OpenAI

from neo4j import GraphDatabase

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================
# LOAD ENV
# ============================================================
load_dotenv()

# ============================================================
# OPENAI
# ============================================================
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# ============================================================
# NEO4J
# ============================================================
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(
        os.getenv("NEO4J_USERNAME"),
        os.getenv("NEO4J_PASSWORD")
    )
)

# ============================================================
# READ PDF
# ============================================================
doc = fitz.open("./policy.pdf")

full_text = ""

for page in doc:

    full_text += page.get_text()

print("PDF Loaded")

# ============================================================
# SMART CHUNKING
# ============================================================
splitter = RecursiveCharacterTextSplitter(

    chunk_size=1000,

    chunk_overlap=200,

    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

chunks = splitter.split_text(full_text)

print(f"Total Chunks: {len(chunks)}")

# ============================================================
# OPTIONAL CLEANUP
# ============================================================
with driver.session() as session:

    session.run(
        """
        MATCH (d:Document)
        DETACH DELETE d
        """
    )

print("Old Documents Deleted")

# ============================================================
# STORE CHUNKS
# ============================================================
with driver.session() as session:

    for i, chunk in enumerate(chunks):

        print(f"Embedding Chunk {i+1}")

        # ----------------------------------------------------
        # CREATE EMBEDDING
        # ----------------------------------------------------
        embedding_response = client.embeddings.create(

            model="text-embedding-3-small",

            input=chunk
        )

        embedding = embedding_response.data[0].embedding

        # ----------------------------------------------------
        # STORE IN NEO4J
        # ----------------------------------------------------
        session.run(
            """
            CREATE (d:Document {
                chunk_id: $chunk_id,
                text: $text,
                embedding: $embedding
            })
            """,

            {
                "chunk_id": i + 1,
                "text": chunk,
                "embedding": embedding
            }
        )

        print(f"Chunk {i+1} Stored")

print("✅ ALL DOCUMENTS STORED")