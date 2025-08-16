import mysql.connector
import sqlglot
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI  # Or use xAI API wrapper (see https://x.ai/api)
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import hashlib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# LLM setup (replace with Grok API or valid OpenAI model)
llm = ChatOpenAI(model_name="gpt-5-mini-2025-08-07")  # Replaced fictional gpt-5 with gpt-4o

# Step 1: Dynamically fetch schema from classicmodels database
def get_schema(db_config):
    """
    Fetch table names, columns, and foreign keys from classicmodels database.
    Returns: List of schema descriptions and schema hash for change detection.
    """
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        schema = []
        
        # Get all tables
        cursor.execute("SELECT TABLE_NAME FROM information_schema.TABLES WHERE TABLE_SCHEMA = %s", (db_config["database"],))
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            
            # Get columns
            cursor.execute("""
                SELECT COLUMN_NAME, DATA_TYPE
                FROM information_schema.COLUMNS
                WHERE TABLE_NAME = %s AND TABLE_SCHEMA = %s
            """, (table_name, db_config["database"]))
            columns = cursor.fetchall()
            column_str = ", ".join([f"{col[0]} {col[1]}" for col in columns])
            
            # Get foreign keys
            cursor.execute("""
                SELECT COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
                FROM information_schema.KEY_COLUMN_USAGE
                WHERE TABLE_NAME = %s AND TABLE_SCHEMA = %s AND REFERENCED_TABLE_NAME IS NOT NULL
            """, (table_name, db_config["database"]))
            fks = cursor.fetchall()
            fk_str = ", ".join([f"{fk[0]} -> {fk[1]}.{fk[2]}" for fk in fks]) or "None"
            
            schema.append(f"Table {table_name}: columns {column_str}. Foreign Keys: {fk_str}")
        
        # Compute schema hash for change detection
        schema_str = "\n".join(sorted(schema))  # Sort for consistent hashing
        schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()
        
        cursor.close()
        conn.close()
        logger.info("Successfully fetched schema from classicmodels")
        return schema, schema_hash
    except mysql.connector.Error as e:
        logger.error(f"Failed to fetch schema: {e}")
        raise

# Step 2: Build or load RAG vector store for schema
def build_schema_vectorstore(table_descriptions, schema_hash, index_path="./faiss_index", force_rebuild=False):
    """
    Load existing FAISS index if available and schema hasn't changed (and force_rebuild=False).
    Otherwise, build and save new vector store.
    Returns: FAISS retriever
    """
    embeddings = OpenAIEmbeddings()  # Or HuggingFaceEmbeddings for local
    hash_file = os.path.join(index_path, "schema_hash.txt")
    
    # Check if index exists and schema hasn't changed
    if os.path.exists(index_path) and os.path.exists(hash_file) and not force_rebuild:
        with open(hash_file, "r") as f:
            stored_hash = f.read().strip()
        if stored_hash == schema_hash:
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Loaded existing FAISS index from {index_path}")
            return vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Build and save new vector store
    documents = [Document(page_content=desc) for desc in table_descriptions]
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(index_path)
    with open(hash_file, "w") as f:
        f.write(schema_hash)
    logger.info(f"Created and saved new FAISS index to {index_path}")
    
    return vectorstore.as_retriever(search_kwargs={"k": 5})

# Step 3: SQL generation with RAG
prompt = PromptTemplate(
    input_variables=["query", "relevant_schema"],
    template="""You are a SQL expert. Given this relevant schema subset ****Only output the SQL query, nothing else, dont need '```sql'****: {relevant_schema}
    Generate a valid MySQL SELECT query for: {query}
     
    Rules:
    1. Important: Use joins, subqueries if needed.
    2. Important: ***make sure the SQL query follows mysql rules correctly***
    3. User may for ask multiple outputs at once, use subqueries intelligently to generate correct answer.
    4. Use aliases on all tables and columns
    5. Make sure the used columns exits in said table, dont use wrong aliases"""
)

def generate_sql(nl_query, retriever):
    relevant_docs = retriever.get_relevant_documents(nl_query)
    relevant_schema = "\n".join([doc.page_content for doc in relevant_docs])
    print(relevant_schema)
    
    chain = prompt | llm | StrOutputParser()
    sql_query = chain.invoke({"query": nl_query, "relevant_schema": relevant_schema})
    # print(sql_query)
    logger.info(f"Generated SQL query: {sql_query}")
    return sql_query.strip()

# Step 4: Validate SQL query
def validate_sql(sql_query):
    try:
        parsed = sqlglot.parse_one(sql_query, read="mysql")
        if parsed.__class__.__name__ != "Select":
            raise ValueError("Only SELECT queries allowed (DML read-only).")
        if any(kw in sql_query.upper() for kw in ["DROP", "DELETE", "INSERT", "UPDATE"]):
            raise ValueError("Invalid DML operation.")
        logger.info("SQL query validated successfully")
        return True, "Valid"
    except Exception as e:
        logger.error(f"SQL validation failed: {e}")
        return False, str(e)

# Step 5: Execute query on classicmodels database
def execute_query(sql_query, db_config):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql_query)
        results = cursor.fetchall()
        logger.info("Successfully executed SQL query")
        return results
    except mysql.connector.Error as e:
        logger.error(f"Query execution failed: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

# Step 6: Summarize results
summary_prompt = PromptTemplate(
    input_variables=["results", "original_query"],
    template="""Summarize these query results in plain English for the question: {original_query}
    Results: {results}"""
)

def generate_summary(results, original_query):
    chain = summary_prompt | llm | StrOutputParser()
    summary = chain.invoke({"results": str(results), "original_query": original_query})
    logger.info("Generated summary from query results")
    return summary

# Full Pipeline
def run_pipeline(nl_query, db_config, index_path="./faiss_index", force_rebuild=False):
    # Fetch schema dynamically
    table_descriptions, schema_hash = get_schema(db_config)
    retriever = build_schema_vectorstore(table_descriptions, schema_hash, index_path, force_rebuild)
    
    # Generate and execute query
    sql = generate_sql(nl_query, retriever)
    # print('***', sql)
    is_valid, msg = validate_sql(sql)
    if not is_valid:
        return f"Invalid query: {msg}"
    try:
        results = execute_query(sql, db_config)
        summary = generate_summary(results, nl_query)
        return summary
    except mysql.connector.Error as e:
        return f"Execution error: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Replace with your MySQL credentials
    db_config = {
        "host": "localhost",
        "user": "root",  # Your MySQL user
        "password": "",  # Your MySQL password
        "database": "classicmodels"
    }
    nl_query = "Which customer paid most amount and its cutomer id, name and how much they paid, and the product they paid most for it per unit, its name and price per unit"
    result = run_pipeline(nl_query, db_config, force_rebuild=False)
    print(result)