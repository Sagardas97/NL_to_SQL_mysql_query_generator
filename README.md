# NL_to_SQL_mysql_query_generator

SQL Query Generator with RAG

This Python script generates and executes SQL SELECT queries for any database using natural language input. It leverages Retrieval-Augmented Generation (RAG) with FAISS vector store and a language model (OpenAI's GPT-5 or compatible xAI API) to create accurate MySQL queries. The pipeline dynamically fetches the database schema, validates generated queries, executes them, and summarizes results in plain English.

# Features

Dynamic Schema Fetching: Retrieves table names, columns, and foreign keys from the database.

RAG with FAISS: Uses embeddings to retrieve relevant schema information for query generation.

SQL Validation: Ensures queries are valid MySQL SELECT statements using sqlglot.

Result Summarization: Converts query results into human-readable summaries.

Error Handling: Includes logging and robust error management for database operations.

Schema Change Detection: Rebuilds the vector store only when the schema changes (using SHA-256 hashing).

The script supports only SELECT queries for safety.

# Requirements

Python 3.8+
Libraries: mysql-connector-python, sqlglot, langchain, langchain-openai, faiss-cpu, python-dotenv
MySQL database
OpenAI API key or xAI API key (see xAI API)

# Usage

Install dependencies: pip install -r requirements.txt
Set up .env with your MySQL credentials and API key.
Run the script: python generate_sql_query_v1.py
Example query: "Which customer paid the most and what product did they pay the most for per unit?"

# Notes

Ensure the database is accessible.


