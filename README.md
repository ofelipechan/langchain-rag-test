# LangChain RAG

A conversational AI assistant built with LangChain that uses Retrieval-Augmented Generation (RAG) to provide intelligent responses based on a knowledge base of documents. This project creates an assistant that can answer questions using context from loaded documents.

**In this implementation, the assistant is configured to work with content in the `./data` folder, which currently contains the book of "Alice in Wonderland" for testing. It uses this content to provide accurate, context-aware responses about the story, characters, and events from the narrative.**

## Features

- **RAG-powered responses**: Uses ChromaDB vector store to retrieve relevant context from documents
- **Conversation history**: Maintains context across multiple interactions
- **Document processing**: Supports Markdown files with automatic chunking
- **OpenAI integration**: Uses GPT-4o-mini for intelligent responses
- **Similarity search**: Finds relevant document chunks based on user queries

## Prerequisites

- Python 3.13 or higher
- Poetry (for dependency management)
- OpenAI API key

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   ```

2. **Install dependencies using Poetry**
   ```bash
   poetry install
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=add-your-openai-api-key-here
   ```

## Usage

### 1. Prepare Your Documents

Place your documents (Markdown files) in the `./data` directory.

### 2. Create the Vector Database

Run the database creation script to process and index your documents:

```bash
poetry run python create_db.py
```

This will:
- Load all Markdown files from the `./data` directory
- Split them into chunks for better retrieval
- Create a ChromaDB vector store in the `./chroma` directory (that gets created automatically)

### 3. Start the Assistant

Run the main application:

```bash
poetry run python app.py
```

The assistant will:
- Load the vector database
- Start an interactive conversation
- Use RAG to provide context-aware responses
- Maintain conversation history

## Configuration

### Environment Variables

- `MY_OPENAI_API_KEY`: Your OpenAI API key (required)

### Customization

You can modify the following in the code:
- **Chunk size**: Adjust `chunk_size` and `chunk_overlap` in `create_db.py`
- **Similarity threshold**: Change the relevance score threshold in `app.py`
- **Model**: Switch to different OpenAI models in `app.py`
- **Temperature**: Adjust response creativity in `app.py`

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your `.env` file contains the correct OpenAI API key
2. **Missing Dependencies**: Run `poetry install` to install all required packages
3. **Document Loading Issues**: Check that your documents are in Markdown format and placed in the `./data` directory
