import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
load_dotenv()

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI # classe para os modelos de chat
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # cria os template de prompt | reserva espaço para o histórico de msg no prompt
from langchain_core.runnables.history import RunnableWithMessageHistory # gerenciador do histórico de conversa
from langchain_community.chat_message_histories import ChatMessageHistory # armazena o histórico da conversa em um formato estruturado
from langchain_chroma import Chroma  # Importa o ChromaDB para armazenamento de vetores
from langchain_openai import OpenAIEmbeddings

CHROMA_PATH = "./chroma" # Path to the ChromaDB database

def generate_assistant():
    template = """Answer the user question based on conversation history:
    
    {history}
    
    ---

    {context}


    Here's the user question: {input}
    """


    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")

    chain = prompt | llm

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    return chain_with_history



def start_assistant():
    print('welcome! I am your interview assistant. Type "exit" to end the session.')
    chat_assistant = generate_assistant()

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Ending session. Goodbye!")
            break

        results = db.similarity_search_with_relevance_scores(user_input, k=4)

        context_text = ""
        sources = ""
        if len(results) == 0 or results[0][1] < 0.7:
            print(f'Unable to find good matching results.')
        else:
            context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

            context_text = f"""
            and based on the information the following context:

            {context}
            ---
            """
            print(f"found {len(results)} context results")

            sources = [doc.metadata.get("source", None) for doc, _score in results]

        answer: AIMessage = chat_assistant.invoke(
            {"input": user_input, "context": context_text},
            config={'configurable': { 'session_id': 'interview_session_123' } }  # Use a fixed session ID for simplicity
        )
 
        print(f"Assistant: {answer.content}")
        if sources:
            print(f"Source: {sources}")

if __name__ == "__main__":
    start_assistant()