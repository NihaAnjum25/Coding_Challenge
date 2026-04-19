from typing import Dict

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

from Day_33.config import GROQ_API_KEY, MODEL_NAME, TEMPERATURE, validate_config


class ChatMemoryStore:
    """
    Stores chat history for each session.
    This allows the bot to remember earlier messages during a conversation.
    """

    def __init__(self) -> None:
        self._store: Dict[str, InMemoryChatMessageHistory] = {}

    def get_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self._store:
            self._store[session_id] = InMemoryChatMessageHistory()
        return self._store[session_id]

    def clear_history(self, session_id: str) -> None:
        self._store[session_id] = InMemoryChatMessageHistory()


def create_llm() -> ChatGroq:
    """Create the Groq chat model."""
    validate_config()
    return ChatGroq(
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        groq_api_key=GROQ_API_KEY,
    )


def build_prompt() -> ChatPromptTemplate:
    """Prompt template that includes previous conversation history."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful conversational chatbot. Use the chat history to "
                "maintain context, answer naturally, and ask brief follow-up questions "
                "when the user's request is ambiguous.",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )


def create_chatbot(memory_store: ChatMemoryStore) -> RunnableWithMessageHistory:
    """
    Build a chatbot chain with memory support using LangChain's message history wrapper.
    """
    chain = build_prompt() | create_llm()

    return RunnableWithMessageHistory(
        chain,
        memory_store.get_history,
        input_messages_key="input",
        history_messages_key="history",
    )


def print_help() -> None:
    """Display supported chat commands."""
    print("Commands:")
    print("  /help   Show available commands")
    print("  /reset  Clear the current chat memory")
    print("  /exit   End the session\n")


def chat() -> None:
    """
    Run an interactive chatbot session with memory.

    System design:
    1. The user sends a message.
    2. LangChain injects prior conversation history into the prompt.
    3. The GenAI model generates a context-aware response.
    4. The new user/AI messages are automatically stored for future turns.
    """
    session_id = "day-35-demo"
    memory_store = ChatMemoryStore()
    chatbot = create_chatbot(memory_store)

    print("Conversational Chatbot with Memory")
    print("The bot remembers earlier messages in this session.")
    print_help()

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "/exit":
            print("Chat ended.")
            break

        if user_input.lower() == "/help":
            print_help()
            continue

        if user_input.lower() == "/reset":
            memory_store.clear_history(session_id)
            print("Bot: Memory cleared. We can start fresh.\n")
            continue

        response = chatbot.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        print(f"Bot: {response.content}\n")


if __name__ == "__main__":
    chat()
