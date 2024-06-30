from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

class ChatGPTService:
    def __init__(self, api_key):
        self.api_key = api_key
        self.llm = ChatOpenAI(model_name='gpt-3.5-turbo', api_key=api_key, temperature=0.8)
        self.parser = StrOutputParser()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You're an assistant who's good at math"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
        self.store = {}
        self.chain = self.prompt | self.llm | self.parser
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def get_response_stream(self, query, session_id='default_session'):
        messages = {"question": query}
        config = {"configurable": {"session_id": session_id}}

        for chunk in self.chain_with_history.stream(messages, config=config):
            yield chunk
