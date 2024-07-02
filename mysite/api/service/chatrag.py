from langchain_openai import ChatOpenAI
from .zhipuai_embedding import ZhipuAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

class ChatRAGService:
    def __init__(self, api_key_openai):
        self.api_key_openai = api_key_openai
        self.embedding = ZhipuAIEmbeddings()
        self.dbpath = 'api/service/vector_db/faiss_index' #根据managy.py所在目录为根目录确定
        self.vectordb = FAISS.load_local(self.dbpath, self.embedding, allow_dangerous_deserialization=True)
        self.llm = ChatOpenAI(model_name='gpt-3.5-turbo', api_key=api_key_openai, temperature=0.75)
        self.parser = StrOutputParser()
        
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 10})
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "根据最新的用户问题和聊天记录，可以引用聊天记录中相关的上下文，总结出一个可以理解的问题"),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ])
        self.history_aware_retriever = create_history_aware_retriever(self.llm, self.retriever, self.contextualize_q_prompt)
        
        self.store = {}
        
        self.prompt_template = (
            "你是一个基于RAG（检索增强生成）的资料搜索与生成机器人"
            "根据用户的问题，你需要使用下面给出的与问题匹配的RAG检索信息，并辅以你自己生成的内容进行回答。"
            "如何用户是向你问好并让你自我介绍，那么你只需要自我介绍，不要使用下面检索到的信息。"
            "如果使用到了下面提供的检索信息，你需要指出你是根据检索的信息给出的回答。例如'根据检索到的信息，我回答的是...'"
            "如果下面的检索信息不符合用户的提问时，你需要指出未找到符合条件的检索信息，然后你需要自己针对新问题生成内容。例如'未检索到相关信息，下面是我对这个问题的理解...'"
            "最后使用中文回答。"
            "\n检索到的信息：\n"
            "{context}"
            "\n用户的问题：\n"
            "{input}"
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt_template),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ])
        
        self.question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.chain = create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain)
        
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
            output_messages_key="answer",
        )

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def get_response_stream(self, query, session_id='default_session'):
        messages = {"input": query}
        config = {"configurable": {"session_id": session_id}}

        for chunk in self.chain_with_history.stream(messages, config=config):
            if 'answer' in chunk:
                yield chunk['answer']
