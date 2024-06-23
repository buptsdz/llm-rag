# 使用智谱 Embedding API，注意，需要将上一章实现的封装代码下载到本地
from zhipuai_embedding import ZhipuAIEmbeddings

from langchain.vectorstores.chroma import Chroma

from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())    # read local .env file
zhipuai_api_key = os.environ['ZHIPUAI_API_KEY']

# 定义 Embeddings
embedding = ZhipuAIEmbeddings()

# 向量数据库持久化路径
persist_directory = './vector_db/faiss_index'

from langchain.vectorstores import FAISS
# 加载数据库
vectordb = FAISS.load_local(persist_directory, embedding,allow_dangerous_deserialization=True)

# question = "What is the KNN algorithm?"
# docs = vectordb.similarity_search(question,k=3)
# print(f"检索到的内容数：{len(docs)}")

# #打印检索到的内容
# for i, doc in enumerate(docs):
#     print(f"检索到的第{i}个内容: \n {doc.page_content}", end="\n-----------------------------------------------------\n")

# 自定义chatglm接入 langchain，需要下载源码
from zhipuai_llm import ZhipuAILLM
from dotenv import find_dotenv, load_dotenv
import os

_ = load_dotenv(find_dotenv())

# 获取环境变量 API_KEY
api_key = os.environ["ZHIPUAI_API_KEY"] #填写控制台中获取的 APIKey 信息

llm = ZhipuAILLM(model="glm-4", temperature=0.5, api_key=api_key)

response = llm.invoke("你好，请你自我介绍一下！")
print(response)

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 创建一个 PromptTemplate
prompt_template = """
你是一个知识丰富的助手，可以回答用户提出的各种问题。以下是用户的聊天历史记录和新问题：

聊天历史记录：
{chat_history}

上下文：
{context}

新问题：
{question}

请根据聊天历史记录和新问题，整合你自身生成的内容和通过RAG检索到的英文信息，提供详细且准确的回答，并用中文回答。
"""
chat_history = []
PROMPT = PromptTemplate(
    input_variables=["chat_history", "question","context"],
    template=prompt_template
)

from langchain.chains import ConversationalRetrievalChain

retriever=vectordb.as_retriever(prompt=PROMPT,search_kwargs={"k": 3})

qa = ConversationalRetrievalChain.from_llm(
    llm,
    chain_type="stuff",
    retriever=retriever,
    memory=memory,
)

query = input("请输入你的问题:\n")
result = qa({"question": query})
# 提取主要的回答和需要的其他信息
# 打印回答
print(result['answer'])



# question2 = "Introduction to the principles and improvements of these algorithms"
# result = qa({"question": question2})
# print(result['answer'])


# prompt_template = """请回答下列问题:
#                             {}""".format(question_1)

# ### 基于大模型的问答
# result=llm.invoke(prompt_template)
# print(result)
# prompt_template = """请回答下列问题:
#                             {}""".format(question1)

# ### 基于大模型的问答
# result=llm.invoke(prompt_template)
# print(result)