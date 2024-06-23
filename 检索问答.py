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
persist_directory = './vector_db/chroma'

# 加载数据库
vectordb = Chroma(
    persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
    embedding_function=embedding
)
print(f"向量库中存储的数量：{vectordb._collection.count()}")

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

llm = ZhipuAILLM(model="glm-4", temperature=0.95, api_key=api_key)

response = llm.invoke("你好，请你自我介绍一下！")
print(response)

# from langchain.prompts import PromptTemplate

# template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
# 案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
# {context}
# 问题: {question}
# """

# QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
#                                  template=template)

# from langchain.chains import RetrievalQA

# qa_chain = RetrievalQA.from_chain_type(llm,
#                                        retriever=vectordb.as_retriever(),
#                                        return_source_documents=True,
#                                        chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

# question_1 = "KNN分类的应用场景"
# question_2 = "What are some parallel algorithms that can compute exact k-nearest neighbors in low dimensions?"

# result = qa_chain({"query": question_1})
# print("大模型+知识库后回答 question_1 的结果：")
# print(result["result"])

# result = qa_chain({"query": question_2})
# print("大模型+知识库后回答 question_2 的结果：")
# print(result["result"])

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
    return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
)

from langchain.chains import ConversationalRetrievalChain

retriever=vectordb.as_retriever()

qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)
question1 = "Details on how to reduce the illusion of llm"
result = qa({"question": question1})
print(result['answer'])

# question2 = "Introduction to the principles and improvements of these algorithms"
# result = qa({"question": question2})
# print(result['answer'])


# prompt_template = """请回答下列问题:
#                             {}""".format(question_1)

# ### 基于大模型的问答
# result=llm.invoke(prompt_template)
# print(result)
prompt_template = """请回答下列问题:
                            {}""".format(question1)

### 基于大模型的问答
result=llm.invoke(prompt_template)
print(result)