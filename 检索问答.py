# 使用智谱 Embedding API，注意，需要将上一章实现的封装代码下载到本地
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from zhipuai_llm import ZhipuAILLM
from langchain.chains import RetrievalQA #这个方法在最新的langchain中已弃用，改为create_retrieval_chain
from langchain.chains import ConversationalRetrievalChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os


_ = load_dotenv(find_dotenv())    # read local .env file

# 定义 Embeddings
embedding = ZhipuAIEmbeddings()

# 向量数据库持久化路径
persist_directory = './vector_db/faiss_index'

# 加载数据库
vectordb = FAISS.load_local(persist_directory, embedding, allow_dangerous_deserialization=True)

# 获取环境变量 API_KEY
api_key = os.environ["ZHIPUAI_API_KEY"] #填写控制台中获取的 APIKey 信息

llm = ZhipuAILLM(model="glm-4", temperature=0.5, api_key=api_key)

response = llm.invoke("你好，请你自我介绍一下！")
print(response)

#如何实例化一个向量数据库为检索器https://python.langchain.com/v0.2/docs/how_to/vectorstore_retriever/
retriever=vectordb.as_retriever(
    search_kwargs={"k": 10},
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

##方法一
# # 创建一个 PromptTemplate
# prompt_template = """
# 你是一个知识丰富的助手，可以回答用户提出的各种问题。以下使用```包围起来的是用户提问的上下文和问题：

# 上下文：
# ```{context}```

# 问题：
# ```{question}```

# 请根据用户的聊天历史记录和提出的问题，综合你自己生成的内容和通过RAG检索到的英文信息，以准确、详细、丰富的方式回答，并注明检索到的信息来源，最后用中文回答。
# """
# chat_history = []
# PROMPT = PromptTemplate(
#     input_variables=["query","context"],
#     template=prompt_template
# )

# chain_type_kwargs = {"prompt": PROMPT}

# qa = RetrievalQA.from_chain_type(
#     llm=llm, 
#     chain_type="stuff", 
#     retriever=retriever, 
#     chain_type_kwargs=chain_type_kwargs,
#     return_source_documents=True
#     )

# query = input("请输入你的问题:\n")
# result = qa({"query": query})
# print(result["result"])

##方法二
#RAG简单实现：https://python.langchain.com/v0.2/docs/tutorials/rag/#go-deeper-4
prompt_template = '''
下面使用```包围起来的是检索到的英文上下文信息和用户的提问。
根据用户的问题，你需要重点使用下面给出的英文检索信息，并辅以你自己生成的内容进行回答。
以准确、详细、丰富的方式分点回答,但是不知道就说不知道。
对于使用到的上下文检索信息，找出其检索的文章来源，包括论文名字和论文网址，论文名一般是以pdf为后缀。
如果没有使用到检索的信息，就提示并未找到符合条件的检索信息。
使用中文回答。
检索到的信息:```{context}```
用户的提问:```{question}```
'''

prompt = PromptTemplate(
    input_variables=["question","context"],
    template=prompt_template
)

def format_docs(docs):
    #print("检索信息：",docs)
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# #创建一个链，用于将文档列表传递给模型。
# # https://api.python.langchain.com/en/latest/chains/langchain.chains.combine_documents.stuff.create_stuff_documents_chain.html
# question_answer_chain = create_stuff_documents_chain(llm, prompt)

query = input("请输入你的问题:\n")
# result=rag_chain.invoke(query)
# print(result)
#print(result['context'])


# 原本大模型的问答
prompt_template = """请回答下列问题:
                            {}""".format(query)
result=llm.invoke(prompt_template)
print(result)