import os
from dotenv import load_dotenv, find_dotenv
import shutil
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from zhipuai_embedding import ZhipuAIEmbeddings
import faiss
import numpy as np
from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

# 读取本地/项目的环境变量
_ = load_dotenv(find_dotenv())
print("环境变量已加载。")

# 获取folder_path下所有文件路径，储存在file_paths里
file_paths = []
folder_path = 'papers'
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)
print(f"文件路径已获取")

# 遍历文件路径并把实例化的loader存放在loaders里
loaders = []
for file_path in file_paths:
    file_type = file_path.split('.')[-1]
    if file_type == 'pdf':
        loaders.append(PyMuPDFLoader(file_path))
    elif file_type == 'md':
        loaders.append(UnstructuredMarkdownLoader(file_path))
print(f"加载器已创建：{len(loaders)} 个加载器")

# 下载文件并存储到text
texts = []
for loader in loaders:
    texts.extend(loader.load())
print(f"文档已加载：{len(texts)} 个文档")

# 切分文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
split_docs = text_splitter.split_documents(texts)
print(f"文档已切分：{len(split_docs)} 个切片")

# 定义 Embeddings
embedding = ZhipuAIEmbeddings()
print("嵌入模型已加载。")

# 定义持久化路径
persist_directory = './vector_db/faiss_index'

def clear_persist_directory(directory):
    """
    清空并重新创建指定文件夹中的所有内容。
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)  # 删除整个文件夹及其内容
        print(f"已清空目录：{directory}")
    os.makedirs(directory)  # 重新创建文件夹
    print(f"已创建目录：{directory}")

# 清空持久化路径
clear_persist_directory(persist_directory)

##方法1
vectordb = FAISS.from_documents(
    documents=split_docs,
    embedding=embedding,
)
vectordb.save_local("vector_db/faiss_index")
print("向量库索引建立成功")

##方法2
# # 获取文档嵌入向量
# doc_vectors = [embedding.embed_query(doc.page_content) for doc in split_docs]
# print(f"文档嵌入已生成：{len(doc_vectors)} 个向量")

# # 转换为numpy数组
# doc_vectors = np.array(doc_vectors).astype("float32")
# print(f"向量已转换为 numpy 数组：形状为 {doc_vectors.shape}")

# # 定义FAISS索引
# d = doc_vectors.shape[1]  # 向量维度
# nlist = 50  # 聚类中心数量
# quantizer = faiss.IndexFlatL2(d)  # 使用L2距离的平面索引
# index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
# print("FAISS索引已定义。")
# index.nprobe = 2  # 查找聚类中心的个数

# # 训练索引
# index.train(doc_vectors)
# print("FAISS索引已训练。")

# # 添加向量到索引
# index.add(doc_vectors)
# print(f"向量已添加到索引：总共 {index.ntotal} 个向量")

# # 创建 Document 对象列表
# documents = [Document(page_content=doc.page_content, metadata={"source": doc.metadata["source"]}) for doc in split_docs]
# print("Document 对象已创建。")

# # 创建 InMemoryDocstore
# docstore = InMemoryDocstore()

# # 添加文档到 InMemoryDocstore
# for doc in documents:
#     docstore.add(doc.id, doc)  # 这里假设 Document 对象有一个唯一的 id 属性

# print("InMemoryDocstore 已创建并添加文档。")

# # 创建索引到 docstore 的映射
# index_to_docstore_id = {i: i for i in range(len(documents))}
# print("索引到 docstore 的映射已创建。")

# # 创建 FAISS 向量数据库
# vectordb = FAISS(embedding_function=embedding, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)
# print("FAISS 向量数据库已创建。")

# # 保存向量数据库到本地
# vectordb.save_local("vector_db/faiss_index")
# print("向量数据库已保存到本地。")

# # 打印向量库中存储的数量
# print(f"向量库中存储的数量：{index.ntotal}")