import os
from dotenv import load_dotenv, find_dotenv
import shutil
# 读取本地/项目的环境变量。
# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中  
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

# 如果你需要通过代理端口访问，你需要如下配置
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'

# 获取folder_path下所有文件路径，储存在file_paths里
file_paths = []
folder_path = 'papers'
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)
print(file_paths[:3])

from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader

# 遍历文件路径并把实例化的loader存放在loaders里
loaders = []

for file_path in file_paths:

    file_type = file_path.split('.')[-1]
    if file_type == 'pdf':
        loaders.append(PyMuPDFLoader(file_path))
    elif file_type == 'md':
        loaders.append(UnstructuredMarkdownLoader(file_path))

# 下载文件并存储到text
texts = []

for loader in loaders: texts.extend(loader.load())

text = texts[1]
print(f"每一个元素的类型：{type(text)}.", 
    f"该文档的描述性数据：{text.metadata}", 
    f"查看该文档的内容:\n{text.page_content[0:]}", 
    sep="\n------\n")

from langchain.text_splitter import RecursiveCharacterTextSplitter

# 切分文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000, chunk_overlap=250)

split_docs = text_splitter.split_documents(texts)

from zhipuai_embedding import ZhipuAIEmbeddings

# 定义 Embeddings
embedding = ZhipuAIEmbeddings()

# 定义持久化路径
persist_directory = './vector_db/chroma'
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

from langchain.vectorstores.chroma import Chroma

vectordb = Chroma.from_documents(
    documents=split_docs[:], # 为了速度，只选择前 20 个切分的 doc 进行生成；使用千帆时因QPS限制，建议选择前 5 个doc
    embedding=embedding,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)

vectordb.persist()

print(f"向量库中存储的数量：{vectordb._collection.count()}")