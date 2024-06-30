## 安装依赖

创建 conda 虚拟环境

```
conda create -n llm-rag python=3.10
```

激活虚拟环境

```
conda activate llm-rag
```

切换到 readme 文件所在目录后执行

```
pip install -r requirements.txt
```

【此时可能还会有依赖不全的报错】

pip 和 conda 源配置查看此文：[anaconda 环境管理](https://www.yuque.com/u39067637/maezfz/syzlisxdbqmp7k6s)

## 运行环境

vscode + python 插件

## 运行代码

### 命令行中运行

1. 首先运行大模型科学检索文件  
   这个文件用来根据用户问题搜索和下载文献  
   下载好的 pdf 会存入当前目录下的 papers 文件夹中
2. 然后运行 embedding 文件，将 pdf 文件嵌入到向量数据库  
   这个数据库会建立在当前目录下的 vector_db 文件夹中
3. 最后运行检索问答文件，这里需要填入你的 openai api key，然后得到经过 rag 之后的回答

zhipuai_embedding 文件使用 langchain 来封装智谱的向量嵌入 api。  
zhipuai_llm 文件定义了一个 ZhipuAILLM 类，继承自 LLM，用于调用智谱 AI 的对话模型。

### 网页中运行

1. 在 mysite/api/views.py 中填入自己的 openai api key
2. 切换目录至 mysite 文件夹下（manage.py 文件所在目录）
3. 运行命令：python manage.py runserver
4. 打开浏览器，访问 http://127.0.0.1:8000/api/chat

目前网页中运行的是已经完成了向量的嵌入的，当前 rag 的关键词是"大模型，提示工程，零样本学习"
