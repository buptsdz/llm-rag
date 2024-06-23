## 安装依赖

创建 conda 虚拟环境

```
conda create -n llm-rag python=3.10
```

激活虚拟环境

```
conda activate llm-rag
```

切换到当前文件目录后执行

```
pip install -r requirements.txt
```

pip 和 conda 源配置查看此文：[anaconda 环境管理](https://www.yuque.com/u39067637/maezfz/syzlisxdbqmp7k6s)

## 运行代码

1. 首先运行大模型科学检索文件
   这个文件用来根据用户问题搜索和下载文献
2. 然后运行 embedding 文件，将 pdf 文件嵌入到向量数据库
3. 最后运行检索问答文件，能得到经过 rag 之后的回答。

zhipuai_embedding 文件使用 langchain 来封装智谱的向量嵌入 api
zhipuai_llm 文件定义了一个 ZhipuAILLM 类，继承自 LLM，用于调用智谱 AI 的对话模型。
