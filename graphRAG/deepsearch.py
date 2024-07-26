
from langchain_community.utilities import SearxSearchWrapper
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.document_transformers import MarkdownifyTransformer
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_community.utilities import SearxSearchWrapper
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
import orjson
import loguru

searcher = SearxSearchWrapper(
    searx_host="https://searx.makelovenowar.win/search")

model = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
compressor = CrossEncoderReranker(model=model, top_n=5)
compressor_for_url = CrossEncoderReranker(model=model, top_n=10)
prompt_template = """
Use the following context as your learned knowledge, inside <context></context> XML tags and their sources.
<context>
    {context}
</context>

When answer to user:
- If you don't know, just say that you don't know.
- If you don't know when you are not sure, ask for clarification.
- If you think it's safe and necessary to make assumptions for reasoning, you can assume it after notice user.
Avoid mentioning that you obtained the information from the context.
And answer according to the language of the user's question.

Given the context information, answer the query.
Give user Reference info from context sources after the answer.

Query: {question}

Please response in this Answer Format:

###Information Collection

###Reasoning Process

###Verification

###Detailed Final Answer

###Reference

\n\nLet's think step by step and answer in Chinese.
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

md = MarkdownifyTransformer()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
)

llm = ChatOpenAI(model='AI4Chem/ChemLLM-20B-Chat-DPO',
                 base_url="https://api.chemllm.org/v1", openai_api_key='123')

huggingface_embeddings = HuggingFaceBgeEmbeddings(
    # alternatively use "sentence-transformers/all-MiniLM-l6-v2" for a light and faster experience.
    model_name="moka-ai/m3e-small",#sentence-transformers/all-MiniLM-L6-v2
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)


def receive_query():
    query = input("请输入您的查询: ")
    return query


def extract_keywords(text, nums=10):
#     message0 = [("system", "You are a helpful assistant that can design plans for Web search."), ('user',f'''Please devise a potential search plans based on the given text. Incorporate both individual queries and multi-keyword combinations to maximize the information gathered. Ensure that each combination is unique and doesn't overlap too much with others.

# {text}

# Focus on splitting the main query into simple sub-query keywords and then combine them effectively. Avoid creating combinations that overlap significantly, as this will yield less useful information.

# Do not give me the list, just give me Only One BEST plan.
# ''')]

#     hint = llm.invoke(message0).content
#     print(hint)

    message = [("system", "You are a helpful assistant that can extract useful keywords from a given text for web search"), ('user',f'''Please extract and paraphrase the top-{nums} concise qeustiones from the given text for web searches. Reply in the user's language.

    There may refer to hard multi-hop questiones that coupling a group of notions that makes hard to obtain related information, but it's easier answer this question by decouple them into sub-queries without overlaps.
                                                                                                                                                                                                                                                                                                                                                                                                                                                            
<text>{text}<text/>

Combine individual queries with multi-keyword combinations to gather more comprehensive information. Split the user's query into simple sub-query keywords and combine them to avoid overlaps between combinations, as combinations with too much overlap will yield zero useful information.

According to hints, Reply in the user's language. Use the prefix "### Keywords-" followed by a number for each combination, with commas separating the keywords.

Return Format:
### Keywords-0:''')]
    ans = llm.invoke(message).content.split('### Keywords-')[1:]
    ans = [i.split(':')[-1].strip().replace('\n', ' ').replace('-', ' ').replace('_', ' ').replace(',',' ')
           for i in ans]
    # if len(ans) == 1:
    #     ans = ans[0].split(' ')
    ans = [item for item in ans if not item.isspace() and item]
    return ans[:nums]


def search_keywords(keywords):
    loguru.logger.info(f"Searching for keywords: {keywords}")
    try:
        results = searcher.results(
            keywords,
            num_results=100,
            time_range="year",
            enabled_engines=["bing", "google", "duckduckgo", "qwant", "yahoo", '	wikipedia',
                            'wikidata', 'wolframalpha', 'semantic scholar', 'google scholar', 'arxiv', 'pubmed']
        )
        return results
    except:
        return []


def search_result_rerank(items, query):
    loguru.logger.info(
        f"Search results via query: {query} from {len(items)} items")
    docs_after_split = [Document(orjson.dumps(item)) for item in items]
    vectorstore = FAISS.from_documents(
        docs_after_split, huggingface_embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 50})
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor_for_url, base_retriever=retriever
    )
    return compression_retriever.invoke(query)


def fetch_web_pages(search_results, query):
    results = []
    url_visited = []
    urls = []
    if isinstance(search_results, list):
        for result_dict in search_results:
            for result in result_dict:
                if 'link' in result and result['link'] not in url_visited:
                    results.append(result)
                    url_visited.append(result['link'])
    else:
        for result in search_results:
            if 'link' in result and result['link'] not in url_visited:
                results.append(result)
                url_visited.append(result['link'])

    reranked_results = search_result_rerank(results, query)

    for result in reranked_results:
        urls.append(orjson.loads(result.page_content)['link'])

    pdf_urls = []

    for url in urls:
        if 'pdf' in url:
            pdf_urls.append(url)
            urls.remove(url)
    loguru.logger.info(f"Fetching {len(urls)} most related web pages")

    # loader = PlaywrightURLLoader(urls=urls)
    docs = []
    for url in urls:
        try:
            loader = AsyncHtmlLoader([url])
            docs.extend(loader.load())
        except:
            pass
    docs_transformed = md.transform_documents(docs)

    pdf_pages = []
    for pdf_url in pdf_urls:
        try:
            pdf_loader = loader = PyMuPDFLoader(pdf_url)
            pdf_docs = pdf_loader.load()
            pdf_docs_transformed = md.transform_documents(pdf_docs)
            pdf_pages.extend(list(pdf_docs_transformed))
        except:
            pass

    intergrated_search_results = "".join(
        [f"[{result['title']}]({result['link'].split('//')[-1].split('/')[0]})\n{result['snippet']}\n\n" for result in results])

    return list(docs_transformed) + pdf_pages + [Document(intergrated_search_results, metadata={'title': 'Search Results'})]


def format_docs(docs) -> str:
    """Convert Documents to a single string.:"""
    formatted = [
        f"Article Title: {doc.metadata['title']}\nArticle Snippet: {doc.page_content}"
        for doc in docs
    ]
    return "\n\n" + "\n\n".join(formatted)


def rag_search(docs_transformed, query):
    docs_after_split = text_splitter.split_documents(docs_transformed)
    loguru.logger.info(
        f"Building Vectorbase for {len(docs_transformed)} document pages")
    vectorstore = FAISS.from_documents(
        docs_after_split, huggingface_embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 100})
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    retrievalQA = (
        {"context": compression_retriever | format_docs,
            "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    result = retrievalQA.invoke(query)
    return result


def synthesis_keyword(keywords):
    keywords = " ".join(keywords)
    keywords = " ".join(set(keywords.split(' ')))
    return keywords


def main():
    query = receive_query()
    keywords = extract_keywords(query)
    search_results = [search_keywords(keyword) for keyword in keywords]
    syn_keywords = synthesis_keyword(keywords)
    docs_transformed = fetch_web_pages(search_results, syn_keywords)
    result = rag_search(docs_transformed, syn_keywords)
    print(result)


if __name__ == "__main__":
    main()
