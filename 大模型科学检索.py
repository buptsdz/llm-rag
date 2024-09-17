import os
import sys
from dotenv import load_dotenv, find_dotenv
from zhipuai import ZhipuAI
from scholarly import scholarly, ProxyGenerator
import requests
from tqdm import tqdm
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from requests.exceptions import RequestException

# 读取本地/项目的环境变量
_ = load_dotenv(find_dotenv())

# 初始化智谱AI客户端
client = ZhipuAI(api_key=os.environ["ZHIPUAI_API_KEY"])

def gen_glm_params(prompt):
    messages = [{"role": "user", "content": prompt}]
    return messages

def get_completion(prompt, model="glm-4", temperature=0.95):
    '''
    使用智谱AI的API生成query关键词
    '''
    messages = gen_glm_params(prompt)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    if len(response.choices) > 0:
        return response.choices[0].message.content
    return "generate answer error"

def search_papers(query, site, total_papers):
    print(f"正在搜索关于 '{query}' 在 {site} 上的最新研究论文...")
    
    # 设置代理地址为本地 Clash for Windows 的地址和端口
    pg = ProxyGenerator()
    pg.SingleProxy('127.0.0.1:7890')

    # 应用代理设置
    scholarly.use_proxy(pg)

    # 构建搜索查询，指定特定网站
    search_query = scholarly.search_pubs(f"{query} site:{site}")
    papers = []

    try:
        with tqdm(total=total_papers, desc="收集论文进度", unit="篇") as pbar:
            for i, paper in enumerate(search_query, start=1):
                # 确保论文来自指定的网站
                if site in paper['pub_url']:
                    # 只保存标题、摘要、URL和出版年份
                    try:
                        papers.append({
                            'title': paper['bib']['title'],
                            'abstract': paper['bib'].get('abstract', 'No abstract available'),
                            'url': paper.get('pub_url', 'No URL available'),
                            'year': paper['bib'].get('pub_year', None)
                        })
                    except KeyError as e:
                        print(f"处理论文时出现键错误: {e}")
                        continue  # 发生错误时跳过当前论文

                    # 更新进度条
                    pbar.update(1)
                    if i >= total_papers:
                        break

    except RequestException as e:
        print(f"搜索请求失败: {e}")
        return []  # 网络请求失败时返回空列表

    # 按出版年份对结果进行排序，年份越新越靠前
    papers.sort(key=lambda x: x['year'] if x['year'] is not None else -float('inf'), reverse=True)
    
    print(f"找到 {len(papers)} 篇关于 '{query}' 在 {site} 上的论文。")
    
    return papers

def sanitize_filename(filename):
    '''
    将文件名中的非字母数字字符替换为下划线
    '''
    return "".join([c if c.isalnum() else "_" for c in filename])

def download_paper(url, output_dir, paper_title, year):
    if url == 'No URL available':
        print("没有可用的下载链接。")
        return None
    
    print(f"正在下载论文：{url} (年份：{year})")

    # 获取页面内容
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # 查找PDF下载链接
    pdf_link = soup.find('a', class_='abs-button download-pdf')
    if not pdf_link:
        print("无法找到PDF下载链接。")
        return None

    pdf_url = urljoin(url, pdf_link.get('href'))

    response = requests.get(pdf_url, stream=True)
    sanitized_title = sanitize_filename(paper_title)
    filename = os.path.join(output_dir, sanitized_title + '.pdf')

    os.makedirs(output_dir, exist_ok=True)
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    with open(filename, 'wb') as f, tqdm(
        total=total_size, unit='iB', unit_scale=True
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            f.write(data)
    
    print(f"论文下载完成：{filename}")
    
    return filename

    
import shutil

def clear_output_directory(output_dir):
    """
    清空指定文件夹中的所有内容。
    """
    if os.path.exists(output_dir):
        # 删除文件夹中的所有文件和子文件夹
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # 删除文件或链接
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 删除文件夹
            except Exception as e:
                print(f"删除 {file_path} 时出错: {e}")

def score_abstract_with_zhipuai(abstract, user_input):
    # 使用智谱AI接口评分
    prompt = f"请评估以下摘要与用户提问'{user_input}'的相关性：\n\n{abstract}\n\n请给出一个相关性评分（0-1之间），你只需要给出一个0-1之间的数，不需要解释："
    score_str = get_completion(prompt).strip()
    
    try:
        score = float(score_str)
        if 0 <= score <= 1:
            return score
        else:
            print(f"得分{score}超出范围，应该在0到1之间。")
            return None
    except ValueError:
        print(f"无法解析评分：{score_str}")
        return None

def handle_request(key_statement, output_dir, site, total_papers, user_input):
    papers = search_papers(key_statement, site, total_papers)
    print(papers)

    total_score = 0
    valid_count = 0
    matching_papers = []

    print("正在计算内容匹配度...")
    for paper in papers:
        score = score_abstract_with_zhipuai(paper['abstract'], user_input)
        if score is not None:
            print(f"论文标题: {paper['title']}, 相关性评分: {score}")
            total_score += score
            valid_count += 1
            matching_papers.append((paper, score))

    if valid_count > 0:
        average_score = total_score / valid_count
    else:
        print("未能计算平均评分。")
        return []

    matching_papers_above_average = [paper for paper, score in matching_papers if score > average_score]

    print(f"找到 {len(matching_papers_above_average)} 篇与'{key_statement}'相关且智谱AI评分高于平均分的论文。")

    downloaded_files = []
    # 在下载新的文件之前清空文件夹内容
    clear_output_directory(output_dir)
    for paper in matching_papers_above_average:
        if paper['url'] != 'No URL available':
            filename = download_paper(paper['url'], output_dir, paper['title'], paper['year'])
            if filename:
                downloaded_files.append(filename)
            time.sleep(0.6)  # 每次下载后休眠0.5秒

    return downloaded_files


def main():
    output_dir = './papers'
    site = "arxiv.org"
    total_papers = 45

    print("欢迎使用智能论文下载助手！输入exit退出。")
    
    user_input = input("请输入您感兴趣的主题或问题：")

    if user_input.lower() == 'exit':
        print("感谢使用！")
        sys.exit(0)

    # 提炼用户输入，作为生成关键语句的提示
    prompt = f"Please\n\n<{user_input}>\n\n"
    prompt = f"""
            Your task is to read the question or topic enclosed in ``` below, and then provide one English keyword that best captures the core idea of the sentence from the perspective of paper keywords. No other sentences or numbers are needed:
            Example: I want to know the breakthroughs in channel coding for 6G technology.
            Answer: channel coding in 6G
            Example: I want to know some technologies and algorithms about digital watermarking.
            Answer: digital watermarking
            Example: I want to know the role of RNA in gene expression.
            Answer: RNA and gene expression
            Example: How to dynamically adjust the K value according to the current sample situation during the prediction process using the KNN algorithm?
            Answer: dynamic adjustment of K in KNN algorithm
            Example: What are the methods to improve the performance of large model RAG?
            Answer: improve llm RAG performance
            \n```<{user_input}>```\n
            """
    try:
        key_statement = get_completion(prompt)
        print(f"智能体生成的关键语句：\n{key_statement}")

        # 通过handle_request函数进行论文检索和下载
        downloaded_files = handle_request(key_statement, output_dir, site, total_papers, user_input)
        if downloaded_files:
            print("下载完成的论文文件：", downloaded_files)
    except Exception as e:
        print(f"出现错误：{str(e)}")

if __name__ == "__main__":
    main()
