import os
from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from itext2kg.documents_distiller import DocumentsDistiller, Product
from itext2kg import iText2KG
from itext2kg.graph_integration import GraphIntegrator

openai_api_key = "sk-bXIazOYTfGvtwIxu573e2b2907Cd4d48B9EfD6152990E086"

# 初始化模型
def initialize_openai_llm_model():
    try:
        llm = ChatOpenAI(
            api_key=openai_api_key,
            base_url="http://one-api.sit.yumc.local/v1",
            model="gpt-4o",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        return llm
    except Exception as e:
        print(f"初始化OpenAI语言模型失败: {e}")
        return None


def create_embeddings_model(base_url="http://one-api.sit.yumc.local/v1", model="text-embedding-ada-002"):
    """
    创建并返回一个OpenAIEmbeddings模型。
    :param base_url: 模型服务的基URL。
    :param model: 要使用的嵌入模型的名称。
    :return: 初始化的OpenAIEmbeddings模型或None（在发生异常时）。
    """

    if not openai_api_key:
        print("警告：API键未设置。请确保环境变量已正确配置。")
        return None

    try:
        # 初始化OpenAIEmbeddings模型
        openai_embeddings_model = OpenAIEmbeddings(
            api_key=openai_api_key,
            base_url=base_url,
            model=model,
        )
        return openai_embeddings_model
    except Exception as e:
        # 异常处理，例如网络错误、无效的API键等
        print(f"初始化OpenAIEmbeddings模型时发生错误: {e}")
        return None

IE_query = '''
# DIRECTIVES : 
- Act like an experienced information extractor. 
- You have a chunk of a 菜品制作指南.
- If you do not find the right information, keep its place empty.
'''

def get_pdf_paths(directory):
    pdf_paths = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            pdf_paths.append(os.path.join(directory, filename))
    return pdf_paths

# 文档信息提取
def build_sections(distiller: DocumentsDistiller, file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"指定的路径不存在 - {file_path}")
    sections = []
    pdf_files = get_pdf_paths(file_path)
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        pages = loader.load_and_split()
        distilled_product = distiller.distill(
            documents=[page.page_content.replace("{", '[').replace("}", "]") for page in pages], IE_query=IE_query,
            output_data_structure=Product)
        semantic_blocks_product = [f"{key} - {value}".replace("{", "[").replace("}", "]") for key, value in distilled_product.items()
                                   if value != [] and value != "" and value is not None]
        print("[INFO]抽取文档信息结果：".join(distilled_product))
        sections.append("".join(semantic_blocks_product))
    return sections


# 使用
openai_llm_model = initialize_openai_llm_model()
openai_embeddings_model= create_embeddings_model(base_url="http://one-api.sit.yumc.local/v1", model="text-embedding-ada-002")

document_distiller = DocumentsDistiller(llm_model=openai_llm_model)

semantic_blocks = []
itext2kg = iText2KG(llm_model=openai_llm_model, embeddings_model=openai_embeddings_model)
try:
    print("[INFO] -------使用给定schema抽取文档信息")
    semantic_blocks = build_sections(document_distiller,f"../datasets/dish")
except Exception as e:
    print(f"处理Product时发生错误: {e}")

URI = "neo4j://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "adminadmin"
DATABASE = "cvdb"

integrator = GraphIntegrator(uri=URI, username=USERNAME, password=PASSWORD, database=DATABASE)


# 构建知识图谱
print("[INFO] -------构建知识图谱")
knowledgeGraph1 = itext2kg.build_graph(semantic_blocks, ent_threshold=0.6, rel_threshold=0.6)
# 知识图谱存储
print("[INFO] -------知识图谱可视化储存")
integrator.visualize_graph(knowledgeGraph1)


print("[INFO]Finished")