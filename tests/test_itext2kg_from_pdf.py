import os
import logging
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader

from itext2kg.documents_distiller import DocumentsDistiller, Product
from itext2kg import iText2KG
from itext2kg.graph_integration import GraphIntegrator
from models import KnowledgeGraph

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_openai_llm_model(api_key, base_url, model="gpt-4o"):
    """
    初始化OpenAI语言模型。

    参数:
    - api_key: API KEY
    - base_url: OpenAI API的基准URL。
    - model: 要使用的语言模型，默认为"gpt-4o"。

    返回:
    - 初始化成功的ChatOpenAI对象，或者在发生错误时返回None。
    """
    try:
        # 创建ChatOpenAI实例，配置模型参数
        llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0,  # 温度设置为0，以获得确定性的结果
            max_tokens=None,  # 设置一个合理的最大token数
            timeout=None,  # 设置一个合理的请求超时时间
            max_retries=2,  # 设置最大重试次数，以处理临时的网络问题
        )
        logging.info(f"初始化OpenAI语言模型成功,model={model}")
        return llm
    except ChatOpenAI.Error as e:
        # 捕获ChatOpenAI的特定错误
        logging.error(f"OpenAI错误: {e}")
        return None
    except Exception as e:
        # 捕获其他异常
        logging.error(f"初始化OpenAI语言模型失败: {e}")
        return None


def create_embeddings_model(api_key, base_url, model="text-embedding-ada-002"):
    """
    创建并返回一个OpenAIEmbeddings模型。
    :param api_key: API KEY
    :param base_url: 模型服务的基URL。
    :param model: 要使用的嵌入模型的名称。
    :return: 初始化的OpenAIEmbeddings模型或None（在发生异常时）。
    """

    if not api_key:
        logging.error("警告：API键未设置。请确保环境变量已正确配置。")
        return None

    try:
        # 初始化OpenAIEmbeddings模型
        embeddings_model = OpenAIEmbeddings(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        logging.info(f"初始化嵌入模型成功,model={model}")
        return embeddings_model
    except Exception as e:
        # 异常处理，例如网络错误、无效的API键等
        logging.error(f"初始化OpenAIEmbeddings模型时发生错误: {e}")
        return None


def get_pdf_paths(directory):
    """
    获取指定目录下的所有PDF文件路径。

    参数:
    directory (str): 需要搜索的目录。

    返回:
    list: 包含所有PDF文件路径的列表。
    """
    # 初始化一个空列表，用于存储PDF文件路径
    pdf_paths = []

    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory):
        # 判断文件是否以'.pdf'结尾
        if filename.endswith('.pdf'):
            # 如果是PDF文件，将其完整路径加入到列表中
            pdf_paths.append(os.path.join(directory, filename))

    # 返回包含所有PDF文件路径的列表
    return pdf_paths


def extract_info_from_pdf(distiller, pdf_path):
    """
    根据schema抽取pdf文档中的信息。

    参数:
    - distiller: 信息提取器，用于从文档中提取关键信息。
    - pdf_path: PDF文档的路径。

    返回:
    - product_description_s: 抽取到的产品描述信息字符串。
    """
    # 初始化PDF加载器
    loader = PyPDFLoader(pdf_path)
    # 加载并拆分PDF文档为页面
    pages = loader.load_and_split()

    # 定义信息抽取查询指令
    ie_query = '''
    # DIRECTIVES : 
    - Act like an experienced information extractor. 
    - You have a chunk of a 菜品制作指南.
    - If you do not find the right information, keep its place empty.
    '''
    # 打印开始信息抽取任务的日志
    logging.info(f"[INFO] -------根据schema抽取文档信息开始,文档路径为：{pdf_path}-------")

    # 使用distiller从页面内容中提取信息，根据Product结构组织输出数据
    product_dict = distiller.distill(
        documents=[page.page_content.replace("{", '[').replace("}", "]") for page in pages], IE_query=ie_query,
        output_data_structure=Product)

    # 处理提取到的信息字典，格式化并过滤掉空值
    product_description = [f"{key} - {value}".replace("{", "[").replace("}", "]") for key, value in product_dict.items()
                           if value != [] and value != "" and value is not None]
    # 将处理后的信息列表连接成字符串
    product_description_s = ",".join(product_description)

    # 打印完成信息抽取任务的日志，包含抽取到的信息
    logging.info(f"[INFO] -------根据schema抽取文档信息结束，结果为：{product_description_s} -------")
    # 返回抽取到的信息字符串
    return product_description_s


def build_knowledge_graph(distiller: DocumentsDistiller, file_dir: str) -> KnowledgeGraph:
    """
    构建知识图谱。

    使用提供的文档处理工具和目录中的PDF文件来构建一个知识图谱。

    参数:
    - distiller: DocumentsDistiller类型，用于从文档中提取信息。
    - file_dir: str类型，指定PDF文件所在的目录路径。

    返回:
    - KnowledgeGraph类型，构建的知识图谱实例。
    """
    # 检查指定的目录路径是否存在，不存在则抛出异常
    if not os.path.exists(file_dir):
        raise FileNotFoundError(f"指定的路径不存在 - {file_dir}")

    # 获取目录下所有的PDF文件路径
    pdf_files = get_pdf_paths(file_dir)
    # 初始化知识图谱对象
    knowledge_graph1 = None

    # 遍历所有PDF文件，从每个文件中提取信息并构建知识图谱
    for pdf in pdf_files:
        # 从PDF文件中提取产品描述信息
        product_description_s = extract_info_from_pdf(distiller, pdf)
        # 构建知识图谱
        logging.info("[INFO] -------构建知识图谱开始---------")
        knowledge_graph1 = itext2kg.build_graph(sections=[product_description_s],
                                                existing_knowledge_graph=knowledge_graph1,
                                                ent_threshold=0.99, rel_threshold=0.99,
                                                entity_name_weight=0.8, entity_label_weight=0.2)
        # 构建知识图谱
        logging.info("[INFO] -------构建知识图谱结束---------")
    # 返回构建完成的知识图谱对象
    return knowledge_graph1


knowledgeGraph = None


def visualize_knowledge_graph(knowledge_graph: KnowledgeGraph):
    """
    将知识图谱可视化存储到Neo4j数据库中。

    参数:
    knowledge_graph (KnowledgeGraph): 要可视化的知识图谱对象。

    返回:
    无
    """
    # 知识图谱存储
    logging.info("[INFO] -------知识图谱可视化储存开始---------")

    # 定义Neo4j数据库的连接信息
    URI = "neo4j://localhost:7687"
    USERNAME = "neo4j"
    PASSWORD = "adminadmin"
    DATABASE = "cvdb"

    # 创建GraphIntegrator对象，用于与Neo4j数据库交互
    integrator = GraphIntegrator(uri=URI, username=USERNAME, password=PASSWORD, database=DATABASE)

    # 调用GraphIntegrator的visualize_graph方法，将知识图谱可视化存储到数据库
    integrator.visualize_graph(knowledge_graph)

    # 存储结束提示
    logging.info("[INFO] -------知识图谱可视化储存结束---------")


try:
    # 从环境变量中获取API密钥
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # 获取OpenAI基础URL环境变量
    openai_base_url = os.getenv("OPENAI_BASE_URL")

    # 初始化OpenAI语言模型，用于后续的自然语言处理任务
    openai_llm_model = initialize_openai_llm_model(api_key=openai_api_key, base_url=openai_base_url)

    # 创建嵌入模型，用于将文本转换为向量表示
    # 指定基础URL和模型类型，这里的模型是text-embedding-ada-002，是OpenAI提供的用于生成文本嵌入的模型
    openai_embeddings_model = create_embeddings_model(api_key=openai_api_key,base_url=openai_base_url)

    # 创建文档蒸馏器实例，用于提炼和处理文档内容
    # 将之前初始化的OpenAI语言模型传递给蒸馏器，以便于它能够利用该模型进行文档的处理
    document_distiller = DocumentsDistiller(llm_model=openai_llm_model)

    # 初始化iText2KG实例，用于从文本中提取知识图谱
    # 传递OpenAI语言模型和嵌入模型实例，以便于它能够利用这些模型进行文本到知识图谱的转换
    itext2kg = iText2KG(llm_model=openai_llm_model, embeddings_model=openai_embeddings_model)

    # 从配置文件或环境变量获取路径是一个好实践
    # dataset_path = config["dataset_path"] 或者从环境变量获取
    dataset_path = "../datasets/dish"

    knowledgeGraph = build_knowledge_graph(document_distiller, dataset_path)
    if knowledgeGraph is not None:
        visualize_knowledge_graph(knowledgeGraph)
except FileNotFoundError:
    logging.error("数据文件或目录未找到，请检查路径是否正确。")
    # 这里可以处理错误或退出
except ValueError as ve:
    logging.error(f"处理数据时发生值错误: {ve}")
    # 这里可以处理错误或退出
except Exception as e:
    # 即使是未预料到的错误类型，也应记录错误并适当处理
    logging.exception("发生未知错误，正在记录错误详情。")
    logging.error(f"处理时发生意外错误: {e}")
