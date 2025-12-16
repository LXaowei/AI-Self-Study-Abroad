#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
#get_ipython().system('ls /home/aistudio/data')


# In[58]:


#wuxg@2025.12.14：  local 向量化方式【已弃用】
# !pip uninstall -y paddlenlp paddlepaddle


# In[28]:


#wuxg@2025.12.14：  local 向量化方式【已弃用】
#!pip install paddlepaddle==2.5.2 -i https://mirror.baidu.com/pypi/simple
#!pip install paddlenlp==2.6.0 -i https://mirror.baidu.com/pypi/simple


# In[26]:


#wuxg@2025.12.14：  local 向量化方式【已弃用】
## 1. 查看Python版本和位数，确认是64位
# !python -c "import sys; print('Python版本:', sys.version); print('是否64位:', sys.maxsize > 2**32)"

## 2. 查看pip源上可用的paddlepaddle版本
# !pip index versions paddlepaddle

## 3. 查看pip源上可用的paddlenlp版本
# !pip index versions paddlenlp


# In[24]:


#wuxg@2025.12.14：  local 向量化方式【已弃用】
## 安装paddlepaddle，信任mirror.baidu.com
#!pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple --trusted-host mirror.baidu.com

## 安装paddlenlp，信任mirror.baidu.com
#!pip install paddlenlp -i https://mirror.baidu.com/pypi/simple --trusted-host mirror.baidu.com


# In[ ]:


#wuxg@2025.12.14：  local 向量化方式【已弃用】
#import paddle
#from paddlenlp.transformers import ErnieTokenizer, ErnieModel
#print(f"PaddlePaddle 版本: {paddle.__version__}")


# In[23]:


# 查看工作区文件，该目录下除data目录外的变更将会持久保存。请及时清理不必要的文件，避免加载过慢。
# View personal work directory. 
# All changes, except /data, under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
#get_ipython().system('ls /home/aistudio')


# In[20]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
#get_ipython().system('mkdir /home/aistudio/external-libraries')
#get_ipython().system('pip install beautifulsoup4')


# In[22]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')


# In[14]:


#get_ipython().system('pip install erniebot==0.5.3')
#get_ipython().system('pip install langchain==0.1.11')
#get_ipython().system('pip install langgraph==0.0.26')


# In[18]:


#wuxg@2025.12.14：  local 向量化方式【已弃用】
#!pip install openai pandas numpy faiss-cpu sentence-transformers paddle


# In[8]:


#get_ipython().system('pip install faiss-gpu')


# In[33]:


#pip list | grep faiss


# In[16]:


#wuxg@2025.12.14：  local 向量化方式【已弃用】
## 设备配置
#print("✅ GPU可用状态：", paddle.is_compiled_with_cuda())
#device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
#paddle.set_device(device)


# In[32]:


#pip install pypdf


# In[56]:


import warnings
warnings.filterwarnings("ignore")
import numpy as np
import faiss
from pypdf import PdfReader
import pandas as pd


#from docx import Document
from typing import List, Dict, Tuple, Optional, Any, Union, Callable, TypedDict  # 导入常用的类型提示


# In[36]:
import os


os.environ["WUXG_API_KEY"] = "678824fbafa46a532fdc555d378ab76d81c768aa"
api_key=os.environ.get("WUXG_API_KEY")


# In[34]:





# In[38]:



# ============================================
# 1. 文本工具类 (TextProcessor)
# ============================================

class TextProcessor:
    """
    文本处理工具类，用于从各种文件格式中提取和处理文本。统一的文本提取和处理接口。
	- 特点 ：
	  - 支持 PDF、Excel、Word、TXT 多种格式
	  - 智能文本分块（支持按段落和固定大小）
	  - 批量文件处理能力
	  - 错误异常处理
	  - 可配置的块大小
    """
    
    def __init__(self, chunk_size: int = 500):
        """
        初始化文本处理器
        
        参数:
        - chunk_size: 文本块大小，默认500字符
        """
        self.chunk_size = chunk_size
        self.supported_formats = ['.pdf', '.xlsx', '.xls', '.docx', '.doc', '.txt']
        
    def extract_from_pdf(self, pdf_path: str) -> List[str]:
        """
        从PDF文件中提取文本并分块
        
        参数:
        - pdf_path: PDF文件路径
        
        返回:
        - 文本块列表
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
            
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            
            if not text.strip():
                print(f"⚠️ PDF文件 {pdf_path} 未提取到文本")
                return []
                
            # 分块处理
            chunks = self._chunk_text(text)
            print(f"✅ 从PDF提取 {len(chunks)} 个文本块: {pdf_path}")
            return chunks
            
        except Exception as e:
            print(f"❌ PDF提取失败 {pdf_path}: {e}")
            return []
    
    def extract_from_excel(self, excel_path: str) -> List[str]:
        """
        从Excel文件中提取文本并分块
        
        参数:
        - excel_path: Excel文件路径
        
        返回:
        - 文本块列表
        """
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Excel文件不存在: {excel_path}")
            
        try:
            excel_file = pd.ExcelFile(excel_path)
            all_chunks = []
            
            for sheet in excel_file.sheet_names:
                try:
                    df = pd.read_excel(excel_file, sheet)
                    sheet_text = df.to_string()
                    
                    if sheet_text.strip():
                        sheet_chunks = self._chunk_text(sheet_text)
                        all_chunks.extend(sheet_chunks)
                        print(f"  📊 工作表 '{sheet}': {len(sheet_chunks)} 个块")
                    else:
                        print(f"  ⚠️ 工作表 '{sheet}' 为空")
                        
                except Exception as e:
                    print(f"  ⚠️ 读取工作表 '{sheet}' 失败: {e}")
                    continue
            
            print(f"✅ 从Excel提取 {len(all_chunks)} 个文本块: {excel_path}")
            return all_chunks
            
        except Exception as e:
            print(f"❌ Excel提取失败 {excel_path}: {e}")
            return []
    
    def extract_from_word(self, word_path: str) -> List[str]:
        """
        从Word文件中提取文本并分块
        
        参数:
        - word_path: Word文件路径
        
        返回:
        - 文本块列表
        """
        if not os.path.exists(word_path):
            raise FileNotFoundError(f"Word文件不存在: {word_path}")
            
        try:
            doc = Document(word_path)
            text = ""
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text + "\n"
            
            if not text.strip():
                print(f"⚠️ Word文件 {word_path} 未提取到文本")
                return []
                
            # 分块处理
            chunks = self._chunk_text(text)
            print(f"✅ 从Word提取 {len(chunks)} 个文本块: {word_path}")
            return chunks
            
        except Exception as e:
            print(f"❌ Word提取失败 {word_path}: {e}")
            return []
    
    def extract_from_text(self, text: str) -> List[str]:
        """
        从纯文本中提取并分块
        
        参数:
        - text: 原始文本
        
        返回:
        - 文本块列表
        """
        if not text.strip():
            return []
            
        chunks = self._chunk_text(text)
        print(f"✅ 从文本提取 {len(chunks)} 个文本块")
        return chunks
    
    def extract_from_file(self, file_path: str) -> List[str]:
        """
        根据文件扩展名自动选择合适的提取方法
        
        参数:
        - file_path: 文件路径
        
        返回:
        - 文本块列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self.extract_from_pdf(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            return self.extract_from_excel(file_path)
        elif file_ext in ['.docx', '.doc']:
            return self.extract_from_word(file_path)
        elif file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.extract_from_text(content)
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}，支持格式: {self.supported_formats}")
    
    def extract_from_multiple_files(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """
        从多个文件中批量提取文本
        
        参数:
        - file_paths: 文件路径列表
        
        返回:
        - 字典，键为文件路径，值为文本块列表
        """
        results = {}
        for file_path in file_paths:
            try:
                chunks = self.extract_from_file(file_path)
                results[file_path] = chunks
            except Exception as e:
                print(f"❌ 文件处理失败 {file_path}: {e}")
                results[file_path] = []
        
        return results
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        内部方法：将文本分块
        
        参数:
        - text: 原始文本
        
        返回:
        - 文本块列表
        """
        if not text.strip():
            return []
        
        # 先按段落分割
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # 如果段落本身就很大，直接分割
            if len(para) >= self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # 分割大段落
                for i in range(0, len(para), self.chunk_size):
                    chunk = para[i:i + self.chunk_size]
                    if chunk.strip():
                        chunks.append(chunk.strip())
            else:
                # 如果当前块加上新段落不超过大小，就合并
                if len(current_chunk) + len(para) + 1 <= self.chunk_size:
                    if current_chunk:
                        current_chunk += "\n" + para
                    else:
                        current_chunk = para
                else:
                    # 否则保存当前块，开始新块
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para
        
        # 处理最后一个块
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks


# In[40]:


#wuxg@2025.12.14: 已弃用 local向量化方式
#from paddlenlp.transformers import ErnieTokenizer, ErnieModel


# In[50]:



# ============================================
# 2. 向量化类 (Vectorizer)
# ============================================
#wuxg@2025.12.14 ： 本向量化方式 【失败。将改为aistudio的在线向量化方式！使用下面的ERNIEVectorizer2】【已弃用】
class ERNIEVectorizer1:
    """
    ERNIE模型向量化类，用于将文本转换为向量表示。使用ERNIE模型进行文本向量化
	-  特点 ：
	  - 支持批量向量化处理
	  - 进度显示和错误处理
	  - 可配置的批处理大小和文本长度
	  - 提供模型信息查询
	  - 支持单个文本向量化
    """
    
    def __init__(self, model_name: str = "ernie-3.0-medium-zh", 
                 batch_size: int = 16, max_length: int = 512):
        """
        初始化ERNIE向量化器
        
        参数:
        - model_name: ERNIE模型名称
        - batch_size: 批处理大小
        - max_length: 最大文本长度
        """
        print(f"🔧 初始化ERNIE向量化器: {model_name}")
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.embedding_dim = 768  # ERNIE-3.0-medium-zh的向量维度
        
        # 加载tokenizer和模型
        try:
            self.tokenizer = ErnieTokenizer.from_pretrained(model_name)
            self.model = ErnieModel.from_pretrained(model_name)
            self.model.eval()
            print(f"✅ ERNIE模型加载成功，向量维度: {self.embedding_dim}")
        except Exception as e:
            print(f"❌ ERNIE模型加载失败: {e}")
            raise
    
    def vectorize(self, text_chunks: List[str]) -> np.ndarray:
        """
        将文本块列表向量化
        
        参数:
        - text_chunks: 文本块列表
        
        返回:
        - 向量矩阵，形状为 (n_samples, embedding_dim)
        """
        if not text_chunks:
            print("⚠️ 文本块列表为空")
            return np.array([])
        
        print(f"🔧 开始向量化 {len(text_chunks)} 个文本块...")
        
        all_vectors = []
        processed_count = 0
        
        with paddle.no_grad():
            # 分批处理
            for i in range(0, len(text_chunks), self.batch_size):
                batch_texts = text_chunks[i:i + self.batch_size]
                batch_vectors = []
                
                for text in batch_texts:
                    if not text or not text.strip():
                        continue
                    
                    try:
                        # 对每个文本进行编码
                        inputs = self.tokenizer(
                            text,
                            truncation=True,
                            max_length=self.max_length,
                            padding="max_length",
                            return_tensors="pd"
                        )
                        
                        # 获取模型输出
                        outputs = self.model(**inputs)
                        
                        # 使用[CLS] token的向量作为文本表示
                        cls_vector = outputs[0][:, 0, :].numpy()
                        batch_vectors.append(cls_vector[0])
                        
                        processed_count += 1
                        
                    except Exception as e:
                        print(f"⚠️ 文本向量化失败（已跳过）: {text[:50]}... - {e}")
                        continue
                
                if batch_vectors:
                    all_vectors.extend(batch_vectors)
                
                # 显示进度
                if i + self.batch_size < len(text_chunks):
                    progress = min(100, int((i + len(batch_texts)) / len(text_chunks) * 100))
                    print(f"  进度: {progress}% ({i + len(batch_texts)}/{len(text_chunks)})")
        
        if all_vectors:
            vectors_array = np.array(all_vectors, dtype=np.float32)
            print(f"✅ 向量化完成: {vectors_array.shape[0]} 个向量，维度 {vectors_array.shape[1]}")
            return vectors_array
        else:
            print("❌ 向量化失败：未生成任何向量")
            return np.array([])
    
    def vectorize_single(self, text: str) -> np.ndarray:
        """
        向量化单个文本
        
        参数:
        - text: 单个文本
        
        返回:
        - 向量，形状为 (embedding_dim,)
        """
        if not text or not text.strip():
            raise ValueError("文本不能为空")
        
        vectors = self.vectorize([text])
        if len(vectors) > 0:
            return vectors[0]
        else:
            raise ValueError("向量化失败")
    
    def get_embedding_dim(self) -> int:
        """
        获取向量维度
        
        返回:
        - 向量维度
        """
        return self.embedding_dim
    
    def get_model_info(self) -> Dict[str, str]:
        """
        获取模型信息
        
        返回:
        - 模型信息字典
        """
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "max_length": self.max_length,
            "batch_size": self.batch_size
        }
        


# In[42]:


from openai import OpenAI
from typing import List, Dict, Any


# In[44]:


client = OpenAI(
    api_key=os.environ.get("WUXG_API_KEY"),
    base_url="https://aistudio.baidu.com/llm/lmapi/v3"
)


# In[43]:


#wuxg@2025.12.14：另一种、在线向量化的实现方式
class ERNIEVectorizer2_bak:
    def __init__(self, client):
        self.client = client
        self.model = "embedding-v1"
    
    def get_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=[text]
        )
        print(response)
        print("------------------------")
        print(response.data)
        print("------------------------")
        print(response.data[0])
        print("------------------------")
        print(response.data[0].embedding)
        return response.data[0].embedding
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]


# In[46]:


#wuxg@2025.12.14：另一种、在线向量化的实现方式
class ERNIEVectorizer2:
    def __init__(self, client, max_batch_size: int = 16):  # 新增批次大小参数
        self.client = client
        self.model = "embedding-v1"
        self.max_batch_size = max_batch_size  # API允许的最大批次大小

    def get_embedding(self, text: str) -> List[float]:
        """获取单个文本的向量"""
        # 直接调用批量接口，但只传一个文本
        return self.get_embeddings_batch([text])[0]

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本向量，自动处理API的批次限制"""
        all_embeddings = []
        total_texts = len(texts)
        
        print(f"  准备向量化 {total_texts} 个文本，将分批处理（每批最多 {self.max_batch_size} 个）...")
        
        # 将文本列表按max_batch_size分成小批次
        for i in range(0, total_texts, self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]
            batch_num = i // self.max_batch_size + 1
            total_batches = (total_texts + self.max_batch_size - 1) // self.max_batch_size
            
            print(f"    正在处理第 {batch_num}/{total_batches} 批 ({len(batch)} 个文本)...")
            
            try:
                # 调用API
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                
                # 提取本批次的向量
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"    ❌ 第 {batch_num} 批处理失败: {e}")
                # 可以选择抛出异常，或者用零向量填充失败批次
                # 这里选择抛出异常，确保问题能被发现
                raise RuntimeError(f"第 {batch_num} 批向量化失败: {e}") from e
        
        print(f"  所有批次处理完成，共生成 {len(all_embeddings)} 个向量。")
        return all_embeddings
        


# In[48]:



# ============================================
# 3. 基于FAISS向量数据库的检索器（保持原有版本）
# ============================================
#wuxg@2025.12.14：本向量失败，【已弃用】因此改为在线向量化方案后。使用下文的FAISSVectorDB2
class FAISSVectorDB:
    """
    基于FAISS的向量数据库类，用于build构建、save存储、load加载、和retrieve检索文本向量(参数：适配 TextProcessor和ERNIEVectorizer2类)
	-  特点 ：
	  - 松耦合设计，依赖注入
	  - 支持多种文件格式混合构建
	  - 向量索引的持久化存储
	  - 详细的统计信息
	  - 灵活的检索功能

    """
    def __init__(self, embedding_dim: int = 768):
        """
        初始化向量数据库
        
        参数:
        - embedding_dim: 向量维度，ERNIE-3.0-medium-zh为768
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.chunks = []  # 存储原始文本块
        self.metadata = []  # 存储元数据（如文档来源、位置等）
        self.is_trained = False
        
    def build_from_processor(self, text_processor: TextProcessor, vectorizer: ERNIEVectorizer2,
                           pdf_path: Optional[str] = None, 
                           excel_path: Optional[str] = None, 
                           word_path: Optional[str] = None) -> bool:
        """
        使用文本处理器和向量化器构建向量索引
        
        参数:
        - text_processor: 文本处理器实例
        - vectorizer: 向量化器实例
        - pdf_path: PDF文件路径
        - excel_path: Excel文件路径
        - word_path: Word文件路径
        
        返回:
        - 成功返回True，失败返回False
        """
        all_chunks = []
        file_sources = []  # 记录每个文本块的来源
        
        # 从PDF提取
        if pdf_path and os.path.exists(pdf_path):
            print(f"📄 从PDF文件提取文本: {pdf_path}")
            pdf_chunks = text_processor.extract_from_pdf(pdf_path)
            all_chunks.extend(pdf_chunks)
            file_sources.extend(["pdf"] * len(pdf_chunks))
        
        # 从Excel提取
        if excel_path and os.path.exists(excel_path):
            print(f"📊 从Excel文件提取文本: {excel_path}")
            excel_chunks = text_processor.extract_from_excel(excel_path)
            all_chunks.extend(excel_chunks)
            file_sources.extend(["excel"] * len(excel_chunks))
        
        # 从Word提取
        if word_path and os.path.exists(word_path):
            print(f"📝 从Word文件提取文本: {word_path}")
            word_chunks = text_processor.extract_from_word(word_path)
            all_chunks.extend(word_chunks)
            file_sources.extend(["word"] * len(word_chunks))
        
        if not all_chunks:
            print("❌ 未提取到任何文本，请检查文件路径！")
            return False
        
        print(f"✅ 共提取到 {len(all_chunks)} 个文本块")
        
        # 向量化文本块
        print("🔧 正在进行文本向量化...")
        all_vectors = vectorizer.vectorize(all_chunks)
        
        if len(all_vectors) == 0:
            print("❌ 向量化失败，无有效向量生成")
            return False
        
        # 创建FAISS索引（使用内积，便于计算余弦相似度）
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # 归一化向量（用于余弦相似度计算）
        faiss.normalize_L2(all_vectors)
        
        # 添加到索引
        self.index.add(all_vectors)
        self.chunks = all_chunks
        
        # 创建元数据
        self.metadata = []
        for i, (chunk, source) in enumerate(zip(all_chunks, file_sources)):
            self.metadata.append({
                "id": i,
                "source": source,
                "chunk_size": len(chunk),
                "preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
            })
        
        print(f"✅ 向量索引构建完成，包含 {len(all_chunks)} 个向量")
        print(f"✅ 向量维度: {self.embedding_dim}")
        print(f"✅ 索引类型: {type(self.index).__name__}")
        
        return True
    
    def retrieve(self, query: str, vectorizer: ERNIEVectorizer2, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        检索与查询最相似的文本块
        
        参数:
        - query: 查询文本
        - vectorizer: 向量化器实例
        - top_k: 返回最相似的k个结果
        
        返回:
        - 列表，每个元素为 (文本, 相似度分数, 元数据)
        """
        if not self.index:
            print("❌ 索引未构建，请先调用 build_from_processor 方法")
            return []
        
        if not query or not query.strip():
            print("❌ 查询文本为空")
            return []
        
        # 向量化查询文本
        print(f"🔍 处理查询: '{query[:50]}...'" if len(query) > 50 else f"🔍 处理查询: '{query}'")
        query_vector = vectorizer.vectorize([query.strip()])
        
        if len(query_vector) == 0:
            print("❌ 查询向量化失败")
            return []
        
        # 归一化查询向量
        faiss.normalize_L2(query_vector)
        
        # 检索相似文本
        similarities, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.chunks) and idx >= 0:
                chunk = self.chunks[idx]
                metadata = self.metadata[idx] if idx < len(self.metadata) else {}
                results.append((chunk, float(similarity), metadata))
        
        # 按相似度降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"✅ 检索完成，找到 {len(results)} 个相关结果")
        return results
    
    def save_index(self, filepath: str):
        """
        保存向量索引到文件
        
        参数:
        - filepath: 保存路径（不带扩展名）
        """
        if not self.index:
            print("❌ 索引未构建，无法保存")
            return
        
        try:
            # 保存FAISS索引
            faiss.write_index(self.index, f"{filepath}.index")
            
            # 保存文本和元数据
            data_to_save = {
                'chunks': self.chunks,
                'metadata': self.metadata,
                'embedding_dim': self.embedding_dim,
                'is_trained': self.is_trained
            }
            
            with open(f"{filepath}.data", 'wb') as f:
                pickle.dump(data_to_save, f)
            
            print(f"✅ 索引已保存到: {filepath}.index 和 {filepath}.data")
            
        except Exception as e:
            print(f"❌ 保存索引失败: {e}")
    
    def load_index(self, filepath: str):
        """
        从文件加载向量索引
        
        参数:
        - filepath: 文件路径（不带扩展名）
        """
        try:
            # 加载FAISS索引
            if os.path.exists(f"{filepath}.index"):
                self.index = faiss.read_index(f"{filepath}.index")
            else:
                print(f"❌ 索引文件不存在: {filepath}.index")
                return False
            
            # 加载文本和元数据
            if os.path.exists(f"{filepath}.data"):
                with open(f"{filepath}.data", 'rb') as f:
                    data_loaded = pickle.load(f)
                
                self.chunks = data_loaded['chunks']
                self.metadata = data_loaded['metadata']
                self.embedding_dim = data_loaded['embedding_dim']
                self.is_trained = data_loaded['is_trained']
            else:
                print(f"❌ 数据文件不存在: {filepath}.data")
                return False
            
            print(f"✅ 索引加载成功，包含 {len(self.chunks)} 个文本块")
            return True
            
        except Exception as e:
            print(f"❌ 加载索引失败: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """
        获取向量数据库统计信息
        
        返回:
        - 统计信息字典
        """
        if not self.index:
            return {"status": "索引未构建"}
        
        stats = {
            "status": "已构建",
            "total_chunks": len(self.chunks),
            "embedding_dim": self.embedding_dim,
            "index_type": type(self.index).__name__,
            "sources": {}
        }
        
        # 统计各来源的文本块数量
        if self.metadata:
            for meta in self.metadata:
                source = meta.get("source", "unknown")
                stats["sources"][source] = stats["sources"].get(source, 0) + 1
        
        return stats


# In[50]:


#wuxg@2025.12.14：使用aistudio的在线token方式
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional, Any
import pickle
import os

class FAISSVectorDB2:
    """
    基于FAISS的向量数据库类，适配新的 ERNIEVectorizer2 API调用方式。
    """
    
    def __init__(self, index_type: str = "flat"):
        """
        初始化向量数据库（不再需要预设embedding_dim）
        
        参数:
        - index_type: 索引类型，支持: "flat" (精确搜索), "ivf" (适合大规模)
        """
        self.index_type = index_type
        self.index = None          # FAISS索引对象
        self.embedding_dim = None  # 向量维度（运行时确定）
        self.chunks = []           # 存储原始文本块
        self.metadata = []         # 存储元数据
        self.is_trained = False
        self.vectorizer=None
    
    def build_index(self, 
                   text_chunks: List[str], 
                   vectorizer: Any,  # 可传入ERNIEVectorizer2实例
                   metadata: Optional[List[Dict]] = None,
                   normalize: bool = True) -> bool:
        """
        构建向量索引（核心方法）
        
        参数:
        - text_chunks: 文本块列表
        - vectorizer: 向量化器实例（需有get_embeddings_batch方法）
        - metadata: 可选的元数据列表
        - normalize: 是否对向量进行L2归一化（建议保持True，以便使用余弦相似度）
        
        返回:
        - 成功返回True，失败返回False
        """
        if not text_chunks:
            print("❌ 文本块列表为空，无法构建索引")
            return False
        
        print(f"🔧 开始构建索引，共 {len(text_chunks)} 个文本块...")
        
        # 1. 批量向量化
        print("  正在进行文本向量化...")
        try:
            # 调用新的向量化接口
            embeddings_list = vectorizer.get_embeddings_batch(text_chunks)
        except Exception as e:
            print(f"❌ 向量化过程出错: {e}")
            return False
        
        # 2. 转换为numpy数组并获取维度
        try:
            embeddings_array = np.array(embeddings_list, dtype=np.float32)
            self.embedding_dim = embeddings_array.shape[1]
            print(f"✅ 向量化完成，维度: {self.embedding_dim}")
        except Exception as e:
            print(f"❌ 向量数据转换失败: {e}")
            return False
        
        # 3. 创建FAISS索引
        print(f"  创建 {self.index_type} 类型索引...")
        try:
            if self.index_type == "flat":
                # 精确搜索，使用内积度量（归一化后即为余弦相似度）
                self.index = faiss.IndexFlatIP(self.embedding_dim)
            elif self.index_type == "ivf":
                # IVF索引，适合大规模数据
                nlist = min(100, int(np.sqrt(len(text_chunks))))  # 聚类中心数
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
                self.is_trained = False
            else:
                print(f"❌ 不支持的索引类型: {self.index_type}")
                return False
        except Exception as e:
            print(f"❌ 创建索引失败: {e}")
            return False
        
        # 4. 归一化向量（用于余弦相似度）
        if normalize:
            print("  归一化向量...")
            faiss.normalize_L2(embeddings_array)
        
        # 5. 训练索引（仅IVF需要）
        if self.index_type == "ivf" and len(text_chunks) >= 100:
            print("  训练IVF索引...")
            try:
                self.index.train(embeddings_array)
                self.is_trained = True
            except Exception as e:
                print(f"⚠️  索引训练失败: {e}")
                # 部分情况下可继续
        
        # 6. 添加向量到索引
        print("  添加向量到索引...")
        try:
            self.index.add(embeddings_array)
        except Exception as e:
            print(f"❌ 添加向量失败: {e}")
            return False
        
        # 7. 保存文本和元数据
        self.chunks = text_chunks
        self.metadata = metadata if metadata else []
        
        # 如果元数据不足，生成默认元数据
        if len(self.metadata) < len(self.chunks):
            self.metadata = self.metadata + [
                {"id": i, "chunk_size": len(chunk), "preview": chunk[:100]}
                for i in range(len(self.metadata), len(self.chunks))
            ]
        
        print(f"✅ 索引构建完成！包含 {self.index.ntotal} 个向量")
        return True
    
    def build_from_files(self, 
                        text_processor: Any,
                        vectorizer: Any,
                        pdf_path: Optional[str] = None,
                        excel_path: Optional[str] = None,
                        word_path: Optional[str] = None) -> bool:
        """
        从文件构建索引的便捷方法（向后兼容）
        
        注意：此方法依赖TextProcessor，如果你的项目中没有，可删除或修改
        """
        all_chunks = []
        file_sources = []
        self.vectorizer= vectorizer
        
        # 从不同文件提取文本
        if pdf_path and os.path.exists(pdf_path):
            print(f"📄 从PDF提取: {pdf_path}")
            chunks = text_processor.extract_from_pdf(pdf_path)
            all_chunks.extend(chunks)
            file_sources.extend(["pdf"] * len(chunks))
        
        if excel_path and os.path.exists(excel_path):
            print(f"📊 从Excel提取: {excel_path}")
            chunks = text_processor.extract_from_excel(excel_path)
            all_chunks.extend(chunks)
            file_sources.extend(["excel"] * len(chunks))
        
        if word_path and os.path.exists(word_path):
            print(f"📝 从Word提取: {word_path}")
            chunks = text_processor.extract_from_word(word_path)
            all_chunks.extend(chunks)
            file_sources.extend(["word"] * len(chunks))
        
        if not all_chunks:
            print("❌ 未提取到任何文本")
            return False
        
        print(f"✅ 共提取到 {len(all_chunks)} 个文本块")
        
        # 创建元数据
        metadata = [
            {
                "id": i,
                "source": source,
                "chunk_size": len(chunk),
                "preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
            }
            for i, (chunk, source) in enumerate(zip(all_chunks, file_sources))
        ]
        
        # 调用核心构建方法
        return self.build_index(all_chunks, vectorizer, metadata)
    
    def retrieve(self, 
                query: str, 
                vectorizer: Any, 
                top_k: int = 5,
                score_threshold: float = 0.0) -> List[Tuple[str, float, Dict]]:
        """
        检索与查询最相似的文本块
        
        参数:
        - query: 查询文本
        - vectorizer: 向量化器实例
        - top_k: 返回最相似的k个结果
        - score_threshold: 相似度分数阈值，低于此值的结果将被过滤
        
        返回:
        - 列表，每个元素为 (文本, 相似度分数, 元数据)
        """
        if self.index is None:
            print("❌ 索引未构建，请先调用 build_index 方法")
            return []
        
        if not query or not query.strip():
            print("❌ 查询文本为空")
            return []
        
        print(f"🔍 查询: '{query[:50]}...'" if len(query) > 50 else f"🔍 查询: '{query}'")
        
        # 1. 向量化查询文本
        try:
            # 使用新的向量化接口
            query_embedding_list = vectorizer.get_embeddings_batch([query.strip()])
            query_vector = np.array(query_embedding_list[0], dtype=np.float32).reshape(1, -1)
        except Exception as e:
            print(f"❌ 查询向量化失败: {e}")
            return []
        
        # 2. 归一化查询向量（必须与索引构建时的处理一致）
        faiss.normalize_L2(query_vector)
        
        # 3. 执行搜索
        try:
            # 注意：IndexFlatIP返回的是内积分数，归一化后即为余弦相似度
            scores, indices = self.index.search(query_vector, top_k)
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return []
        
        # 4. 整理结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            # 检查索引有效性
            if idx == -1 or idx >= len(self.chunks):
                continue
            
            # 应用分数阈值
            if score < score_threshold:
                continue
            
            chunk = self.chunks[idx]
            metadata = self.metadata[idx] if idx < len(self.metadata) else {}
            results.append((chunk, float(score), metadata))
        
        print(f"✅ 检索完成，返回 {len(results)} 个结果")
        return results
    
    def save_index(self, filepath: str):
        """保存索引到文件"""
        if self.index is None:
            print("❌ 索引未构建，无法保存")
            return
        
        try:
            # 保存FAISS索引
            faiss.write_index(self.index, f"{filepath}.index")
            
            # 保存数据
            data_to_save = {
                'chunks': self.chunks,
                'metadata': self.metadata,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type,
                'is_trained': self.is_trained
            }
            
            with open(f"{filepath}.data", 'wb') as f:
                pickle.dump(data_to_save, f)
            
            print(f"✅ 索引已保存: {filepath}.index, {filepath}.data")
            
        except Exception as e:
            print(f"❌ 保存失败: {e}")
    
    def load_index(self, filepath: str) -> bool:
        """从文件加载索引"""
        try:
            # 加载FAISS索引
            index_path = f"{filepath}.index"
            if not os.path.exists(index_path):
                print(f"❌ 索引文件不存在: {index_path}")
                return False
            
            self.index = faiss.read_index(index_path)
            
            # 加载数据
            data_path = f"{filepath}.data"
            if not os.path.exists(data_path):
                print(f"❌ 数据文件不存在: {data_path}")
                return False
            
            with open(data_path, 'rb') as f:
                data_loaded = pickle.load(f)
            
            self.chunks = data_loaded['chunks']
            self.metadata = data_loaded['metadata']
            self.embedding_dim = data_loaded['embedding_dim']
            self.index_type = data_loaded['index_type']
            self.is_trained = data_loaded['is_trained']
            
            print(f"✅ 索引加载成功，包含 {len(self.chunks)} 个文本块")
            return True
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        if self.index is None:
            return {"status": "索引未构建"}
        
        stats = {
            "status": "已构建",
            "total_chunks": len(self.chunks),
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "is_trained": self.is_trained,
            "index_size": self.index.ntotal,
            "sources": {}
        }
        
        # 统计来源分布
        if self.metadata:
            for meta in self.metadata:
                source = meta.get("source", "unknown")
                stats["sources"][source] = stats["sources"].get(source, 0) + 1
        
        return stats
    
    def similarity_search(self, 
                         query_vector: np.ndarray,
                         top_k: int = 5) -> List[Tuple[int, float]]:
        """
        直接使用向量进行搜索（高级用法）
        
        参数:
        - query_vector: 已向量化的查询，形状为 (1, embedding_dim)
        - top_k: 返回数量
        
        返回:
        - 列表，每个元素为 (索引, 分数)
        """
        if self.index is None:
            raise ValueError("索引未构建")
        
        # 确保查询向量已归一化
        faiss.normalize_L2(query_vector)
        
        scores, indices = self.index.search(query_vector, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append((int(idx), float(score)))
        
        return results


# In[48]:



# ============================================
# 阶段性小结：TextProcess+Embedding+Faiss 使用示例
# ============================================

def usage_example():  #已弃用（改为在向量化方式）
    """使用示例"""
    
    # 1. 创建文本处理器
    print("1. 初始化文本处理器...")
    text_processor = TextProcessor(chunk_size=500)
    
    # 2. 创建ERNIE向量化器
    print("2. 初始化ERNIE向量化器...")
    vectorizer = ERNIEVectorizer1(model_name="ernie-3.0-medium-zh", batch_size=8)
    
    # 3. 创建向量数据库
    print("3. 初始化FAISS向量数据库...")
    vector_db = FAISSVectorDB(embedding_dim=vectorizer.get_embedding_dim())
    
    # 4. 构建索引
    print("4. 构建向量索引...")
    success = vector_db.build_from_processor(
        text_processor=text_processor,
        vectorizer=vectorizer,
        pdf_path="保.pdf",      # 替换为实际文件路径
        excel_path="冲稳.xlsx"  # 替换为实际文件路径
    )
    
    if success:
        # 5. 显示统计信息
        print("\n5. 向量数据库统计信息:")
        stats = vector_db.get_stats()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        
        # 6. 测试检索
        print("\n6. 测试检索功能...")
        test_queries = [
            "美国留学的申请截止日期是什么时候？",
            "需要准备哪些申请材料？",
            "留学费用大概是多少？"
        ]
        
        for query in test_queries:
            print(f"\n🔍 查询: {query}")
            results = vector_db.retrieve(query, vectorizer, top_k=3)
            
            if results:
                print(f"  找到 {len(results)} 个相关结果:")
                for i, (chunk, score, metadata) in enumerate(results):
                    print(f"  {i+1}. [相似度: {score:.4f}] {chunk[:80]}...")
                    print(f"     来源: {metadata.get('source', 'unknown')}, 大小: {metadata.get('chunk_size', 0)}字符")
            else:
                print("  未找到相关结果")
        
        # 7. 保存索引
        print("\n7. 保存向量索引...")
        vector_db.save_index("faiss_index_example")
        
    else:
        print("❌ 向量索引构建失败")

if __name__ == "__main__":
    usage_example()


# In[52]:


def usage_example_with_api():
    #使用在线API的ERNIEVectorizer2和FAISSVectorDB的示例"""
    
    print("🚀 开始基于在线API的文档检索系统演示")
    print("=" * 50)
    
    # 1. 创建文本处理器
    print("1. 初始化文本处理器...")
    text_processor = TextProcessor(chunk_size=500)
    
    # 2. 创建ERNIEVectorizer2 (在线API)
    print("2. 初始化ERNIEVectorizer2 (在线API)...")
    # 注意: 你需要在此处正确初始化你的API客户端
    # 假设你的ERNIEVectorizer2接收一个已配置的client对象
    from openai import OpenAI # 示例：使用OpenAI格式的客户端
    # 请替换为你的实际API配置
    client = OpenAI(
        api_key=os.environ.get("WUXG_API_KEY"),
        base_url="https://aistudio.baidu.com/llm/lmapi/v3"
    )
    vectorizer = ERNIEVectorizer2(client=client)
    print("   ✅ 在线向量化器准备就绪")
    
    # 3. 创建向量数据库 (适配ERNIEVectorizer2的版本)
    print("3. 初始化FAISS向量数据库...")
    # 注意: 这里使用的是你已重写的、无需预设维度的FAISSVectorDB类
    vector_db = FAISSVectorDB2(index_type="flat")
    
    # 4. 构建索引
    print("4. 构建向量索引...")
    success = vector_db.build_from_files(
        text_processor=text_processor,
        vectorizer=vectorizer,
        pdf_path="保.pdf",      # 替换为实际文件路径
        excel_path="冲稳.xlsx"  # 替换为实际文件路径
        # 可选: word_path="your_doc.docx"
    )
    
    if success:
        # 5. 显示统计信息
        print("\n5. 向量数据库统计信息:")
        stats = vector_db.get_stats()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        
        # 6. 测试检索
        print("\n6. 测试检索功能...")
        test_queries = [
            "美国留学的申请截止日期是什么时候？",
            "需要准备哪些申请材料？",
            "留学费用大概是多少？"
        ]
        
        for query in test_queries:
            print(f"\n🔍 查询: {query}")
            # 注意: retrieve方法现在接收ERNIEVectorizer2作为参数
            results = vector_db.retrieve(query, vectorizer, top_k=3)
            
            if results:
                print(f"  找到 {len(results)} 个相关结果:")
                for i, (chunk, score, metadata) in enumerate(results):
                    # 格式化输出，控制预览长度
                    preview = chunk if len(chunk) <= 80 else chunk[:77] + "..."
                    print(f"  {i+1}. [相似度: {score:.4f}] {preview}")
                    print(f"     来源: {metadata.get('source', 'unknown')}, 大小: {len(chunk)}字符")
            else:
                print("  未找到相关结果")
        
        # 7. 保存索引 (可选)
        print("\n7. 保存向量索引到文件...")
        vector_db.save_index("faiss_index_online_example")
        print("   ✅ 索引已保存，可用于后续快速加载")
        
    else:
        print("❌ 向量索引构建失败")
    
    print("\n" + "=" * 50)
    print("演示结束")

if __name__ == "__main__":
    # 执行基于在线API的示例
    usage_example_with_api()


# In[58]:


# ============================================
#  4.  Self-RAG状态与工作流
# ============================================
class GraphState(TypedDict):
    keys: Dict[str, str]

class SelfRAGGraph:
    def __init__(self, vector_db: FAISSVectorDB2): #【已弃用FAISSVectorDB】使用FAISSVectorDB2
        self.vector_db = vector_db
        self.nodes = {
            "retrieve": self.retrieve_node,
            "generate": self.generate_node,
            "grade": self.grade_node
        }
        self.entry = "retrieve"

    def retrieve_node(self, state):
        query = state["keys"]["question"]
        print(f"🔍 检索节点: 处理查询 '{query}'")
        
        # 使用向量数据库检索相关文档
        retrieved_results = self.vector_db.retrieve(query, self.vector_db.vectorizer,top_k=3)
        
        if retrieved_results:
            # 提取检索到的文本
            documents = [result[0] for result in retrieved_results]
            state["keys"]["documents"] = "\n".join(documents)
            # 保存相似度信息供后续使用
            state["keys"]["retrieval_scores"] = str([result[1] for result in retrieved_results])
            print(f"✅ 检索到 {len(documents)} 个相关文档")
        else:
            state["keys"]["documents"] = "未找到相关文档"
            state["keys"]["retrieval_scores"] = "[]"
            print("⚠️ 未找到相关文档")
        
        return state

    def generate_node(self, state):
        print("🤖 生成节点: 生成答案...")
        documents = state["keys"]["documents"]
        query = state["keys"]["question"]
        
        # 简单生成逻辑：从文档中提取相关信息
        if documents != "未找到相关文档":
            if "截止日期" in query and "截止日期" in documents:
                # 查找包含截止日期的文档行
                doc_lines = documents.split("\n")
                answer_lines = [line for line in doc_lines if "截止日期" in line]
                answer = answer_lines[0] if answer_lines else "文档中没有找到具体的截止日期信息"
            elif "申请" in query and "申请" in documents:
                # 查找包含申请信息的文档行
                doc_lines = documents.split("\n")
                answer_lines = [line for line in doc_lines if "申请" in line]
                answer = answer_lines[0] if answer_lines else "文档中没有找到具体的申请信息"
            else:
                # 默认返回文档摘要
                answer = "根据检索到的文档，相关信息如下：" + documents[:300]
        else:
            answer = "抱歉，没有找到与您问题相关的文档信息。"
        
        state["keys"]["generation"] = answer
        return state

    def grade_node(self, state):
        print("📊 评分节点: 评估生成质量...")
        query = state["keys"]["question"]
        generation = state["keys"]["generation"]
        documents = state["keys"]["documents"]
        
        # 简单评分逻辑：检查生成内容是否包含文档中的关键词
        # 提取查询关键词（中文分词简化版）
        keywords = []
        for kw in ["截止日期", "申请", "要求", "条件", "时间", "费用", "材料"]:
            if kw in query:
                keywords.append(kw)
        
        # 如果没找到特定关键词，使用通用词
        if not keywords:
            keywords = [word for word in query.split() if len(word) > 1 and word not in ["的", "了", "在", "是", "有"]]
        
        # 检查生成内容的质量
        if documents == "未找到相关文档":
            state["keys"]["final_score"] = "no_documents"
            state["keys"]["assessment"] = "未找到相关文档，无法进行有效回答"
        elif any(kw in generation for kw in keywords) and len(generation) > 10:
            state["keys"]["final_score"] = "useful"
            state["keys"]["assessment"] = "生成内容与查询相关且信息完整"
        else:
            state["keys"]["final_score"] = "not_useful"
            state["keys"]["assessment"] = "生成内容与查询相关性不足"
        
        return state

    def run(self, state):
        print("🚀 开始Self-RAG工作流...")
        current = self.entry
        while current:
            print(f"➡️ 当前节点: {current}")
            state = self.nodes[current](state)
            # 工作流流转：retrieve → generate → grade → end
            if current == "retrieve":
                current = "generate"
            elif current == "generate":
                current = "grade"
            else:
                current = None
        print("🏁 Self-RAG工作流完成")
        return state


# In[62]:


# 5. 执行入口（更新为使用FAISSVectorDB2）
if __name__ == "__main__":
    # 初始化向量数据库
    print("=" * 50)
    print("🔧 初始化FAISS向量数据库")
    print("=" * 50)

    # 1. 创建文本处理器
    print("1. 初始化文本处理器...")
    text_processor = TextProcessor(chunk_size=500)
    
    # vector_db = FAISSVectorDB(embedding_dim=768)  # ERNIE-3.0-medium-zh的向量维度#【已弃用FAISSVectorDB】 
    vector_db = FAISSVectorDB2( )  # ERNIE-3.0-medium-zh的向量维度#【】使用FAISSVectorDB2 
    
    # 构建索引（替换为你的文件路径）
    # success = vector_db.build_from_files(#【已弃用FAISSVectorDB】 
    #     pdf_path="保.pdf",      # 替换为你的PDF文件路径
    #     excel_path="冲稳.xlsx"  # 替换为你的Excel文件路径
    #     # word_path="example.docx"  # 如有Word文件可添加
    # )
    
    #【】使用FAISSVectorDB2
    vectorizer = ERNIEVectorizer2(client=client)
    vector_db = FAISSVectorDB2(index_type="flat")
    
    # 4. 构建索引
    print("4. 构建向量索引...")
    success = vector_db.build_from_files(
        text_processor=text_processor,
        vectorizer=vectorizer,
        pdf_path="保.pdf",      # 替换为实际文件路径
        excel_path="冲稳.xlsx"  # 替换为实际文件路径
        # 可选: word_path="your_doc.docx"
    )

    if not success:
        print("❌ 向量数据库构建失败，程序退出")
        exit(1)
    
    # 显示统计信息
    print("\n📊 向量数据库统计信息:")
    stats = vector_db.get_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # 保存索引（可选）
    print("\n💾 保存向量索引...")
    vector_db.save_index("faiss_index")
    
    # 初始化Self-RAG工作流
    print("\n" + "=" * 50)
    print("🧠 初始化Self-RAG工作流")
    print("=" * 50)
    graph = SelfRAGGraph(vector_db)
    
    # 测试问题
    test_queries = [
        "美国留学的申请截止日期是什么时候？",
        "申请需要准备哪些材料？",
        "留学费用大概是多少？"
    ]
    
    for query in test_queries:
        print(f"\n" + "=" * 50)
        print(f"❓ 测试查询: {query}")
        print("=" * 50)
        
        test_state = {
            "keys": {"question": query}
        }
        
        final_state = graph.run(test_state)
        
        # 输出结果
        print("\n📋 ============Self-RAG 结果:==================")
        print(f"【问题】：{final_state['keys']['question']}")
        print(f"【检索状态】：{final_state['keys'].get('retrieval_scores', 'N/A')}")
        print(f"【生成答案】：{final_state['keys']['generation']}")
        print(f"【结果判定】：{final_state['keys']['final_score']} ({final_state['keys'].get('assessment', 'N/A')})")
        print(f"【相关文档预览】：{final_state['keys']['documents'][:200]}..." if len(final_state['keys']['documents']) > 200 else f"相关文档：{final_state['keys']['documents']}")


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
	