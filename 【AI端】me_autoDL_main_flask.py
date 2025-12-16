from flask import Flask, jsonify, request
# 仅引入接口所需的核心组件（去除无关导入）
#from demo import client, TextEmbedding, SimpleVectorStore, RAGSystem, knowledge_base 
from me_autoDL_SelfRagSystem import client, TextProcessor, ERNIEVectorizer2, FAISSVectorDB2, SelfRAGGraph 
from me_autoDL_MultiAgentsSystem import   MainRun

# 初始化 Flask 应用
app = Flask(__name__)

# ---------------------- 初始化 TextProcessor----------------------
text_processor = TextProcessor(chunk_size=500)

# ---------------------- 初始化 向量数据库----------------------
vectorizer = ERNIEVectorizer2(client=client)
vector_db = FAISSVectorDB2(index_type="flat")

# ---------------------- 构建向量索引----------------------
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

# ---------------------- 初始化Self-RAG工作流----------------------
print("\n" + "=" * 50)
print("🧠 初始化Self-RAG工作流")
print("=" * 50)
graph = SelfRAGGraph(vector_db)


# ---------------------- 多智能体系统：初始化----------------------
mr  =  MainRun()

# ---------------------- 启动http 接口 ----------------------
@app.route('/')
def home():
    return 'Hello哇! （ai留学项目的关键字是：/study_abroad_api）'

@app.route('/study_abroad_api', methods=['POST'])
async def rag_qa():
    """POST 方法：RAG 问答接口（仅保留核心逻辑）"""
    try:
        data = request.get_json()
        question = data.get('question', '什么是人工智能？')  # 默认查询
        
        # 调用核心 RAG 逻辑
        #result = rag_system.query(question)
		# ---------------------- Self-RAG执行----------------------
        graph_state = {
                "keys": {"question": question}
        }
			
        final_state = graph.run(graph_state)

		# 输出结果
        print("\n📋 ============Self-RAG 结果:==================")
        print(f"【问题】：{final_state['keys']['question']}")
        print(f"【检索状态】：{final_state['keys'].get('retrieval_scores', 'N/A')}")
        print(f"【生成答案】：{final_state['keys']['generation']}")
        print(f"【结果判定】：{final_state['keys']['final_score']} ({final_state['keys'].get('assessment', 'N/A')})")
        print(f"【相关文档预览】：{final_state['keys']['documents'][:200]}..." if len(final_state['keys']['documents']) > 200 else f"相关文档：{final_state['keys']['documents']}")


        similarity = -1

		# ---------------------- 基于RAG结果：构建prompt----------------------
        query_combine = question
        if ('useful' in final_state['keys']['final_score']  and 'not_useful' not in final_state['keys']['final_score'] ):
                query_combine +=  final_state['keys']['generation']

        #print(result)
        print("---"*80)
        #判断result是否是：特定的json格式
        #if ('status' in result and  'question' in result and  'sources' in result): #说明是旧版RAGSystem的返回。	
        #        similarity =round(item['similarity'], 4)		
        #        ret=  await mr.run_once(query_combine)
        #else:
        #        ret=  await mr.run_once(query_combine)
        ret=  await mr.run_once(query_combine)
        print(f"  ret   ")
        print("=="*80)


        return jsonify({
            'status': 'success' ,
            'question': question,
            'answer': ret ,
            'sources': [
                {
                    'document': 'null',
                    'similarity':  similarity 
                } 
            ]
        })

		
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"处理失败：{str(e)}"
        }), 500

# 启动应用（仅保留核心启动逻辑）
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080) # 生产环境建议关闭 debug// 在Codelab开发测试阶段端口号可以修改，正式部署时必须为8080
