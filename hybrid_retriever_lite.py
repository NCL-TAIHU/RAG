import os
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    WeightedRanker,
)
import ollama
ef = BGEM3EmbeddingFunction(device="cuda")

if os.path.exists("./milvus.db"):
    print("milvus.db 已存在，跳過連線建立")
else:

    abstract_path = "/home/pinyi/NCL_RAG_langgraph/history_paper_metadata/all/all摘要"
    data_path = "/home/pinyi/NCL_RAG_langgraph/history_paper_metadata/all/all論文基本資料_論文名稱＿中文關鍵字"
    content_path = "/home/pinyi/NCL_RAG_langgraph/history_paper_metadata/all/all目次"
    num_files = 2437
    metadata = {
                "id": [],
                "data": [],
                "content": [],
            }
    abstract = []
    for i in range(1, num_files + 1):
        filename = f"{i}.txt"
        file1_path = os.path.join(abstract_path, filename)
        file2_path = os.path.join(data_path, filename)
        file3_path = os.path.join(content_path, filename)

        with open(file1_path, "r", encoding="utf-8") as f1, \
            open(file2_path, "r", encoding="utf-8") as f2, \
            open(file3_path, "r", encoding="utf-8") as f3:
            content1 = f1.read()
            content2 = f2.read()
            content3 = f3.read()
            metadata['id'].append(str(i))
            metadata['data'].append(content2)
            metadata['content'].append(content3)
            abstract.append(content1)


    
    dense_dim = ef.dim["dense"]

    # Generate embeddings using BGE-M3 model
    docs_embeddings = ef(abstract)

    # Connect to Milvus given URI
    connections.connect(uri="./milvus.db")

    # Specify the data schema for the new Collection
    fields = [
        # Use auto generated id as primary key
        FieldSchema(
            name="pk", dtype=DataType.VARCHAR, is_primary=True, max_length=100
        ),
        # Store the original text to retrieve based on semantically distance
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
        # Milvus now supports both sparse and dense vectors,
        # we can store each in a separate field to conduct hybrid search on both vectors
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
        FieldSchema(name="data", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=20000),
        
    ]
    schema = CollectionSchema(fields)

    # Create collection (drop the old one if exists)
    col_name = "hybrid_demo"
    if utility.has_collection(col_name):
        Collection(col_name).drop()
    col = Collection(col_name, schema, consistency_level="Strong")

    # To make vector search efficient, we need to create indices for the vector fields
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    col.create_index("sparse_vector", sparse_index)
    dense_index = {"index_type": "AUTOINDEX", "metric_type": "IP"}
    col.create_index("dense_vector", dense_index)
    col.load()

    # For efficiency, we insert 50 records in each small batch
    for i in range(0, len(abstract), 50):
        batched_entities = [
            metadata['id'][i : i + 50],
            abstract[i : i + 50],
            docs_embeddings["sparse"][i : i + 50],
            docs_embeddings["dense"][i : i + 50],
            metadata['data'][i : i + 50],
            metadata['content'][i : i + 50],
        ]
        col.insert(batched_entities)
    print("Number of entities inserted:", col.num_entities)

# Enter your search query
query = "日治時期台灣的媒體在普及現代營養知識方面扮演了哪些角色？"
print(query)

# Generate embeddings for the query
query_embeddings = ef([query])
connections.connect(uri="./milvus.db")
col = Collection("hybrid_demo")  # collection 名稱
col.load()
def dense_search(col, query_dense_embedding, limit=3):
    search_params = {"metric_type": "IP", "params": {}}
    res = col.search(
        [query_dense_embedding],
        anns_field="dense_vector",
        limit=limit,
        output_fields=["pk", "text", "data", "content"],
        param=search_params,
    )[0]
    #return [hit.get("text") for hit in res]
    return res


def sparse_search(col, query_sparse_embedding, limit=3):
    search_params = {
        "metric_type": "IP",
        "params": {},
    }
    res = col.search(
        [query_sparse_embedding],
        anns_field="sparse_vector",
        limit=limit,
        output_fields=["pk", "text", "data", "content"],
        param=search_params,
    )[0]
    #return [hit.get("text") for hit in res]
    return res


def hybrid_search(
    col,
    query_dense_embedding,
    query_sparse_embedding,
    sparse_weight=1.0,
    dense_weight=1.0,
    limit=3,
):
    dense_search_params = {"metric_type": "IP", "params": {}}
    dense_req = AnnSearchRequest(
        [query_dense_embedding], "dense_vector", dense_search_params, limit=limit
    )
    sparse_search_params = {"metric_type": "IP", "params": {}}
    sparse_req = AnnSearchRequest(
        [query_sparse_embedding], "sparse_vector", sparse_search_params, limit=limit
    )
    rerank = WeightedRanker(sparse_weight, dense_weight)
    res = col.hybrid_search(
        [sparse_req, dense_req], rerank=rerank, limit=limit, output_fields=["pk", "text", "data", "content"]
    )[0]
    #return [hit.get("text") for hit in res]
    return res


print("Please enter the retrieval method -> dense_search / sparse_search / hybrid_search")
retrieval_method = input(": ")
if retrieval_method == "dense_search":
    retrieved_doc  = dense_search(col, query_embeddings["dense"][0])
    
        
elif retrieval_method == "sparse_search":
    retrieved_doc = sparse_search(col, query_embeddings["sparse"]._getrow(0))
    
elif retrieval_method == "hybrid_search":
    sparse_weight = float(input("Please enter the sparse_weight (0.0 ~ 1.0): "))
    dense_weight = float(input("Please enter the dense_weight (0.0 ~ 1.0): "))
    retrieved_doc = hybrid_search(
        col,
        query_embeddings["dense"][0],
        query_embeddings["sparse"]._getrow(0),
        sparse_weight=sparse_weight,
        dense_weight=dense_weight,
    )
    

else:
    print("you have entered the wrong term")

for doc in retrieved_doc:
    title_line = doc.get('data').splitlines()[0]
    print(f"id: {doc.get('pk')} {title_line}")

# print("dense_results")
# for hit in dense_results:
#     title_line = hit.get('data').splitlines()[0]
#     print(f"id: {hit.get('pk')} {title_line}")
#     # print(f"text: {hit.get('text')}")
#     # print(f"data: {hit.get('data')}")
#     # print(f"content: {hit.get('content')}")
#     # print(f"distance: {hit.distance}")  # 相似度
#     # print("------------")
# print("sparse_results")
# for hit in sparse_results:
#     title_line = hit.get('data').splitlines()[0]
#     print(f"id: {hit.get('pk')} {title_line}")
# print("hybrid_results")
# for hit in hybrid_results:
#     title_line = hit.get('data').splitlines()[0]
#     print(f"id: {hit.get('pk')} {title_line}")

prompt= ""
for i, doc in enumerate(retrieved_doc):
    prompt = prompt + f"{i+1}.\n{doc.get('data')}\n{doc.get('content')}\n"
    #print(doc.metadata['source'])
    #print(f"{i+1}.\n{doc.metadata['data']}\n\n目次：\n{doc.metadata['content']}\n\n摘要：\n{doc.page_content}")

prompt = prompt + f"\n問題： {query}\n"
prompt= prompt + "以上為和\"問題\"相關的\"論文標題\"以及\"論文關鍵字\"和\"目錄\"。先列出這5篇相關論文的論文名稱，再根據以上資訊對這些論文做總結，並給出可能的相關研究議題。只需列出論文名稱，不需列出論文其他資訊。總結以一段文字呈現，不要列點。生成格式為：\n以上論文皆...，根據搜尋結果，可以總結出以下...相關研究議題有..."

generation = ollama.generate(
    model='llama3.1:latest', 
    prompt=prompt, 
    stream = False,
    options={'num_predict': -1, 'keep_alive': 0},
    
)
response = generation['response']
print(f"LLM generation:\n{response}")

