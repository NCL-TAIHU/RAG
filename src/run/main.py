from src.run.factory import AppFactory
from src.run.app import SearchApp
from src.core.filter import NCLFilter, Filter
DATASET = "ncl"  # Default dataset to use

def interact(app: SearchApp): 
    print("\n📚 Welcome to the Interactive Search App!")
    print("Type your query and press Enter to search.")
    print("Type `:rag` to toggle RAG mode, `:topk <num>` to change result count, or `:exit` to quit.")
    
    rag_enabled = False
    top_k = 5

    while True:
        user_input = input("\n>>> ").strip()
        if user_input.lower() in {":exit", "exit", "quit"}:
            print("👋 Exiting. Goodbye!")
            break
        elif user_input.lower() == ":rag":
            rag_enabled = not rag_enabled
            print(f"RAG mode {'enabled' if rag_enabled else 'disabled'}.")
            continue
        elif user_input.startswith(":topk"):
            try:
                top_k = int(user_input.split()[1])
                print(f"Result limit set to {top_k}.")
            except (IndexError, ValueError):
                print("Usage: :topk <int>")
            continue
        elif user_input.startswith(":"):
            print("❓ Unknown command.")
            continue

        # Run search
        f = Filter()
        results = app.search(query=user_input, filter = f, limit=top_k)
        print("\n🔍 Search Results:")
        if not results:
            print("No results found.")
            continue

        for i, doc in enumerate(results, 1):
            print(f"[{i}] {doc.key()}")
            for field, data in doc.content().items():
                print(f"   {field}: {data.contents}...")

        if rag_enabled:
            response = app.rag(user_input, results)
            print("\n💬 LLM Response:")
            print(response["generation"])

def test(app: SearchApp): 
    filters = [
        NCLFilter().set_fields(year=[109]), 
        NCLFilter().set_fields(category=["碩士"]),
        NCLFilter().set_fields(school_chinese=["國立中山大學"]),
        NCLFilter().set_fields(dept_chinese=["資訊工程學系"]),
        NCLFilter().set_fields(authors_chinese=['許佩鈴']),
        NCLFilter().set_fields(advisors_chinese=['魏大華', '陳洋元'])
    ]
    queries = [
        "深度學習",
        "自然語言處理",
        "機器學習",
        "資料挖掘",
        "人工智慧", 
        "計算機視覺",
    ]

    for query, filter in zip(queries, filters):
        print(f"\n🔍 Searching for: {query}")
        results = app.search(query=query, filter=filter, limit=5)
        if not results:
            print("No results found.")
            continue
        for i, doc in enumerate(results, 1):
            print(f"[{i}] {doc.key()}")
            for field, data in doc.content().items():
                print(f"   {field}: {data.contents}...")

        response = app.rag(query, results)
        print("\n💬 LLM Response:")
        print(response["generation"])

def main():
    print("🔎 Initializing SearchApp...")
    app = AppFactory.from_default(name = "dev", dataset=DATASET).build()
    test(app)
    interact(app)

if __name__ == "__main__":
    main()