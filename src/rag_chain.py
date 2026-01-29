from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def build_rag_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })
    
    prompt = PromptTemplate(
        template = """
        You are a helpful and knowledgeable assistant.
        
        Answer ONLY from the provided context.
        If the context is insufficient, respond with: "I don't know."
        
        Context:
        {context}
        
        Question:
        {question}
        """,
        input_variables=["context", "question"]
    )
    
    parser = StrOutputParser()
    
    main_chain = parallel_chain | prompt | llm | parser
    return main_chain
