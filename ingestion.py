import asyncio
import os
import ssl
from typing import Dict,List, Any


from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

import certifi
from dotenv import load_dotenv

from logger import (Colors, log_error, log_header, log_info,log_success,log_warning)


load_dotenv()


# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUEST_CA_BUNDLE'] = certifi.where()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", show_progress_bar=True, chunk_size=50, retry_min_seconds=10)

vectorstore = PineconeVectorStore(index_name='langchain-docs-2026', embedding=embeddings)
# chroma = Chroma(persist_directory='chroma_db', embedding_function=embeddings)

tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages =1000)
tavily_crawl = TavilyCrawl()
tavily_extract = TavilyExtract()

def chunk_urls(urls: List[str], chunk_size: int = 20) -> List[List[str]]:
    """Split urls into chunks of specified size."""
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i : i + chunk_size]
        chunks.append(chunk)
    return chunks

async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str,Any]]:
    """Extract documents from a batch of urls"""
    try:
        log_info(f"TavilyExtract: Processing batch {batch_num} with {len(urls)} URLs", Colors.BLUE)
        docs = await tavily_extract.ainvoke(input= {"urls" : urls})
        log_success(f"TavilyExtract: Completed batch {batch_num} - extracted {len(docs.get('results', []))} documents")
        return docs
    except Exception as e:
        log_error(f"TavilyExtract: Failed to extract batch {batch_num} - {e}")
        return []
    
async def async_extract(url_batches: List[List[str]]):
    log_header("DOCUMENT EXTRACTION PHASE")
    log_info(f"TavilyExtract: Starting concurrent extraction of {len(url_batches)} batches", Colors.DARKCYAN)

    tasks = [extract_batch(batch, i+1) for i, batch in enumerate(url_batches)]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and flatten results
    all_pages = []
    failed_batches = 0
    for result in results:
        if isinstance(result, Exception):
            log_error(f"TavilyExtract: Batch failed with exception - {result}")
            failed_batches += 1
        else:
            for extracted_page in result["results"]:
                document = Document(page_content=extracted_page['raw_content'], metadata = {"source": extracted_page["url"]})
                all_pages.append(document)
    
    return all_pages


async def main():
    """Main async function to orchestrate the entire process"""

    # log_header("DOCUMENTATION INGESTION PIPELINE")
    # log_info("Tavily Crawl: Starting to crawl documentation from https://python.langchain.com/", Colors.PURPLE)

    # # crawl the documentation site
    # res = tavily_crawl.invoke({
    #     "url": "https://python.langchain.com", 
    #     "max_depth": 5 , 
    #     "extract_depth": "a"
    #     "dvanced",
    # })
    # all_docs = [Document(page_content=result['raw_content'], metadata= {"source": result['url']}) for result in res['results']]

    # log_success(f"Tavily Crawl: Successfully crawled {len(all_docs)} URLs from the documentation site")

    log_header("DOCUMENTATION INGESTION PIPELINE")
    log_info("Tavily Map: Starting to map documentation from https://python.langchain.com/", Colors.PURPLE)

    site_map = tavily_map.invoke('https://python.langchain.com/')

    log_success(f"Tavily Map: Successfully mapped {len(site_map['results'])} URLs from the documentation site")

    # Split urls into batches
    url_batches = chunk_urls(site_map['results'], chunk_size=20)
    log_info(f"URL Processing : {len(site_map['results'])} URLs into {len(url_batches)}", Colors.BLUE)

    # Extract document from url
    all_docs = await async_extract(url_batches)


    log_header("DOCUMENT CHUNKING PHASE")
    log_info(f"Text Splitter : Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap", Colors.YELLOW)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)

    log_success(f"Text Splitter : Created {len(splitted_docs)} chunks from {len(all_docs)} documents")

    # INdex documents into pinecone
    log_header("DOCUMENT INDEXING PHASE")
    log_info(f"Pinecone: Indexing {len(splitted_docs)} documents into Pinecone", Colors.GREEN)

    await vectorstore.aadd_documents(splitted_docs)

    log_success(f"Pinecone: Successfully indexed {len(splitted_docs)} documnts")

if __name__ == '__main__':
    asyncio.run(main())