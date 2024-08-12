import pytesseract

from langchain.document_loaders import UnstructuredFileLoader

from chinese_text_splitter import ChineseTextSplitter

pytesseract.pytesseract.tesseract_cmd = 'F:\\software\\Tesseract-OCR\\tesseract.exe'

loader = UnstructuredFileLoader(
    'F:\\workspace\\github\\xiedongmingming\\LangChain-ChatGLM-Webui\\demo\\汽车行业上海车展新车专题.pdf'
)

textsplitter = ChineseTextSplitter(pdf=True)

docs = loader.load_and_split()

print(docs)
