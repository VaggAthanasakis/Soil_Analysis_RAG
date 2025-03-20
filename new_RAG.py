from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.core import VectorStoreIndex, DocumentSummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama
from llama_index.core import Settings
from langchain.llms import BaseLLM
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
import pathlib
import pdfplumber
import ocrmypdf
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
from langdetect import detect
import json
import os
import re
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import asyncio
from llama_index import SimpleDirectoryReader




# Function to extract text from a scanned PDF using OCR
# Converts the pdf pages to images and then performs OCR on each page
# in order to extract the text

def extract_text_from_scanned_pdf(file_path):
    pages = convert_from_path(file_path, dpi=600)  # Convert PDF pages to images
    extracted_text = ""
    
    for page in pages:
        page_text = pytesseract.image_to_string(page)  # Perform OCR on each page
        extracted_text += f"{page_text}\n"             # Add the text to the output
    
    return extracted_text


# Function to detect if the PDF is scanned or not
def is_pdf_scanned(file_path):
    reader = PdfReader(file_path)
    num_pages = len(reader.pages)
    
    # Try to extract text from each page
    for page_num in range(num_pages):
        page = reader.pages[page_num]
        text = page.extract_text()
        if text and text.strip():  # If there's text, it's not scanned
            return False   
        
    return True  # No text found, likely a scanned PDF


# The line BaseLLM.predict = patched_predict overrides the deprecated predict method and uses invoke instead.
# This should ensure that anywhere the predict method is called within llama_index, it uses invoke
def patched_predict(self, prompt, **kwargs):
    return self.invoke(prompt, **kwargs)


# load a specific prompt from a given file
def load_prompt(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()


# function to split text into segments of max_length length
def split_text(text, max_length=2000):
    segments = []
    while len(text) > max_length:
        # Find the last space within the 5000 character limit to avoid splitting words
        split_index = text[:max_length].rfind(' ')
        segments.append(text[:split_index])
        text = text[split_index + 1:]
    segments.append(text)  # Add any remaining text as the last segment
    return segments

# Convert Greek format numbers (X.XXX,XX to X,XXX.XX)
def preprocess_greek_numbers(text):   
    return re.sub(
        r'(\d{1,3}(?:\.\d{3})*)(,)(\d+)',
        lambda match: match.group(1).replace('.', ',') + '.' + match.group(3),
        text
    )



def translate_text_in_chunks(text, source_lang='auto', target_lang='en'):

    segments = split_text(text)
    translated_segments = []

    for segment in segments:
        # Preprocess numbers before translation
        processed_segment = preprocess_greek_numbers(segment)
        translated_text = GoogleTranslator(source=source_lang, target=target_lang).translate(processed_segment)
        translated_segments.append(translated_text)

    return ' '.join(translated_segments)


# function to translate the input of the LLM in English In order to achieve betterc accuracy
def text_translator(greek_text):
    translated_text = translate_text_in_chunks(greek_text)
   
    return translated_text

#############################################################
async def async_generate(chain, chunk, chunk_id):
    """Asynchronously generates corrected text for a given chunk and streams progress."""
    print(f"🔹 Processing chunk {chunk_id + 1}...")
    response = await asyncio.to_thread(chain.generate, [{"text": chunk}])  # Runs in separate thread
    corrected_text = response.generations[0][0].text
    #print(f"✅ Chunk {chunk_id + 1} completed:\n{chunk[:200]}\n =-=-=-=-= \n{corrected_text[:200]}...\n====================================\n")  # Print a preview
    return corrected_text


# postprocess
async def postprocess_extracted_text(extracted_text):
    """
    Processes extracted text in parallel using async LLaMA calls.
    - Uses overlapping chunks for better context.
    - Streams per-chunk output.
    """

    # Define a structured prompt for the model
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template=(
            "You are a professional text corrector and formatter specializing in "
            "Greek technical reports concerning soil analysis. Your task is to clean, format, and correct the extracted text while "
            "preserving its structure and its original meaning and without removing any information. You have to consider that "
            "most of the information was extracted from tables and the information should be structured in appropriate columns where applicable\n\n"
            "**Instructions:**\n"
            "- Do NOT remove any sections or numerical data.\n"
            "- Fix any **missing spaces** or **merged words**.\n"
            "- Properly **format section headers, tables, and data**.\n"
            "- Ensure **scientific units (%, mg/Kg, g/cm³) are correctly placed**.\n"
            "- Ensure **scientific names and acronyms (i.e. N, Fe, NO3-N, CaCO3) are corrected and preserved"
            "- Ensure **the measurement methodology is identified and associated to the type of measurement and their scientific units"
            "- Separate values from descriptions **for better readability**.\n"
            "- Keep the original technical terms in **Greek** without translation.\n\n"
            "**Extracted Text:**\n{text}\n\n"
            "**Corrected and Formatted Text (DO NOT OMIT ANYTHING):**"
        )
    )
    print("[INFO] Splitting text into chunks with overlap...")
    #chunks = split_text(extracted_text, chunk_size=800, overlap=50)
    chunks = split_text(extracted_text)

    # Define LangChain LLMChain
    #llm = OllamaLLM(model="llama3.1:8b", temperature = 0)
    llm = OllamaLLM(model="llama3.3", temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt_template)

    print("[INFO] Processing chunks asynchronously...")

    # Run all chunks asynchronously
    corrected_chunks = await asyncio.gather(*(async_generate(chain, chunk, i) for i, chunk in enumerate(chunks)))

    corrected_text = "\n".join(corrected_chunks)
    print("[INFO] Completed text correction.\n")
    return corrected_text



## main
BaseLLM.predict = patched_predict

#input_file_path = "/home/eathanasakis/Thesis/Soil_Analysis_RAG/Resources/Soil_Analysis_Resources/Soilanalysis-38-Zannias/240436_Ζαννιάς-Κάμπος-Θ.Κώστας.pdf"
#input_file_path = "/home/eathanasakis/Thesis/Soil_Analysis_RAG/Resources/Soil_Analysis_Resources/Soilanalysis-38-Zannias/240438-zannias-kephales-bio.pdf" 
#input_file_path = "/home/eathanasakis/Thesis/Soil_Analysis_RAG/Resources/Soil_Analysis_Resources/Soilanalysis-38-Zannias/240440-kottas_midNYYf.pdf" 

#input_file_path = "/home/eathanasakis/Thesis/Soil_Analysis_RAG/Resources/Soil_Analysis_Resources/Soilanalysis-38-Zannias/240440-kottas.pdf"
#input_file_path = "/home/eathanasakis/Thesis/Soil_Analysis_RAG/Resources/Soil_Analysis_Resources/Soilanalysis-38-Zannias/240445-zannias-beli.pdf" 
input_file_path = "/home/eathanasakis/Thesis/Soil_Analysis_RAG/Resources/Soil_Analysis_Resources/240437 Δούμα.pdf" # to Ca 



################################ Second PDF type ################################
 
#input_file_path = "/home/eathanasakis/Thesis/Soil_Analysis_RAG/Resources/Soil_Analysis_Resources/ΕΔΑΦΟΣ ΤΟΠΟΘ ΚΑΣΑΠΑΚΗΣ/SoilAnalysis-Kasapakis_1_merge.pdf" # xalia
#input_file_path = "/home/eathanasakis/Thesis/Soil_Analysis_RAG/Resources/Soil_Analysis_Resources/ΕΔΑΦΟΣ ΤΟΠΟΘ ΚΑΣΑΠΑΚΗΣ/ΕΔΑΦΟΣ ΤΟΠΟΘ 1 20221103 114361 (1).pdf"
# input_file_path = "/home/eathanasakis/Thesis/Soil_Analysis_RAG/Resources/Soil_Analysis_Resources/ΕΔΑΦΟΣ ΤΟΠΟΘ ΚΑΣΑΠΑΚΗΣ/ΕΔΑΦΟΣ ΤΟΠΟΘ 2 20221103 114362 (1).pdf"
# input_file_path = "/home/eathanasakis/Thesis/Soil_Analysis_RAG/Resources/Soil_Analysis_Resources/ΕΔΑΦΟΣ ΤΟΠΟΘ ΚΑΣΑΠΑΚΗΣ/ΕΔΑΦΟΣ ΤΟΠΟΘ 7 20221103 114363 (1).pdf"
# input_file_path = "/home/eathanasakis/Thesis/Soil_Analysis_RAG/Resources/Soil_Analysis_Resources/ΕΔΑΦΟΣ ΤΟΠΟΘ ΚΑΣΑΠΑΚΗΣ/ΕΔΑΦΟΣ ΤΟΠΟΘ 9 20221103 114364 (1).pdf"


response_file = "/home/eathanasakis/Thesis/Soil_Analysis_RAG/outputs/RESPONSE.txt"
text_output_file = ("/home/eathanasakis/Thesis/Soil_Analysis_RAG/outputs/TEST_PDF_TEXT.txt")
corrected_text_file = "/home/eathanasakis/Thesis/Soil_Analysis_RAG/outputs/GREEK_CORRECTED.txt"


# Detect if the PDF is scanned
if is_pdf_scanned(input_file_path):
    # If the PDF is scanned, use OCR to extract the text
    # using ocr and then process it normally
    print("\nPerforming OCR...\n")
    ocrmypdf.ocr(input_file_path, input_file_path, image_dpi=600)    
    
# If it's not scanned, load the document normally
# Use PDFPlumber
translated_output_file = "outputs/Translated.txt"
greek_output_file = "outputs/Greek.txt"

with pdfplumber.open(input_file_path) as pdf:
    full_text = ""
    for page in pdf.pages:
        full_text += page.extract_text()  # Extracts text page-by-page

# full_text = ""
# with open(input_file_path, "r", encoding="utf-8") as file:
#     full_text = file.read()  # Reads the entire file at once


pathlib.Path(greek_output_file).write_bytes(full_text.encode())

corrected_text = asyncio.run(postprocess_extracted_text(full_text))

pathlib.Path(corrected_text_file).write_bytes(corrected_text.encode())


# Check if we have to translate the text
if (detect(corrected_text) != 'en'):
    print("\nTranslating the text..")
    translated_text = text_translator(corrected_text)
else:
    translated_text = corrected_text

pathlib.Path(translated_output_file).write_bytes(translated_text.encode())

documents = [Document(text=translated_text)]




# load the LLM that we are going to use
llm = OllamaLLM(model="llama3.1:8b", temperature = 0)
#llm = OllamaLLM(model="llama3.3:latest", temperature = 0)


embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# The Settings class in llama_index (formerly known as GPT Index) is used to configure
# global parameters that influence how the library interacts with language models (LLMs),
# embedding models, and other system components.
Settings.llm = llm
Settings.embed_model = embed_model
Settings.context_window = 2048


general_prompt = load_prompt("/home/eathanasakis/Thesis/RAG_Query/Prompts/Soil_Analysis_JSON_prompt.txt")
plumbing_prompt = load_prompt("/home/eathanasakis/Thesis/RAG_Query/Prompts/Plumbing_prompt.txt")
Mechanical_comp_prompt = load_prompt("/home/eathanasakis/Thesis/RAG_Query/Prompts/Mechanical_comp.txt")
Physicochemical_prompt = load_prompt("/home/eathanasakis/Thesis/RAG_Query/Prompts/Physicochemical_prompt.txt")
Nutrients_prompt_1 = load_prompt("/home/eathanasakis/Thesis/RAG_Query/Prompts/Nutrients_prompt_1.txt")
Nutrients_prompt_2 = load_prompt("/home/eathanasakis/Thesis/RAG_Query/Prompts/Nutrients_prompt_2.txt")
Nutrients_prompt_3 = load_prompt("/home/eathanasakis/Thesis/RAG_Query/Prompts/Nutrients_prompt_3.txt")
Nutrients_prompt_4 = load_prompt("/home/eathanasakis/Thesis/RAG_Query/Prompts/Nutrients_prompt_4.txt")
specific_params_prompt = load_prompt("/home/eathanasakis/Thesis/RAG_Query/Prompts/specific_params_prompt.txt")

prompts = [
    plumbing_prompt,
    Mechanical_comp_prompt,
    Physicochemical_prompt,
    Nutrients_prompt_1,
    Nutrients_prompt_2,
    Nutrients_prompt_3,
    Nutrients_prompt_4,
    specific_params_prompt
]

# Dictionary to store merged results
combined_results = {}

# Specify the splitter
splitter = SentenceSplitter(chunk_size=700)



#  This is a class from the llama_index library that represents a vector store index for efficient retrieval
#  of documents based on their semantic similarity.
vector_store_index = VectorStoreIndex.from_documents(documents, splitter=splitter)   

# Build the query engine from the vector store index
query_engine_vector_index = vector_store_index.as_query_engine()


#query_engine = vector_store_index.as_query_engine(llm=llm, similarity_top_k=5, similarity_threshold=0.7)


# Process each prompt and merge results
max_retries = 3  # Maximum number of retries per prompt

for prompt in prompts:
    for attempt in range(max_retries):
        response = query_engine_vector_index.query(prompt)
        print("\n",response)
        # Convert string response to dictionary
        try:
            response_dict = json.loads(str(response).replace("'", '"'))
            combined_results.update(response_dict)
            break  # Exit retry loop if successful
        except json.JSONDecodeError as e:
            if attempt == max_retries - 1:
                print(f"Skipping prompt: {prompt[:50]} after {max_retries} attempts.")

# Write combined results to file
with open(response_file, "w", encoding='utf-8') as file:
    json.dump(combined_results, file, indent=4, ensure_ascii=False)



