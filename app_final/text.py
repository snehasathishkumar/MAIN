import pymupdf # imports the pymupdf library
from pathlib import Path

def converter(file_path):
    
    extracted_text = ""
    doc = pymupdf.open(file_path) # open a document
    for page in doc: # iterate the document pages
        extracted_text += page.get_text() # get plain text encoded as UTF-8
    
    return extracted_text