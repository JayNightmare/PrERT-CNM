import sys
from pypdf import PdfReader

def extract_pdf(path):
    try:
        reader = PdfReader(path)
        text = ""
        for i, page in enumerate(reader.pages):
            text += f"--- PAGE {i+1} ---\n"
            text += page.extract_text() + "\n"
        
        with open("iso_text.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print("Successfully extracted to iso_text.txt")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    extract_pdf(sys.argv[1])
