import pymupdf
import os

fileAddress = os.path.join(os.getcwd(), "Emerson Resume.pdf")
document = pymupdf.open(fileAddress)

page = document[0]
temp = page.get_textpage_ocr()

output = page.get_text(textpage = temp)

print(output)