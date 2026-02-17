from docx import Document
import os

def export_word(data, roles, output_dir, filename):

    os.makedirs(output_dir, exist_ok=True)

    doc = Document()
    doc.add_heading("咨询记录", level=1)

    for item in data:
        role = roles[item["speaker"]]
        doc.add_paragraph(f"{role}：{item['text']}")

    path = os.path.join(output_dir, f"{filename}.docx")
    doc.save(path)

    return path
