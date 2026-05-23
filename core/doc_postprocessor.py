from docx import Document


def merge_speaker_text(doc_path, output_path=None):
    """
    读取 Word 文档，按说话人合并连续内容，用空格隔开
    doc_path: 输入的 Word 文档路径
    output_path: 输出文件路径，如果不填则打印
    """
    doc = Document(doc_path)

    merged_lines = []
    last_speaker = None

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # 判断说话人
        if "：" in text:
            speaker, content = text.split("：", 1)
            speaker = speaker.strip()
            content = content.strip()
        else:
            speaker = last_speaker
            content = text

        if last_speaker == speaker and merged_lines:
            # 同一说话人，直接追加空格
            merged_lines[-1] += " " + content
        else:
            # 新说话人，添加新条目
            merged_lines.append(content if speaker is None else f"{speaker}: {content}")

        last_speaker = speaker

    # 输出结果
    result = "\n\n".join(merged_lines)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
    else:
        print(result)


# ================= 使用示例 =================
if __name__ == "__main__":
    doc_path = "consult.docx"  # 替换为你的 Word 文件路径
    output_path = "merged.txt"  # 输出文件路径
    merge_speaker_text(doc_path, output_path)
