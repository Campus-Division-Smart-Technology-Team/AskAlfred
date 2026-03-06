from markdown_pdf import MarkdownPdf, Section

with open("README.md", "r", encoding="utf-8") as f:
    markdown_text = f.read()

pdf = MarkdownPdf(toc_level=2)
pdf.add_section(Section(markdown_text))
pdf.save("README.pdf")
print("PDF created: README.pdf")
