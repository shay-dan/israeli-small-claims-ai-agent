from pathlib import Path

import pypdfium2 as pdfium


root = Path(__file__).resolve().parents[1]
pdf_dir = root / "output" / "deep_audit" / "pdf"
render_dir = root / "output" / "deep_audit" / "rendered"
render_dir.mkdir(parents=True, exist_ok=True)

for stale in render_dir.glob("*.png"):
    stale.unlink()

count = 0
for pdf_path in sorted(pdf_dir.glob("*.pdf")):
    document = pdfium.PdfDocument(pdf_path)
    for page_index in range(len(document)):
        page = document[page_index]
        image = page.render(scale=1.65).to_pil()
        target = render_dir / f"{pdf_path.stem}_page_{page_index + 1:02d}.png"
        image.save(target)
        count += 1
    document.close()

print(f"Rendered {count} pages to {render_dir}")
