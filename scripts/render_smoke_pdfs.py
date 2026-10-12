from pathlib import Path
import sys

import pypdfium2 as pdfium

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


root = Path(__file__).resolve().parents[1]
source = root / "output" / "smoke" / "pdf"
target_dir = root / "output" / "smoke" / "rendered"
target_dir.mkdir(parents=True, exist_ok=True)
for stale_image in target_dir.glob("smoke_*.png"):
    stale_image.unlink()

for pdf_path in sorted(source.glob("*.pdf")):
    document = pdfium.PdfDocument(str(pdf_path))
    for index, page in enumerate(document):
        target = target_dir / f"{pdf_path.stem}-page-{index + 1}.png"
        page.render(scale=1.6).to_pil().save(target)
        print(target)
