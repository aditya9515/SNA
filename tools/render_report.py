from pathlib import Path
import json

import pymupdf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tmp" / "pdf_qa"
OUTPUT.mkdir(parents=True, exist_ok=True)
document = pymupdf.open(ROOT / "Project_Report.pdf")
page_info = []
for i, page in enumerate(document):
    pix = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
    pix.save(OUTPUT / f"page_{i+1:02}.png")
    words = page.get_text("words")
    outside = [word[4] for word in words if word[0] < 15 or word[2] > page.rect.width-15 or word[1] < 10 or word[3] > page.rect.height-10]
    page_info.append({"page": i+1, "characters": len(page.get_text()), "outside_page_margin": outside,
                      "start": page.get_text()[:160].replace("\n", " / ")})
for start in range(0, len(document), 12):
    sheet = Image.new("RGB", (1240, 1410), "#dbe3ed")
    draw = ImageDraw.Draw(sheet)
    for offset, number in enumerate(range(start, min(start+12, len(document)))):
        page_image = Image.open(OUTPUT / f"page_{number+1:02}.png")
        page_image.thumbnail((292, 414))
        x, y = (offset % 4)*310+9, (offset//4)*470+25
        sheet.paste(page_image, (x, y))
        draw.text((x, y-17), f"Page {number+1}", fill="black")
    sheet.save(OUTPUT / f"contact_{start//12+1}.png")
(OUTPUT / "page_audit.json").write_text(json.dumps(page_info, indent=2), encoding="utf-8")
print(json.dumps({"pages": len(document), "pages_with_outside_text": [r for r in page_info if r["outside_page_margin"]],
                  "page_starts": [{"page": r["page"], "start": r["start"]} for r in page_info]}, indent=2))
