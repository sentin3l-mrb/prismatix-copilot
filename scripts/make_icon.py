"""Generate assets/appicon.png — a minimal, handcrafted app icon.

Draws the icon as a tiny hand-written PDF (a dark rounded square with a white
'P'), then macOS's own conversion tools turn it into .icns for the .app.
No third-party imaging libraries needed.
"""

import sys
from pathlib import Path

W = 512

PDF_TEXT = f"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {W}]
           /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length 220 >>
stream
0.055 0.067 0.082 rg
34 34 444 444 re f
0.415 0.718 1.0 rg
BT /F1 300 Tf 150 178 Td (P) Tj ET
endstream
endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >> endobj
xref
0 6
trailer << /Size 6 /Root 1 0 R >>
%%EOF"""

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "assets/appicon.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    # /Length must equal the real stream byte count; compute it
    stream = f"0.055 0.067 0.082 rg\n34 34 444 444 re f\n0.415 0.718 1.0 rg\nBT /F1 300 Tf 150 178 Td (P) Tj ET"
    body = PDF_TEXT.replace("/Length 220", f"/Length {len(stream.encode())}")
    out.write_text(body)
    print(f"wrote {out}")
