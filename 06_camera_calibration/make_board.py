"""Generate a vector A4 calibration board; requires reportlab."""
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.pagesizes import A4, landscape

output = Path(__file__).parent / 'output/pdf/chessboard_10x7_25mm_A4.pdf'
output.parent.mkdir(parents=True, exist_ok=True)
c = canvas.Canvas(str(output), pagesize=landscape(A4))
c.setTitle('Camera calibration chessboard - 10 x 7 squares - 25 mm')
width, height = landscape(A4)
x0, y0 = (width - 250 * mm) / 2, (height - 175 * mm) / 2
c.setFillColorRGB(1, 1, 1)
c.rect(0, 0, width, height, fill=1, stroke=0)
c.setFillColorRGB(0, 0, 0)
for row in range(7):
    for col in range(10):
        if (row + col) % 2 == 0:
            c.rect(x0 + col * 25 * mm, y0 + row * 25 * mm,
                   25 * mm, 25 * mm, fill=1, stroke=0)
c.setFont('Helvetica', 9)
c.drawCentredString(width / 2, 201 * mm,
                   '10 x 7 squares | 9 x 6 inner corners | Square: 25 mm | Print at 100% / Actual size')
c.setFont('Helvetica', 8)
c.drawCentredString(width / 2, 11.5 * mm, 'Scale check: this line must measure 100 mm after printing')
c.setLineWidth(0.6)
start, end = width / 2 - 50 * mm, width / 2 + 50 * mm
c.line(start, 8 * mm, end, 8 * mm)
for x in (start, end):
    c.line(x, 6.5 * mm, x, 9.5 * mm)
c.save()
print(output.resolve())
