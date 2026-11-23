"""Contact sheet of composited RGB frames: python3 tools/sheet.py DIR OUT COLS WIDTH"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
src = Path(sys.argv[1]); out = sys.argv[2]; cols = int(sys.argv[3]); tw = int(sys.argv[4])
files = sorted(src.glob("*.png"))
ims = [Image.open(f).convert("RGB") for f in files]
w0, h0 = ims[0].size
th = int(h0 * tw / w0)
rows = (len(ims) + cols - 1) // cols
sheet = Image.new("RGB", (cols * tw, rows * (th + 18)), (255, 255, 255))
font = ImageFont.truetype("/usr/share/fonts/truetype/montserrat/Montserrat-SemiBold.ttf", 12)
for i, (f, im) in enumerate(zip(files, ims)):
    x, y = (i % cols) * tw, (i // cols) * (th + 18)
    sheet.paste(im.resize((tw, th), Image.LANCZOS), (x, y + 18))
    fr = int(f.stem); ImageDraw.Draw(sheet).text((x + 4, y + 2), f"f{fr}  {fr/30:.2f}s", fill=(60, 90, 110), font=font)
sheet.save(out); print(out, sheet.size)
