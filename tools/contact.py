"""Contact sheet of RGBA plate frames over the stage gradient."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
src = Path(sys.argv[1]); out = sys.argv[2]; cols = int(sys.argv[3]) if len(sys.argv) > 3 else 8
files = sorted(src.glob("*.png"))
ims = [Image.open(f).convert("RGBA") for f in files]
w, h = ims[0].size
rows = (len(ims) + cols - 1) // cols
sheet = Image.new("RGB", (cols * w, rows * (h + 18)), (255, 255, 255))
font = ImageFont.truetype("/usr/share/fonts/truetype/montserrat/Montserrat-SemiBold.ttf", 12)
for i, (f, im) in enumerate(zip(files, ims)):
    bg = Image.new("RGBA", im.size, (236, 243, 248, 255)); bg.alpha_composite(im)
    x, y = (i % cols) * w, (i // cols) * (h + 18)
    sheet.paste(bg.convert("RGB"), (x, y + 18))
    fr = int(f.stem); ImageDraw.Draw(sheet).text((x + 4, y + 2), f"f{fr}  {fr/30:.2f}s", fill=(60, 90, 110), font=font)
sheet.save(out)
print(out, sheet.size)
