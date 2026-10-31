"""Builds brand assets (cropped logo, transparent variants, favicons, OG card) from the source JPG."""
from PIL import Image, ImageDraw, ImageFont
import numpy as np, sys
SRC = sys.argv[1] if len(sys.argv) > 1 else '/workspace/brand/svmllm-logo.jpg'
SITE = '/workspace/svllm-site'
im = Image.open(SRC).convert('RGB'); a = np.asarray(im).astype(int)
nw = a.min(2) < 225
def crop(box, pad, square=False):
    x0, y0, x1, y1 = box
    c = im.crop((x0 - pad, y0 - pad, x1 + pad, y1 + pad))
    if square:
        s = max(c.size); bg = Image.new('RGB', (s, s), 'white'); bg.paste(c, ((s - c.size[0]) // 2, (s - c.size[1]) // 2)); c = bg
    return c
ys, xs = np.where(nw); full = crop((xs.min(), ys.min(), xs.max(), ys.max()), 60)
rows = nw.sum(1); gap = next(y for y in range(ys.min() + 50, ys.max()) if rows[y] == 0)
ys2, xs2 = np.where(nw[:gap]); mark = crop((xs2.min(), ys2.min(), xs2.max(), ys2.max()), 40, square=True)
def transparent(c):
    arr = np.asarray(c).astype(float); w = arr.min(2)
    alpha = np.clip((240 - w) / 90.0, 0, 1)
    rgb = (arr - 255 * (1 - alpha[..., None])) / np.maximum(alpha[..., None], 1e-3)
    return Image.fromarray(np.dstack([np.clip(rgb, 0, 255), alpha * 255]).astype('uint8'), 'RGBA')
for d in ['assets', SITE + '/assets']:
    full.save(f'{d}/svmllm-logo.png'); mark.resize((512, 512), Image.LANCZOS).save(f'{d}/svmllm-mark.png')
    transparent(full).save(f'{d}/svmllm-logo-transparent.png')
    transparent(mark).resize((512, 512), Image.LANCZOS).save(f'{d}/svmllm-mark-transparent.png')
transparent(mark).resize((32, 32), Image.LANCZOS).save(SITE + '/favicon-32.png')
mark.resize((180, 180), Image.LANCZOS).save(SITE + '/apple-touch-icon.png')
mark.resize((48, 48), Image.LANCZOS).save(SITE + '/favicon.ico', sizes=[(16, 16), (32, 32), (48, 48)])
W, H = 1200, 630
yy, xx = np.mgrid[0:H, 0:W]
base, pur, grn = np.array([9, 8, 16]), np.array([136, 43, 237]), np.array([23, 195, 139])
d1 = np.exp(-(((xx - 250) / 420) ** 2 + ((yy - 120) / 300) ** 2)); d2 = np.exp(-(((xx - 1000) / 420) ** 2 + ((yy - 560) / 300) ** 2))
og = Image.fromarray((base + d1[..., None] * (pur - base) * 0.35 + d2[..., None] * (grn - base) * 0.3).clip(0, 255).astype('uint8'))
tl = transparent(full); tl.thumbnail((520, 440), Image.LANCZOS); og.paste(tl, (60, (H - tl.size[1]) // 2), tl)
dr = ImageDraw.Draw(og)
fb = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 42); fr = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 26)
for i, t in enumerate(['A Solana VM', 'compiled into', 'transformer weights']): dr.text((640, 200 + 56 * i), t, font=fb, fill=(240, 238, 250))
dr.text((640, 390), 'sBPF runs in forward passes.', font=fr, fill=(170, 168, 190)); dr.text((640, 428), 'No training. Exact results.', font=fr, fill=(170, 168, 190))
og.save(SITE + '/assets/og.png'); og.save('assets/og.png')
