"""Draws the judge handoff diagram and the withdrawal prompt card.

Matches the flat card style of assets/judgement-check.png. The prompt text and the
JSON reply come from svllm.judge, so the picture shows the same words as the code.
Needs Pillow. Outputs assets/ and site/assets/.
"""
import os
from PIL import Image, ImageDraw, ImageFont
from svllm.judge import run_examples

ROOT = os.path.dirname(os.path.abspath(__file__))
SANS = '/usr/share/fonts/truetype/macos/Inter-Regular.ttf'
SANS_MED = '/usr/share/fonts/truetype/macos/Inter-Medium.ttf'
SANS_BOLD = '/usr/share/fonts/truetype/macos/Inter-Bold.ttf'
MONO = '/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf'
MONO_MED = '/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Medium.ttf'

NAVY = (15, 21, 32)
INK = (22, 24, 31)
MUTED = (90, 96, 112)
LINE = (226, 228, 234)
BLUE_BG = (243, 246, 251)
BLUE_LINE = (210, 220, 234)
PINK_BG = (255, 247, 245)
PINK_LINE = (244, 214, 208)
GREEN = (18, 138, 98)
GREEN_BG = (232, 247, 239)
GREEN_LINE = (168, 214, 190)
PURPLE = (136, 43, 237)
AMBER = (146, 84, 12)
AMBER_BG = (255, 244, 224)
RED = (190, 40, 20)
WHITE = (255, 255, 255)


def font(path, size):
    return ImageFont.truetype(path, size)


def wrap(draw, text, fnt, max_w):
    lines = []
    for para in text.split('\n'):
        if para == '':
            lines.append('')
            continue
        cur = ''
        for word in para.split(' '):
            trial = word if not cur else cur + ' ' + word
            if draw.textlength(trial, font=fnt) <= max_w:
                cur = trial
                continue
            if cur:
                lines.append(cur)
            if draw.textlength(word, font=fnt) <= max_w:
                cur = word
                continue
            buf = ''
            for ch in word:
                if draw.textlength(buf + ch, font=fnt) <= max_w:
                    buf += ch
                else:
                    lines.append(buf)
                    buf = ch
            cur = buf
        if cur:
            lines.append(cur)
    return lines


def text(draw, xy, s, fnt, fill):
    draw.text(xy, s, font=fnt, fill=fill)


def pill(draw, xy, s, fnt, fg, bg, pad_x=10, pad_y=4):
    x, y = xy
    w = draw.textlength(s, font=fnt) + pad_x * 2
    h = fnt.size + pad_y * 2
    draw.rounded_rectangle((x, y, x + w, y + h), radius=h / 2, fill=bg)
    text(draw, (x + pad_x, y + pad_y - 1), s, fnt, fg)
    return w, h


def rounded(draw, box, fill, outline, width=2, radius=22):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, x0, x1, y, color, weight=4):
    draw.line((x0, y, x1 - 10, y), fill=color, width=weight)
    draw.polygon([(x1, y), (x1 - 14, y - 8), (x1 - 14, y + 8)], fill=color)


def paste_mark(im, xy, size=56):
    mark = Image.open(os.path.join(ROOT, 'assets', 'svmllm-mark-transparent.png')).convert('RGBA')
    mark.thumbnail((size, size), Image.Resampling.LANCZOS)
    im.paste(mark, xy, mark)


def handoff():
    W, H = 1600, 900
    im = Image.new('RGB', (W, H), WHITE)
    d = ImageDraw.Draw(im)
    title_f = font(SANS_BOLD, 46)
    sub_f = font(SANS, 22)
    h_f = font(SANS_BOLD, 22)
    b_f = font(SANS, 18)
    small = font(SANS_MED, 15)
    tag_f = font(SANS_BOLD, 13)
    foot_f = font(SANS, 18)
    text(d, (48, 40), 'Handoff to the judge', title_f, INK)
    text(d, (48, 102), 'The program runs exactly. The result is handed over in a prompt. The reviewer is simulated.', sub_f, MUTED)
    paste_mark(im, (1500, 36), 64)

    labels = [
        ('Transaction', NAVY, WHITE, WHITE, (186, 196, 214)),
        ('VM circuits', BLUE_BG, INK, BLUE_LINE, MUTED),
        ('Prompt wrapper', (246, 242, 255), INK, (214, 196, 240), MUTED),
        ('LLM judge', PINK_BG, INK, PINK_LINE, MUTED),
        ('allow, hold or block', WHITE, INK, LINE, MUTED),
    ]
    bodies = [
        'Wallet X sends a request: call a program, or withdraw an amount.',
        'Hand-set weights run the program exactly.',
        'The VM result, the history, and the run rules go inside one prompt.',
        'A canned reply for this scenario. No real language model is called.',
        '',
    ]
    y0, bh = 168, 430
    pad = 44
    gaps = [28, 156, 28, 28]
    avail = W - pad * 2 - sum(gaps)
    bw = avail // 5
    xs = []
    x = pad
    for i in range(5):
        xs.append(x)
        if i < 4:
            x += bw + gaps[i]
    for i, (name, fill, ink, outline, mut) in enumerate(labels):
        rounded(d, (xs[i], y0, xs[i] + bw, y0 + bh), fill, outline, 2, 26)
        text(d, (xs[i] + 20, y0 + 22), f'{i + 1}', font(SANS_BOLD, 14), mut)
        # wrap the title inside the box
        tw = bw - 40
        ty = y0 + 48
        for line in wrap(d, name, h_f, tw):
            text(d, (xs[i] + 20, ty), line, h_f, ink)
            ty += 28
        if i == 1:
            pill(d, (xs[i] + 20, ty + 6), 'exact result', tag_f, GREEN, (214, 240, 226))
            ty += 40
        if i == 3:
            pill(d, (xs[i] + 20, ty + 6), 'SIMULATED', tag_f, WHITE, RED)
            ty += 40
        ty += 12
        for line in wrap(d, bodies[i], b_f, tw):
            text(d, (xs[i] + 20, ty), line, b_f, mut if fill != NAVY else (214, 220, 230))
            ty += 26
        if i == 1:
            ty += 16
            for row in ('exit code', 'compute units', 'return data'):
                rounded(d, (xs[i] + 20, ty, xs[i] + bw - 20, ty + 36), WHITE, BLUE_LINE, 1, 10)
                text(d, (xs[i] + 32, ty + 7), row, small, INK)
                ty += 46
        if i == 4:
            ty += 8
            for lab, fg, bg in (('allow', (14, 122, 78), (214, 244, 228)), ('hold', AMBER, AMBER_BG), ('block', WHITE, RED)):
                pill(d, (xs[i] + 20, ty), lab, font(SANS_BOLD, 16), fg, bg, 14, 6)
                ty += 42
            ty += 10
            for line in wrap(d, 'With a reason. The decision can only get stricter.', b_f, tw):
                text(d, (xs[i] + 20, ty), line, b_f, MUTED)
                ty += 26

    # arrows. The wide gap between box 2 and box 3 is the handoff.
    mid_y = y0 + bh // 2
    for i, gap in enumerate(gaps):
        x0 = xs[i] + bw + 6
        x1 = xs[i] + bw + gap - 6
        if i == 1:
            arrow(d, x0, x1, mid_y - 16, GREEN, 6)
            label = 'handed over here'
            lf = font(SANS_BOLD, 15)
            lw = d.textlength(label, font=lf)
            lx = (x0 + x1) / 2 - lw / 2
            pill(d, (lx - 8, mid_y + 6), label, lf, WHITE, GREEN, 10, 5)
        else:
            arrow(d, x0, x1, mid_y, (180, 184, 196), 3)

    foot_y = y0 + bh + 36
    text(d, (48, foot_y), 'Trustless execution, then context-based judgement.', foot_f, MUTED)
    note = 'The model box is simulated. One merged model is future work and is untested.'
    nw = d.textlength(note, font=foot_f)
    text(d, (W - 48 - nw, foot_y), note, foot_f, MUTED)
    # Crop the unused canvas under the footer.
    return im.crop((0, 0, W, foot_y + 48))


def prompt_card():
    rows = {r['id']: r for r in run_examples('ref')}
    ex = rows['big-new-address']
    prompt = ex['prompt'].rstrip('\n')
    reason = ex['simulated']['reason']
    decision = ex['simulated']['decision']
    vm_at = prompt.index('VM RESULT\n')
    after_at = prompt.index('\nSENDER AND RECEIVER HISTORY')
    before, vm, after = prompt[:vm_at].rstrip('\n'), prompt[vm_at:after_at], prompt[after_at + 1:]

    W = 1600
    pad = 48
    inner = W - pad * 2
    mono = font(MONO, 18)
    mono_b = font(MONO_MED, 18)
    lh = 28
    probe = ImageDraw.Draw(Image.new('RGB', (10, 10)))
    text_w = inner - 64
    before_lines = wrap(probe, before, mono, text_w)
    vm_lines = wrap(probe, vm, mono_b, text_w - 28)
    after_lines = wrap(probe, after, mono, text_w)
    json_lines = wrap(probe, '{\n  "decision": "%s",\n  "reason": "%s"\n}' % (decision, reason), mono_b, text_w - 36)

    header_h = 156
    label_h = 28
    gap_before_vm = 14
    vm_pad = 16
    vm_label_h = 32
    gap_after_vm = 16
    card_pad_bot = 28
    before_h = len(before_lines) * lh
    vm_h = vm_pad + vm_label_h + len(vm_lines) * lh + vm_pad
    after_h = len(after_lines) * lh
    card_top = header_h
    before_y = card_top + 24 + label_h
    vm_top = before_y + before_h + gap_before_vm
    vm_bot = vm_top + vm_h
    after_y = vm_bot + gap_after_vm
    card_bot = after_y + after_h + card_pad_bot
    out_top = card_bot + 28
    out_h = 24 + 40 + 12 + len(json_lines) * lh + 28
    out_bot = out_top + out_h
    cap = 'The reply cites the history: 20500 is 100x the usual amount, and the destination has never been seen. The decision is block, which is not looser than the rule verdict.'
    cap_lines = wrap(probe, cap, font(SANS, 18), inner)
    H = out_bot + 20 + len(cap_lines) * 26 + 36

    im = Image.new('RGB', (W, H), WHITE)
    d = ImageDraw.Draw(im)
    text(d, (pad, 36), 'Withdrawal: prompt and simulated answer', font(SANS_BOLD, 40), INK)
    text(d, (pad, 92), 'Wallet X would like to withdraw 20500. Simulated model output, no real LLM was called.', font(SANS, 20), MUTED)
    paste_mark(im, (W - pad - 64, 32), 60)

    rounded(d, (pad, card_top, W - pad, card_bot), (250, 250, 252), LINE, 2, 24)
    text(d, (pad + 28, card_top + 22), 'PROMPT', font(SANS_BOLD, 14), PURPLE)
    yy = before_y
    for line in before_lines:
        text(d, (pad + 32, yy), line, mono, (40, 44, 58))
        yy += lh
    rounded(d, (pad + 20, vm_top, W - pad - 20, vm_bot), GREEN_BG, GREEN_LINE, 2, 16)
    text(d, (pad + 36, vm_top + 14), 'exact VM result, wrapped in the prompt', font(SANS_BOLD, 14), GREEN)
    vy = vm_top + vm_pad + vm_label_h
    for line in vm_lines:
        text(d, (pad + 40, vy), line, mono_b, (16, 70, 52))
        vy += lh
    yy = after_y
    for line in after_lines:
        text(d, (pad + 32, yy), line, mono, (40, 44, 58))
        yy += lh

    rounded(d, (pad, out_top, W - pad, out_bot), PINK_BG, PINK_LINE, 2, 24)
    pill(d, (pad + 24, out_top + 22), 'SIMULATED', font(SANS_BOLD, 14), WHITE, RED, 12, 5)
    text(d, (pad + 168, out_top + 24), 'Simulated model output', font(SANS_BOLD, 22), INK)
    jy = out_top + 76
    for line in json_lines:
        text(d, (pad + 32, jy), line, mono_b, NAVY)
        jy += lh
    cy = out_bot + 18
    for line in cap_lines:
        text(d, (pad, cy), line, font(SANS, 18), MUTED)
        cy += 26
    return im


def main():
    images = {
        'judge-handoff.png': handoff(),
        'judge-prompt-withdrawal.png': prompt_card(),
    }
    for name, im in images.items():
        for folder in ('assets', os.path.join('site', 'assets')):
            path = os.path.join(ROOT, folder, name)
            im.save(path, 'PNG')
            print(path, im.size)


if __name__ == '__main__':
    main()
