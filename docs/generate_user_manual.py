#     *    
#    * *   
#   * * *  
#  *  *  * 
# *********
#  *  *  * 
#   * * *  
#    * *   
#     *    

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'out'
ASSETS = OUT / 'assets'
ASSETS.mkdir(parents=True, exist_ok=True)
PDF_PATH = OUT / 'Grok-Video-Studio-User-Manual.pdf'

W, H = 1400, 820
BG = '#0f172a'
CARD = '#1e293b'
ACCENT = '#38bdf8'
TXT = '#e2e8f0'
SUB = '#94a3b8'

try:
    FONT = ImageFont.truetype('DejaVuSans.ttf', 28)
    FONT_B = ImageFont.truetype('DejaVuSans-Bold.ttf', 34)
    FONT_S = ImageFont.truetype('DejaVuSans.ttf', 22)
except Exception:
    FONT = FONT_B = FONT_S = ImageFont.load_default()

shots = [
    ('step-1-dashboard.png', 'Step 1 - Dashboard & Prompt Setup', [
        'Enter concept, style, and duration settings.',
        'Choose prompt source: Grok API, OpenAI API, or local Ollama.',
        'Select provider: Grok Imagine, Sora 2, or Seedance 2.0.',
    ]),
    ('step-2-settings.png', 'Step 2 - Model/API Settings', [
        'Configure GROK_API_KEY, OPENAI_API_KEY, and SEEDANCE_API_KEY.',
        'Set your chat/video models and optional Ollama endpoint.',
        'Save platform credentials for YouTube/TikTok/Facebook/Instagram.',
    ]),
    ('step-3-stitching.png', 'Step 3 - Clip Review and Stitching', [
        'Select clips in Generated Videos list.',
        'Enable crossfade, interpolation (48/60fps), and upscale if needed.',
        'Optionally mix background music and export final render.',
    ]),
    ('step-4-upload.png', 'Step 4 - Social Publishing Workflow', [
        'Open platform upload tab (YouTube, TikTok, Facebook, Instagram).',
        'Choose API upload or browser automation upload.',
        'For CDP/UDP mode: Start Automation Chrome, Connect CDP, run workflow.',
    ]),
]

for fname, title, bullets in shots:
    img = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((40, 40, W-40, H-40), radius=22, fill=CARD, outline='#334155', width=3)
    d.text((80, 80), title, font=FONT_B, fill=ACCENT)
    d.rounded_rectangle((80, 150, W-80, H-120), radius=16, fill='#0b1220', outline='#334155', width=2)
    y = 210
    for b in bullets:
        d.text((120, y), f'• {b}', font=FONT, fill=TXT)
        y += 90
    d.text((120, H-165), 'Illustrative workflow screenshot generated from project documentation.', font=FONT_S, fill=SUB)
    img.save(ASSETS / fname)

c = canvas.Canvas(str(PDF_PATH), pagesize=letter)
page_w, page_h = letter


def draw_wrapped(text, x, y, width, leading=14, font_name='Helvetica', font_size=10):
    c.setFont(font_name, font_size)
    words = text.split()
    line = ''
    for w in words:
        test = (line + ' ' + w).strip()
        if c.stringWidth(test, font_name, font_size) <= width:
            line = test
        else:
            c.drawString(x, y, line)
            y -= leading
            line = w
    if line:
        c.drawString(x, y, line)
        y -= leading
    return y

# Cover
c.setFont('Helvetica-Bold', 24)
c.drawString(72, page_h - 72, 'Grok Video Studio - Full User Guide & Workflow Manual')
c.setFont('Helvetica', 12)
c.drawString(72, page_h - 98, 'Version: repository current state')
c.drawString(72, page_h - 116, 'Includes setup instructions, end-to-end workflows, and screenshot walkthroughs.')

cover_points = [
    'Generate AI videos using Grok/OpenAI/Seedance providers.',
    'Build prompts with Grok API, OpenAI API, or local Ollama.',
    'Preview, stitch, interpolate, upscale, and mix music before export.',
    'Publish by API or browser automation to YouTube, TikTok, Facebook, and Instagram.',
    'Use Automation Chrome + CDP + UDP service for resilient posting workflows.',
]
y = page_h - 150
for p in cover_points:
    y = draw_wrapped(f'• {p}', 86, y, page_w - 160, leading=16, font_size=11)

c.showPage()

sections = [
    ('1) Installation & First Run', [
        'Install Python 3.11+ and ffmpeg in PATH.',
        'Create a virtual environment and install requirements.txt.',
        'Launch with: python app.py',
        'Optional for automation browser control: python -m playwright install chromium',
    ]),
    ('2) Initial Configuration Checklist', [
        'Open Model/API Settings tab.',
        'Set GROK_API_KEY.',
        'Set OPENAI_API_KEY or OPENAI_ACCESS_TOKEN if using OpenAI services.',
        'Set OLLAMA_API_BASE and OLLAMA_CHAT_MODEL for local prompting (optional).',
        'Set SEEDANCE_API_KEY for Seedance generation.',
        'Configure upload credentials for each social platform you use.',
    ]),
    ('3) End-to-End Video Creation Workflow', [
        'Step A: Draft prompt and select prompt source.',
        'Step B: Choose video provider (Grok Imagine / Sora 2 / Seedance).',
        'Step C: Run generation for one or multiple variants.',
        'Step D: Review clips in Generated Videos and play previews.',
        'Step E: Stitch clips and enable crossfade/interpolation/upscale if needed.',
        'Step F: Export final video.',
    ]),
    ('4) Social Publishing Workflow', [
        'Open platform tab for YouTube, TikTok, Facebook, or Instagram.',
        'Select final video clip/file.',
        'Use Upload via API for direct integrations.',
        'Use Automate in Browser when API paths are not available or need parity with UI flow.',
        'For Automation Chrome path: Start Automation Chrome → Connect CDP → run upload action.',
    ]),
    ('5) Troubleshooting', [
        'If uploads fail, refresh credentials and re-authenticate platform accounts.',
        'If automation selectors drift, update workflows and validate current platform UI.',
        'If ffmpeg operations fail, verify ffmpeg is installed and in PATH.',
        'If no previews appear, confirm output directories and generation success logs.',
    ]),
]

for title, bullets in sections:
    c.setFont('Helvetica-Bold', 16)
    c.drawString(72, page_h - 72, title)
    y = page_h - 104
    for b in bullets:
        y = draw_wrapped(f'• {b}', 86, y, page_w - 160, leading=16, font_size=11)
    c.showPage()

for idx, (fname, title, bullets) in enumerate(shots, start=1):
    c.setFont('Helvetica-Bold', 16)
    c.drawString(72, page_h - 72, f'Screenshot Walkthrough {idx}: {title}')
    img_path = ASSETS / fname
    c.drawImage(str(img_path), 72, page_h - 420, width=page_w - 144, height=300, preserveAspectRatio=True, anchor='n')
    y = page_h - 450
    c.setFont('Helvetica', 11)
    for b in bullets:
        y = draw_wrapped(f'• {b}', 86, y, page_w - 160, leading=16, font_size=11)
    c.showPage()

c.setFont('Helvetica-Bold', 16)
c.drawString(72, page_h - 72, 'Appendix: Recommended Daily Workflow')
workflow = [
    '1. Start app and verify API keys in Model/API Settings.',
    '2. Generate 3-5 variants for each concept to improve selection quality.',
    '3. Review outputs and shortlist clips.',
    '4. Stitch + upscale only final candidates to save processing time.',
    '5. Publish via API first; use browser automation as fallback.',
    '6. Track platform responses in Activity Log for auditing.',
]
y = page_h - 104
for line in workflow:
    y = draw_wrapped(line, 86, y, page_w - 160, leading=16, font_size=11)

c.save()
print(f'Generated {PDF_PATH}')
print(f'Screenshots saved under {ASSETS}')
