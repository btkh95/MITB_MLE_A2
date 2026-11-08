"""Build the platypus mascot pose set used in the slide deck.

Writes one standalone SVG per pose next to this script, then renders each one
to a PNG (800px wide, transparent background) with headless Chrome or Edge.

Usage, from the repo root:
    python docs/mascot/make_mascots.py           # SVGs and PNGs
    python docs/mascot/make_mascots.py --no-png  # SVGs only

All poses share the same body, face and palette (constants below), so a change
to one of those constants updates every pose at once.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

out = Path(__file__).resolve().parent

RUST, RUST_DARK, CREAM, INK = "#D9773F", "#A9502A", "#F6E1CF", "#2C2C2A"
BLUSH, SHINE = "#EFA77C", "#F0B48F"
GOLD, GOLD_DARK, GOLD_LIGHT = "#F2B544", "#C98A16", "#FAD98C"
BILL_GREEN, BILL_LINE, BILL_LIGHT = "#7FA878", "#4F7A4A", "#B9D3B2"
PIPE, PIPE_LIGHT, PIPE_DARK = "#8A8780", "#A8A59D", "#6E6B65"

TAIL = f'<ellipse cx="160" cy="154" rx="30" ry="13" fill="{RUST_DARK}" transform="rotate(-22 160 154)"/>'
BODY = f'<ellipse cx="100" cy="112" rx="66" ry="62" fill="{RUST}"/>'
BELLY = f'<ellipse cx="100" cy="146" rx="40" ry="26" fill="{CREAM}"/>'
FEET = (f'<ellipse cx="80" cy="174" rx="13" ry="6" fill="{INK}"/>'
        f'<ellipse cx="120" cy="174" rx="13" ry="6" fill="{INK}"/>')
FACE = (f'<ellipse cx="62" cy="120" rx="7" ry="4" fill="{BLUSH}"/>'
        f'<ellipse cx="138" cy="120" rx="7" ry="4" fill="{BLUSH}"/>'
        f'<rect x="84" y="112" width="32" height="15" rx="7.5" fill="{INK}"/>'
        f'<path d="M90 116 H104" stroke="#5F5E5A" stroke-width="2" stroke-linecap="round"/>'
        f'<path d="M60 70 Q72 62 84 68" stroke="{SHINE}" stroke-width="4" fill="none" stroke-linecap="round"/>')
EYE_L = (f'<circle cx="75" cy="106" r="7" fill="{INK}"/>'
         f'<circle cx="77.5" cy="103.5" r="2.4" fill="#FFFFFF"/>')
EYE_R = (f'<circle cx="125" cy="106" r="7" fill="{INK}"/>'
         f'<circle cx="127.5" cy="103.5" r="2.4" fill="#FFFFFF"/>')
EYES_HAPPY = (f'<path d="M68 108 Q75 99 82 108" stroke="{INK}" stroke-width="4" fill="none" stroke-linecap="round"/>'
              f'<path d="M118 108 Q125 99 132 108" stroke="{INK}" stroke-width="4" fill="none" stroke-linecap="round"/>')
STANDING = TAIL + BODY + BELLY + FEET + FACE


def arm(cx, cy, rx=8, ry=10, rot=0):
    t = f' transform="rotate({rot} {cx} {cy})"' if rot else ""
    return f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{RUST_DARK}"{t}/>'


def coin(cx, cy, r=25):
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{GOLD}"/>'
            f'<circle cx="{cx}" cy="{cy}" r="{r * 0.72:.1f}" fill="none" stroke="{GOLD_DARK}" stroke-width="3"/>'
            f'<path d="M{cx} {cy - 9} V{cy + 9} M{cx - 5} {cy - 5} Q{cx} {cy - 10} {cx + 5} {cy - 5} '
            f'Q{cx - 5} {cy} {cx + 5} {cy + 5} Q{cx} {cy + 10} {cx - 5} {cy + 5}" '
            f'stroke="{GOLD_DARK}" stroke-width="2.5" fill="none" stroke-linecap="round"/>'
            f'<path d="M{cx - 14} {cy - 16} Q{cx - 10} {cy - 20} {cx - 5} {cy - 19}" '
            f'stroke="{GOLD_LIGHT}" stroke-width="3" fill="none" stroke-linecap="round"/>')


def small_coin(cx, cy, r, w=2):
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{GOLD}" stroke="{GOLD_DARK}" stroke-width="{w}"/>'


def flat_coin(cx, cy):
    return f'<ellipse cx="{cx}" cy="{cy}" rx="17" ry="6" fill="{GOLD}" stroke="{GOLD_DARK}" stroke-width="2"/>'


poses = {}

poses["platypus-base"] = STANDING + EYE_L + EYE_R + arm(38, 132, 7, 11, 25) + arm(162, 126, 7, 11, -25)

poses["platypus-title-bill"] = (
    STANDING + EYE_L + EYE_R
    + '<g transform="rotate(-6 100 150)">'
    + f'<rect x="64" y="134" width="72" height="34" rx="3" fill="{BILL_GREEN}"/>'
    + f'<rect x="69" y="139" width="62" height="24" rx="2" fill="none" stroke="{BILL_LINE}" stroke-width="2"/>'
    + f'<circle cx="100" cy="151" r="8" fill="{BILL_LIGHT}"/>'
    + f'<path d="M100 145 V157 M97 148 Q100 145 103 148 Q97 151 103 154 Q100 157 97 154" '
      f'stroke="{BILL_LINE}" stroke-width="1.8" fill="none" stroke-linecap="round"/>'
    + '</g>' + arm(62, 150) + arm(138, 146))

poses["platypus-modelbank-coin"] = STANDING + EYE_L + EYE_R + coin(100, 150) + arm(77, 150) + arm(123, 150)

poses["platypus-results-cheer"] = (
    STANDING + EYES_HAPPY + arm(40, 134, 7, 11, 25) + arm(164, 82, 7, 13, 25)
    + '<g transform="rotate(-14 172 52)">'
    + f'<rect x="150" y="40" width="46" height="24" rx="2" fill="{BILL_GREEN}"/>'
    + f'<rect x="154" y="44" width="38" height="16" rx="1.5" fill="none" stroke="{BILL_LINE}" stroke-width="1.6"/>'
    + f'<circle cx="173" cy="52" r="5" fill="{BILL_LIGHT}"/></g>'
    + flat_coin(66, 196) + flat_coin(66, 190) + flat_coin(132, 196) + flat_coin(100, 200)
    + small_coin(152, 186, 10))

# Pipe is drawn after the body so the lower body hides inside it.
poses["platypus-pipeline-peek"] = (
    f'<rect x="0" y="178" width="28" height="26" fill="{PIPE}"/>'
    f'<rect x="6" y="172" width="10" height="38" rx="2" fill="{PIPE_DARK}"/>'
    f'<ellipse cx="176" cy="104" rx="24" ry="10" fill="{RUST_DARK}" transform="rotate(-40 176 104)"/>'
    + BODY + FACE + EYE_L + EYE_R
    + f'<rect x="24" y="146" width="152" height="69" fill="{PIPE}"/>'
    f'<rect x="34" y="158" width="9" height="57" fill="{PIPE_LIGHT}"/>'
    f'<rect x="14" y="136" width="172" height="22" rx="6" fill="{PIPE_DARK}"/>'
    f'<rect x="20" y="140" width="160" height="4" rx="2" fill="{PIPE}"/>'
    + arm(66, 139, 12, 8) + arm(134, 139, 12, 8)
    + '<path d="M62 142 V146 M70 142 V146 M130 142 V146 M138 142 V146" '
      'stroke="#8A3F1F" stroke-width="1.6" stroke-linecap="round"/>'
    + small_coin(26, 54, 8) + small_coin(14, 82, 5.5, 1.5) + small_coin(40, 28, 4.5, 1.5))

poses["platypus-setup-carry-pipe"] = (
    STANDING + EYE_L + EYE_R
    + '<g transform="rotate(-20 100 145)">'
    + f'<rect x="22" y="134" width="156" height="22" rx="3" fill="{PIPE}"/>'
    + f'<rect x="22" y="138" width="156" height="4" fill="{PIPE_LIGHT}"/>'
    + f'<rect x="16" y="129" width="11" height="32" rx="2" fill="{PIPE_DARK}"/>'
    + f'<rect x="173" y="129" width="11" height="32" rx="2" fill="{PIPE_DARK}"/>'
    + '</g>' + arm(74, 154, 9, 11) + arm(126, 135, 9, 11))

poses["platypus-setup-wrench"] = (
    STANDING + EYE_L + EYE_R + arm(40, 134, 7, 11, 25)
    + '<g transform="rotate(30 166 66)">'
    + f'<rect x="160" y="44" width="12" height="50" rx="5" fill="{PIPE_DARK}"/>'
    + f'<path d="M158 24 A12 12 0 1 0 174 24" stroke="{PIPE_DARK}" stroke-width="9" fill="none" stroke-linecap="round"/>'
    + '</g>' + arm(160, 88, 8, 12, 25))

poses["platypus-monitoring-magnifier"] = (
    STANDING + EYE_L
    + f'<circle cx="126" cy="104" r="11" fill="{INK}"/><circle cx="130" cy="100" r="3.5" fill="#FFFFFF"/>'
    + f'<circle cx="126" cy="104" r="21" fill="#FFFFFF" fill-opacity="0.3" stroke="{INK}" stroke-width="5"/>'
    + f'<line x1="141" y1="119" x2="160" y2="142" stroke="{INK}" stroke-width="8" stroke-linecap="round"/>'
    + arm(160, 146))

def sparkle(cx, cy, s):
    """Four-pointed gold sparkle of half-size s."""
    k = s * 0.3
    return (f'<path d="M{cx} {cy - s} L{cx + k} {cy - k} L{cx + s} {cy} L{cx + k} {cy + k} '
            f'L{cx} {cy + s} L{cx - k} {cy + k} L{cx - s} {cy} L{cx - k} {cy - k} Z" fill="{GOLD}"/>')


# Trophy with a star, held by both handles in front of the belly. For model selection or winners.
poses["platypus-trophy"] = (
    STANDING + EYE_L + EYE_R
    + '<g transform="translate(0 11)">'  # keeps the rim clear of the bill
    + f'<path d="M80 131 Q63 129 65 142 Q67 152 81 149" stroke="{GOLD_DARK}" stroke-width="5" fill="none" stroke-linecap="round"/>'
    + f'<path d="M120 131 Q137 129 135 142 Q133 152 119 149" stroke="{GOLD_DARK}" stroke-width="5" fill="none" stroke-linecap="round"/>'
    + f'<rect x="95" y="154" width="10" height="12" fill="{GOLD_DARK}"/>'
    + f'<rect x="83" y="164" width="34" height="10" rx="2.5" fill="{GOLD}"/>'
    + f'<rect x="90" y="167" width="20" height="4" rx="1.5" fill="{GOLD_LIGHT}"/>'
    + f'<path d="M78 125 H122 Q122 153 100 158 Q78 153 78 125 Z" fill="{GOLD}"/>'
    + f'<rect x="75" y="120" width="50" height="8" rx="3" fill="{GOLD_DARK}"/>'
    + f'<path d="M85 132 Q85 144 91 151" stroke="{GOLD_LIGHT}" stroke-width="3" fill="none" stroke-linecap="round"/>'
    + '<polygon points="100,130 102,135.3 107.6,135.5 103.2,139 104.7,144.5 100,141.4 '
      '95.3,144.5 96.8,139 92.4,135.5 98,135.3" fill="#FFF4D6"/>'
    + '</g>'
    + arm(66, 153, 8, 10) + arm(134, 153, 8, 10)
    + sparkle(36, 76, 8) + sparkle(168, 62, 7) + sparkle(152, 36, 4.5))

def question_mark(x, y, scale=1.0, rot=0):
    """Hand-drawn question mark whose top-left sits near (x, y)."""
    return (f'<g transform="translate({x} {y}) rotate({rot}) scale({scale})">'
            f'<path d="M0 12 Q0 0 12 0 Q24 0 24 12 Q24 20 14 24 Q10 26 10 34" '
            f'stroke="{INK}" stroke-width="6" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
            f'<circle cx="10" cy="45" r="4" fill="{INK}"/></g>')


# Confused: one paw scratching the side of the head, one brow raised, a sweat drop and two
# question marks. For open questions, limitations or "what went wrong" slides.
poses["platypus-confused"] = (
    STANDING + EYE_L + EYE_R
    + f'<path d="M66 86 Q74 80 83 84" stroke="{INK}" stroke-width="3.5" fill="none" stroke-linecap="round"/>'
    + f'<path d="M117 87 L133 86" stroke="{INK}" stroke-width="3.5" stroke-linecap="round"/>'
    + '<path d="M152 70 Q159 81 152 86 Q145 81 152 70 Z" fill="#CFE6F3"/>'
    + arm(40, 80, 7, 12, 20) + arm(162, 126, 7, 11, -25)
    + question_mark(150, 6, 1.0, 12) + question_mark(124, 18, 0.6, -14))

# Round glasses sit over the normal eyes: clear lenses (a tint turns the eyes grey),
# a glint on each lens, a curved bridge over the bill and short arms back to the head edge.
GLASSES = (
    f'<path d="M62 102 L46 98 M138 102 L154 98" stroke="{INK}" stroke-width="4" stroke-linecap="round"/>'
    f'<circle cx="75" cy="105" r="14" fill="none" stroke="{INK}" stroke-width="4"/>'
    f'<circle cx="125" cy="105" r="14" fill="none" stroke="{INK}" stroke-width="4"/>'
    f'<path d="M89 103 Q100 95 111 103" stroke="{INK}" stroke-width="4" fill="none" stroke-linecap="round"/>'
    '<path d="M66 99 Q69 95 73 94 M116 99 Q119 95 123 94" stroke="#FFFFFF" stroke-width="2.5" '
    'fill="none" stroke-linecap="round" stroke-opacity="0.8"/>')

# In the magnifier pose the lens covers one eye, so the glasses are pushed up onto the forehead.
GLASSES_UP = (
    f'<path d="M68 73 L54 78 M132 73 L146 78" stroke="{INK}" stroke-width="4" stroke-linecap="round"/>'
    f'<circle cx="80" cy="70" r="12" fill="none" stroke="{INK}" stroke-width="4"/>'
    f'<circle cx="120" cy="70" r="12" fill="none" stroke="{INK}" stroke-width="4"/>'
    f'<path d="M92 69 Q100 62 108 69" stroke="{INK}" stroke-width="4" fill="none" stroke-linecap="round"/>'
    '<path d="M73 66 Q75 62 79 61 M113 66 Q115 62 119 61" stroke="#FFFFFF" stroke-width="2.5" '
    'fill="none" stroke-linecap="round" stroke-opacity="0.8"/>')

# Footer icon: same details as the full poses, drawn a little chunkier so they survive at 20-30px.
FOOTER_EYES = (f'<circle cx="75" cy="104" r="10" fill="{INK}"/><circle cx="79" cy="100" r="3.6" fill="#FFFFFF"/>'
               f'<circle cx="125" cy="104" r="10" fill="{INK}"/><circle cx="129" cy="100" r="3.6" fill="#FFFFFF"/>')
FOOTER_GLASSES = (
    f'<path d="M60 101 L44 96 M140 101 L156 96" stroke="{INK}" stroke-width="5" stroke-linecap="round"/>'
    f'<circle cx="75" cy="104" r="16" fill="none" stroke="{INK}" stroke-width="5"/>'
    f'<circle cx="125" cy="104" r="16" fill="none" stroke="{INK}" stroke-width="5"/>'
    f'<path d="M91 102 Q100 93 109 102" stroke="{INK}" stroke-width="5" fill="none" stroke-linecap="round"/>')

poses["platypus-footer-icon"] = (
    TAIL + BODY + BELLY + FEET
    + f'<ellipse cx="60" cy="122" rx="10" ry="6" fill="{BLUSH}"/>'
    + f'<ellipse cx="140" cy="122" rx="10" ry="6" fill="{BLUSH}"/>'
    + FOOTER_EYES
    + f'<rect x="82" y="112" width="36" height="18" rx="9" fill="{INK}"/>'
    + f'<path d="M58 72 Q72 61 88 68" stroke="{SHINE}" stroke-width="6" fill="none" stroke-linecap="round"/>'
    + arm(38, 132, 8, 12, 25) + arm(162, 126, 8, 12, -25))

# Every pose also gets a "-glasses" twin saved as a separate file; the originals are untouched.
# The glasses go in straight after the eyes, so arms and props still sit on top of them.
glasses_spots = {
    "platypus-results-cheer": (EYES_HAPPY, GLASSES),
    "platypus-monitoring-magnifier": (STANDING, GLASSES_UP),
    "platypus-footer-icon": (FOOTER_EYES, FOOTER_GLASSES),
}
for name, body in list(poses.items()):
    anchor, glasses = glasses_spots.get(name, (EYE_L + EYE_R, GLASSES))
    assert anchor in body, f"no place to put glasses on {name}"
    poses[f"{name}-glasses"] = body.replace(anchor, anchor + glasses, 1)

# The footer icons are cropped tight so they line up with footer text.
view_boxes = {name: (26, 44, 168, 142) for name in poses if name.startswith("platypus-footer-icon")}

def folder(name):
    """Glasses ("specs") versions live in their own subfolder."""
    return out / "specs" if name.endswith("-glasses") else out


for name, body in poses.items():
    x, y, w, h = view_boxes.get(name, (0, 0, 200, 215))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x} {y} {w} {h}" width="{w * 2}" height="{h * 2}">'
           f'<title>{name}</title>{body}</svg>\n')
    folder(name).mkdir(exist_ok=True)
    (folder(name) / f"{name}.svg").write_text(svg, encoding="utf-8")
    print(f"wrote {folder(name).name}/{name}.svg")


def find_browser():
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    for name in ("google-chrome", "chromium", "chrome", "msedge"):
        if found := shutil.which(name):
            candidates.insert(0, found)
    return next((c for c in candidates if Path(c).exists()), None)


def render_pngs():
    browser = find_browser()
    if browser is None:
        print("No Chrome or Edge found, so PNGs were not rendered.")
        return
    with tempfile.TemporaryDirectory() as tmp:
        for name in poses:
            _, _, w, h = view_boxes.get(name, (0, 0, 200, 215))
            width, height = 800, round(800 * h / w)
            page = Path(tmp) / f"{name}.html"
            page.write_text(
                "<!doctype html><html><head><style>html,body{margin:0;background:transparent}"
                f"img{{display:block;width:{width}px;height:{height}px}}</style></head>"
                f"<body><img src='{(folder(name) / f'{name}.svg').as_uri()}'></body></html>",
                encoding="utf-8")
            # A separate profile stops headless Chrome handing off to an open browser window.
            subprocess.run([
                browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                f"--user-data-dir={Path(tmp) / 'profile'}",
                "--default-background-color=00000000",
                f"--window-size={width},{height}",
                f"--screenshot={folder(name) / f'{name}.png'}",
                page.as_uri(),
            ], check=True, capture_output=True)
            print(f"wrote {folder(name).name}/{name}.png")


if "--no-png" not in sys.argv:
    render_pngs()
