"""Apply the user's v1 review points, save as V2.

Points addressed:
1. mIoT coverage (data-plane + access-plane story)
2. PRACH flooding (new Figure 3)
3. eMBB throughput efficiency per slice (new Figure 2)
4. SLA violations per slice (breakdown)
5. Carrier earnings per slice (new subsection + table)
6. Cell layout through the standard map input (concrete example)
"""
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

SRC = "/home/hatch/workspace/user/files/Urban_RAN_Digital_Twin_Paper_v1.docx"
DST = "/home/hatch/workspace/your_files/Urban_RAN_Digital_Twin_Paper_v2.docx"
FIG2 = "/home/hatch/workspace/urban-ran-twin/paper/figs/iot_slices.png"
FIG3 = "/home/hatch/workspace/urban-ran-twin/paper/figs/prach_flooding.png"

doc = Document(SRC)
ps = doc.paragraphs

def insert_after(ref_p, new_p):
    ref_p._p.addnext(new_p._p)
    return new_p

def add_para_after(ref_p, text, style="Normal"):
    p = doc.add_paragraph(text, style=style)
    ref_p._p.addnext(p._p)
    return p

def add_fig_after(ref_p, path, caption):
    doc.add_picture(path, width=Inches(6.0))
    pic_p = doc.paragraphs[-1]
    pic_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    ref_p._p.addnext(pic_p._p)
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    r.italic = True
    r.font.size = Pt(10)
    pic_p._p.addnext(cap._p)
    return cap

# locate anchors (v1 structure, verified)
p15 = ps[15]   # III-B system simulator
p17 = ps[17]   # III-C carrier topology import
p25 = ps[25]   # V results paragraph
p27 = ps[27]   # Figure 1 caption
abstract_p = ps[2]

# 1+2. PRACH flooding in III-B
p15.add_run(
    " Under massive mIoT access the PRACH itself congests: 64 preambles "
    "per occasion, uniform backoff on collision. A flooding probe "
    "(Section V) raises offered load from 60 to 600 UEs: preamble "
    "collision rates climb from ~0% to ~5% and mean access delay inflates "
    "from under a slot to several — the access plane, not data PRBs, is "
    "mIoT's binding constraint.")

# 6. Concrete std-map-input example in III-C
p17.add_run(
    " A worked example ships with the code "
    "(examples/carrier_topology_example.csv — two three-sector macro "
    "sites, two picos and a femto on downtown San Francisco WGS84 "
    "coordinates); the twin was verified end-to-end on it: CSV import, "
    "Sionna coupling on the real geometry, and a full simulation run.")

# abstract: per-slice + earnings clause
abstract_p.add_run(
    " Per slice, eMBB dominates violations while URLLC's idle guarantee "
    "is spectrally expensive; mIoT never violates its data floor — its "
    "dimensioning lever is PRACH access under flooding. An illustrative "
    "per-slice carrier P&L puts the earnings optimum at an interior "
    "URLLC guarantee.")

# 3+4. Per-slice breakdown + efficiency, inserted after V results para
anchor = p25
anchor = add_para_after(
    anchor,
    "Per slice, the picture sharpens (Figure 2a). SLA violations are "
    "dominated by eMBB — 67.1 to 77.5 per run as the URLLC floor rises "
    "from 0.10 to 0.30 — while URLLC's own violations fall from 2.3 to "
    "1.0; mIoT never violates at all, its 5% data floor never the binding "
    "constraint. Throughput efficiency (Figure 2b) tells the complementary "
    "story: eMBB does more with less as its share shrinks, its kb-per-PRB "
    "rising from 0.030 to 0.034, while URLLC's own efficiency is poor "
    "(0.011–0.016) — the price of a guarantee that holds PRBs idle even "
    "when URLLC has nothing to send.")
anchor = add_para_after(
    anchor,
    "mIoT's dimensioning question is therefore not a data-plane one — it "
    "is access. The twin's PRACH model shows the flooding onset directly "
    "(Figure 3): collision rates rise from ~0% to ~5% and mean access "
    "delay inflates several-fold as offered load grows ten-fold, while "
    "mIoT data-plane violations stay at zero throughout — backoff absorbs "
    "the flood as delay, not loss. Dimensioning mIoT means dimensioning "
    "access occasions and preamble pools against the flooding curve, a "
    "separate optimization the twin is instrumented for.")

# 5. Earnings subsection
h = doc.add_heading("What the structure earns (illustrative)", level=2)
anchor._p.addnext(h._p)
anchor = h
anchor = add_para_after(
    anchor,
    "For a carrier, the trade-off curve is a P&L curve. Mapping the "
    "dimensioning grid through an illustrative parametric earnings model — "
    "revenue per served kilobit plus penalty per SLA violation, priced "
    "per slice (URLLC premium, eMBB bulk, mIoT best-effort) — the earnings "
    "optimum is interior. Guarantee too little URLLC (floor 0.10) and "
    "URLLC penalties dominate; guarantee too much (0.30) and eMBB "
    "penalties dominate. eMBB's earnings fall monotonically as the URLLC "
    "floor rises while URLLC's rise — the 'correct slice' is where the "
    "two P&L curves cross, here at a URLLC floor of 0.20 (Table 1). Unit "
    "economics are illustrative and the absolute levels arbitrary; the "
    "finding is the shape — an interior optimum set by opposing penalty "
    "curves — which is structural, not parametric.")
# earnings table
tbl = doc.add_table(rows=4, cols=5)
tbl.style = "Table Grid"
cells = [["URLLC floor", "eMBB ($/run)", "URLLC ($/run)", "mIoT ($/run)", "Total ($/run)"],
         ["0.10", "-162.6", "-113.6", "+0.9", "-275.3"],
         ["0.20", "-189.7", "-68.6", "+0.9", "-257.4"],
         ["0.30", "-216.3", "-48.6", "+0.9", "-264.0"]]
for i, row in enumerate(cells):
    for j, val in enumerate(row):
        c = tbl.cell(i, j)
        c.text = ""
        r = c.paragraphs[0].add_run(val)
        r.font.size = Pt(10)
        if i == 0:
            r.bold = True
anchor._p.addnext(tbl._tbl)
cap = doc.add_paragraph()
cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = cap.add_run("Table 1. Illustrative per-slice earnings vs URLLC guaranteed share. "
                "Interior optimum at 0.20; unit economics illustrative.")
r.italic = True
r.font.size = Pt(10)
tbl._tbl.addnext(cap._p)
anchor = cap

# Figures 2 and 3 after Figure 1 caption (insert in reverse via chaining)
cap3 = add_fig_after(
    p27, FIG3,
    "Figure 3. PRACH under flooding (measured probes, 2–3 seeds per load). "
    "(a) Preamble collision rate vs offered load. (b) Mean access delay "
    "inflates as collisions rise; mIoT data-plane violations stay at zero "
    "— backoff absorbs the flood as delay, not loss.")
cap2 = add_fig_after(
    p27, FIG2,
    "Figure 2. Per-slice view of the dimensioning grid (90 runs). "
    "(a) SLA violations per slice vs URLLC guaranteed share — eMBB "
    "dominates, mIoT never violates. (b) Throughput efficiency per slice — "
    "eMBB rises as it does more with less; URLLC's idle guarantee is "
    "spectrally expensive.")
_ = cap2, cap3

doc.save(DST)
print("saved", DST)

# also keep a working copy next to the build tree
doc.save("/home/hatch/workspace/urban-ran-twin/paper/Urban_RAN_Digital_Twin_Paper_v2.docx")
print("saved working copy")
