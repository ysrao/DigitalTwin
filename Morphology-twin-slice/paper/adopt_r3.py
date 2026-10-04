"""Adopt Revision 3 as v3 with fixes:
1. Restore user's title.
2. Remove the [2] revision note from the manuscript -> notes file.
3. Extract Section IX + checklist table -> notes file.
4. Fill tables[1] (traffic/KPI definitions) from code; tables[2] SD column.
5. Resolve stale TBDs I can resolve (dup table values, kb/PRB units).
6. Add [DRAFT 3] label.
"""
from docx import Document
from docx.shared import Pt

SRC = "/home/hatch/workspace/user/files/RAN_Digital_Twin_Slice_Dimensioning___Revision_3.docx"
V3 = "/home/hatch/workspace/your_files/RAN_Digital_Twin_Slice_Dimensioning_v3.docx"
NOTES = "/home/hatch/workspace/urban-ran-twin/paper/revision_notes_r3.md"

doc = Document(SRC)
ps = doc.paragraphs

# 1. title
USER_TITLE = "A 6G RAN Digital Twin for Carrier Slice Dimensioning based on Morphology"
assert "RAN Digital Twin" in ps[0].text
ps[0].runs[0].text = USER_TITLE

# 2. extract the [2] revision note
rev_note = ps[2].text
assert rev_note.startswith("Revision note")
p2_el = ps[2]._p
p2_el.getparent().remove(p2_el)

# 3. extract Section IX + checklist table (tables[3])
notes = ["# Revision notes extracted from Revision 3 (not for the manuscript)",
         "", "## Front revision note", "", rev_note, ""]
# find IX heading
ix_idx = next(i for i, p in enumerate(doc.paragraphs)
              if p.style.name == "Heading 1" and "Revision Checklist" in p.text)
ix_texts = []
i = ix_idx
while i < len(doc.paragraphs):
    t = doc.paragraphs[i].text.strip()
    if t:
        ix_texts.append(("H1" if doc.paragraphs[i].style.name == "Heading 1" else "", t))
    i += 1
notes.append("## Section IX content")
for _, t in ix_texts:
    notes.append("- " + t[:200])
# checklist table -> markdown
t3 = doc.tables[3]
notes.append("")
notes.append("## Checklist table (from manuscript Table 4)")
hdr = [c.text.strip() for c in t3.rows[0].cells]
notes.append("| " + " | ".join(hdr) + " |")
notes.append("| " + " | ".join(["---"] * len(hdr)) + " |")
for r in t3.rows[1:]:
    notes.append("| " + " | ".join(c.text.strip()[:80] for c in r.cells) + " |")
open(NOTES, "w").write("\n".join(notes) + "\n")
# remove tables[3] element and IX section paragraphs (from end backwards)
t3._tbl.getparent().remove(t3._tbl)
for i in sorted(range(ix_idx, len(doc.paragraphs)), reverse=True):
    el = doc.paragraphs[i]._p
    el.getparent().remove(el)

# 4. fill tables[1] (slice/traffic/KPI definitions) from code
t1 = doc.tables[1]
fill = {
    "eMBB": ["Per-slot Bernoulli, mean inter-arrival 40 slots (20 ms) per UE",
             "Exponential, mean 12,000 bits",
             "20 ms deadline; 99% reliability target",
             "Packet dropped at deadline expiry"],
    "URLLC": ["Per-slot Bernoulli, mean inter-arrival 200 slots (100 ms) per UE",
              "Exponential, mean 256 bits",
              "5 ms deadline; 99.9% reliability target",
              "Packet dropped at deadline expiry"],
    "mIoT": ["PRACH-gated: occasions every 10 slots, p_access 0.3, 64 preambles",
             "800 bits fixed",
             "1000 ms deadline; 95% reliability target",
             "Packet dropped at deadline expiry"],
}
for r in t1.rows[1:]:
    sname = r.cells[0].text.strip()
    if sname in fill:
        for j, val in enumerate(fill[sname], start=1):
            r.cells[j].text = ""
            run = r.cells[j].paragraphs[0].add_run(val)
            run.font.size = Pt(9)
# tables[2] per-run SD column
t2 = doc.tables[2]
for i, sd in enumerate(["±170.8", "±152.6", "±148.6"], start=1):
    c = t2.rows[i].cells[5]
    c.text = ""
    run = c.paragraphs[0].add_run(sd)
    run.font.size = Pt(10)

# 5. resolve stale TBDs
resolutions = 0
for p in doc.paragraphs:
    if "second, unlabeled value set in each cell" in p.text:
        # stale: duplicate runs were cleaned in v2.5; earnings table is clean
        p.text = ("[Resolved in v3: the duplicated cell values were a document-"
                  "formatting artifact, cleaned in Draft 2.5. Table 3 is clean.]")
        resolutions += 1
print("stale TBD resolved:", resolutions)
# kb/PRB unit labels -> per slot
n_unit = 0
for p in doc.paragraphs:
    if "kb per PRB" in p.text and "per slot" not in p.text:
        p.text = p.text.replace("kb per PRB", "kb per PRB per 0.5-ms slot")
        n_unit += 1
print("unit labels fixed:", n_unit)

# 6. DRAFT 3 label after authors
for p in doc.paragraphs:
    if p.text.strip() == "Rao S. Yenamandra":
        lab = doc.add_paragraph("[DRAFT 3 — authorship and affiliations to be confirmed]")
        lab.alignment = 1
        p._p.addnext(lab._p)
        print("DRAFT 3 label added")
        break

doc.save(V3)
doc.save("/home/hatch/workspace/urban-ran-twin/paper/RAN_Digital_Twin_Slice_Dimensioning_v3.docx")
print("saved", V3)
