"""Build the arXiv paper draft DOCX for the urban RAN digital twin.

Oriented to Part 2 (per user direction 2026-10-03): the carrier's
question — finding the correct slice structure from urban morphology,
coverage, capacity, and KPIs. Part 1 (predictive vs reactive
allocation) is reported as supporting theory.
"""
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()

style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(11)
for i in range(1, 4):
    hs = doc.styles[f"Heading {i}"]
    hs.font.name = "Times New Roman"
    hs.font.color.rgb = RGBColor(0, 0, 0)

def title(t):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(t)
    r.bold = True
    r.font.size = Pt(17)

def authors(t):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(t)
    r.font.size = Pt(12)

def abstract(t):
    p = doc.add_paragraph()
    r = p.add_run("Abstract — ")
    r.bold = True
    r.italic = True
    p.add_run(t)

def h1(t):
    doc.add_heading(t, level=1)

def h2(t):
    doc.add_heading(t, level=2)

def para(t):
    doc.add_paragraph(t)

def bullets(items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")

def fig(path, caption):
    doc.add_picture(path, width=Inches(6.0))
    p = doc.paragraphs[-1]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cp = doc.add_paragraph()
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cp.add_run(caption)
    r.italic = True
    r.font.size = Pt(10)

# ---------------- title ----------------
title("Finding the Correct Slice: An Urban RAN Digital Twin for Carrier "
      "Slice Dimensioning from Urban Morphology")
authors("Rao S. Yenamandra")
authors("[DRAFT — authorship and affiliations to be confirmed]")
doc.add_paragraph()

abstract(
    "A carrier's network is sunk cost: the towers stand where they stand "
    "and the spectrum is what was licensed. Slicing, by contrast, is "
    "software — and the carrier's real question is structural, not "
    "algorithmic: given the network I already own, what slice structure "
    "correctly matches my urban morphology, my coverage and capacity, and "
    "my KPI targets? We present an offline digital twin of an urban "
    "heterogeneous RAN built to answer exactly that. The twin fuses NVIDIA "
    "Sionna-based 3GPP TR 38.901 channel realizations over explicit "
    "macro/pico/femto geometry with a deterministic system simulator "
    "(mobility and handover, FDD+TDD carrier aggregation, per-slice queues, "
    "minimum-share PRB budgets with borrowing, PRACH). Carriers feed their "
    "real topology as a standard inputtable CSV of WGS84 cell coordinates; "
    "everything else — the dimensioning search, the KPI trade-off map — "
    "runs unchanged. On heavy-load urban morphologies the twin derives a "
    "concrete answer: the URLLC guaranteed share is the binding dimensioning "
    "decision (each avoided URLLC violation costs on the order of eight "
    "eMBB violations — six to twelve depending on the morphology), "
    "the eMBB floor is non-binding because reactive borrowing covers it, "
    "and mIoT needs only its 5% floor. A supporting study finds per-epoch "
    "predictive allocation (constrained MPC with a learned forecaster) "
    "ties reactive backlog-driven allocation (-0.16% served, +3.8% "
    "violations on matched realizations) — consistent with "
    "throughput-optimality theory, and evidence that the gains are in the "
    "slice structure, not the controller. Code, corpora, checkpoints, and "
    "per-claim reproduction pointers are released alongside this paper.")

# ---------------- I ----------------
h1("I. Introduction")
para(
    "Network slicing is usually studied as a control problem: given fixed "
    "slices, allocate spectrum among them as cleverly as possible. For a "
    "carrier this gets the question backwards. The carrier already owns "
    "the network — site positions, antenna heights, sector azimuths, "
    "licensed bands are sunk capex that will not move. What the carrier "
    "can still decide, in software, is the slice structure itself: how "
    "many slices, which services they carry, what fraction of spectrum "
    "each is guaranteed, on which layers. That is a slice dimensioning "
    "and optimization problem — finding the correct slice for a given "
    "urban morphology — and it is "
    "answered with coverage maps, capacity analysis, and KPI targets, not "
    "with a better scheduler.")
para(
    "This paper presents a digital twin built for that question: a "
    "controlled laboratory that takes a carrier's real topology, derives "
    "coverage and capacity from 3GPP channel models, and searches slice "
    "structures against KPI objectives on bit-identical realizations. "
    "Inside the same laboratory we also run the conventional control "
    "study — predictive versus reactive per-epoch allocation — and report "
    "it honestly as supporting theory, including a negative result: once "
    "the structure is right, the controller barely matters.")
para(
    "Our contributions are: (1) a deterministic urban HetNet twin fusing "
    "NVIDIA Sionna 3GPP channels with a full slicing stack, fed by "
    "standard, inputtable carrier topology files; (2) a slice-dimensioning "
    "methodology — morphology to coverage to capacity to KPI — with a "
    "concrete derived answer on heavy-load urban morphologies; (3) an "
    "action-conditioned KPI predictor with calibrated uncertainty, "
    "including a gray-box repair for learned action-ignoring and the "
    "justification for a small local model; and (4) a matched-realization "
    "comparison showing predictive per-epoch control ties reactive "
    "control, locating the gains in structure rather than scheduling.")

# ---------------- II ----------------
h1("II. Related Work")
para(
    "NVIDIA Sionna [1] — NVIDIA's open-source library for physical-layer "
    "research — provides GPU-accelerated 3GPP channel models and PHY "
    "components, but stops short of system-level slicing, mobility, and "
    "multi-cell coordination — the layers we build on top. O-RAN's RIC "
    "architecture [2] motivates data-driven RAN control, and several "
    "testbeds (notably OpenRAN Gym [3]) target over-the-air "
    "experimentation; our twin complements them as a fast, deterministic, "
    "offline counterpart. Throughput-optimality of backlog-based scheduling "
    "goes back to Tassiulas and Ephremides [4]; our empirical finding that "
    "a learned predictive controller rediscovers reactive-like allocations "
    "echoes that theory. Slice-dimensioning and optimization studies [5] "
    "typically assume stochastic-geometry topologies; we instead start "
    "from the carrier's actual site list and derive the structure "
    "per morphology. (This is dimensioning and optimization of slices on "
    "an existing network — not greenfield network planning.)")

# ---------------- III ----------------
h1("III. The Twin")
h2("A. Radio layer")
para(
    "Coupling loss between every user terminal (UT) and cell is computed "
    "with NVIDIA Sionna 2.2's 3GPP TR 38.901 Urban Macro model [6] over explicit "
    "HetNet geometry: macro sectors (with analytic 65-degree sector "
    "patterns), pico and femto small cells at configurable coordinates, "
    "heights, and azimuths. A critical methodological detail: Sionna's "
    "fading draws were unseeded in our first builds, which silently broke "
    "cross-policy comparisons — two 'identically seeded' runs saw different "
    "channels. We seed Sionna's generator per (master seed, call index) via "
    "sionna.phy.config.seed, making every fading realization bit-identical "
    "across repeat runs. Matched comparisons are meaningless without this.")
h2("B. System simulator")
para(
    "On top of the radio layer sits a slot-level simulator (0.5 ms slots, "
    "20-slot decision epochs): proportional-fair per-slice scheduling, "
    "per-UE queues with deadline expiry, mixed stationary/pedestrian/"
    "vehicular mobility with A3-style handover, FDD+TDD component carriers "
    "per cell (3GPP TR 38.104 FR1 PRB tables), a PRACH collision/backoff "
    "model for massive IoT access, and per-slice PRB budgets. Budgets "
    "enforce minimum shares; latency-critical slices (URLLC) hold their "
    "guarantee even when idle, while best-effort slices lend idle budget "
    "to a shared pool redistributed by backlog. Gate tests verify PRB "
    "accounting over all slot-cells and exact queue conservation "
    "(arrived = served + dropped + queued).")
h2("C. Carrier topology import")
para(
    "The twin runs on synthetic hexagonal topologies out of the box, but "
    "its purpose is the carrier's real network. A carrier feeds one "
    "standard, inputtable CSV: one row per cell (each macro sector its own "
    "row, sectors sharing a site_id), with WGS84 lat/lon coordinates — "
    "the carrier-standard — plus cell type, antenna height, sector azimuth, "
    "and optional per-cell transmit power. The scenario file points at the "
    "CSV and gives a nearby WGS84 reference point for projection to local "
    "meters. Files are validated before execution (required columns, "
    "unique cell IDs, coordinate ranges). In this carrier mode the "
    "synthetic generator is skipped entirely: every cell comes from the "
    "carrier's file, and the full dimensioning workflow runs unchanged. "
    "The format is documented in docs/input_formats.md with a worked "
    "example; further RAN configuration blocks (carriers, mobility, slices) "
    "are already separate documented sections of the scenario schema, so "
    "additional carrier inputs plug in the same way.")
h2("D. Scenario control")
para(
    "A validated scenario loader (YAML/JSON/CSV) configures topology, "
    "carriers, mobility mix, slice SLAs, and traffic. A web-based input "
    "artifact lets operators compose named scenarios and export them for "
    "batch execution. All randomness — traffic, mobility, fading — derives "
    "from a single master seed, so any experiment is exactly replayable.")

# ---------------- IV ----------------
h1("IV. The Predictor (Supporting Instrument)")
para(
    "Dimensioning search is simulation-based, but a forecaster both "
    "accelerates it as a surrogate and supports per-epoch studies. We "
    "train a gated recurrent unit (GRU) network — two layers of 64 units — "
    "on a 32-run corpus: each sample is one (cell, slice, decision time) "
    "with six epochs of KPI history plus the planned six-epoch budget "
    "vector, mapped to six-epoch forecasts of served traffic and SLA "
    "violations. Gaussian mean/log-variance heads give calibrated "
    "uncertainty (temperature scaling restores 89.6% coverage of nominal "
    "90% intervals on held-out layouts).")
para(
    "Two lessons are worth recording. First, the base model ignored its "
    "action input — shifting planned budgets by two standard deviations "
    "moved forecasts by 0.02 units — because reactive training budgets "
    "were a deterministic function of backlog already in the history. The "
    "repair is gray-box: served = min(demand, budget x spectral "
    "efficiency), with the GRU predicting demand and efficiency and the "
    "budget entering analytically. Second, the model is deliberately "
    "small and local: a forward pass takes ~15 ms on CPU, training takes "
    "three minutes, every run is bit-identical. The failure mode was "
    "structure, not capacity, and no larger or cloud-hosted model fixes "
    "action-ignoring better than putting the physics in the architecture. "
    "Small and structural beats large and generic for this task.")

# ---------------- V ----------------
h1("V. Deriving the Correct Slice")
para(
    "The dimensioning experiment fixes the morphology and the heavy-load "
    "regime — the conditions under which slicing decisions bind — and "
    "searches guaranteed-share structures: eMBB floors of 0.30/0.45/0.60 "
    "crossed with URLLC floors of 0.10/0.20/0.30 (mIoT at 0.05), reactive "
    "borrowing enabled, two urban layouts x five seeds (ten matched runs "
    "per structure). Each structure is scored on served traffic, SLA "
    "violations per slice, PRB utilization, and spectral efficiency.")
para(
    "The twin derives a concrete, morphology-specific answer (Figure 1). "
    "The URLLC guaranteed share is the binding dimensioning decision: "
    "raising it from 0.10 to 0.30 cuts URLLC violations from 2.3 to 1.0 "
    "per run but raises eMBB violations from ~67 to ~78 — each avoided "
    "URLLC violation costs on the order of eight eMBB violations (six to "
    "twelve across the two morphologies) — while served "
    "traffic falls slightly and spectral efficiency rises (URLLC's small "
    "packets use PRBs efficiently). The eMBB floor, by contrast, is "
    "non-binding: moving it from 0.30 to 0.60 shifts total violations "
    "from 71.2 to 68.0 per run, within noise, "
    "because reactive borrowing already routes idle budget to eMBB's "
    "backlog. mIoT never violates at its 5% floor. The correct slice for "
    "these morphologies is therefore: guarantee URLLC only what its SLA "
    "needs (0.10-0.20 here — the operator's chosen point on the "
    "trade-off), keep the eMBB floor low and let borrowing work, and hold "
    "mIoT at its floor.")
fig("paper/figs/dimensioning.png",
    "Figure 1. Slice dimensioning on heavy-load urban morphologies "
    "(2 layouts x 5 seeds, matched realizations). (a) The URLLC guaranteed "
    "share is the binding decision: it trades URLLC violations against "
    "eMBB violations. (b) Served traffic falls slightly while spectral "
    "efficiency rises with the URLLC guarantee.")
para(
    "This is the carrier's answer in miniature: not 'allocate smarter "
    "every 20 ms' but 'guarantee this much, borrow the rest.' The "
    "morphology enters through coverage and capacity — a sparser or "
    "denser deployment moves the trade-off point, which is exactly why "
    "the structure must be derived per network rather than assumed.")

# ---------------- VI ----------------
h1("VI. Allocation Dynamics Are Second-Order (Supporting Theory)")
para(
    "With the structure fixed, we ask the conventional question: does "
    "twin-assisted predictive allocation beat reactive allocation "
    "per epoch? The controller is constrained model-predictive control "
    "with receding horizon — the GRU predictor scores candidate budget "
    "vectors (reactive, pairwise tilts, bold 60%-to-one-slice "
    "reallocations, all-minimum, equal, demand-proportional) over six "
    "epochs by predicted served traffic minus violation and uncertainty "
    "penalties, subject to minimum-share and PRB-total constraints; the "
    "winner's first step is applied and the controller replans, falling "
    "back to reactive under high uncertainty.")
para(
    "An honest measurement needed two fixes beyond the predictor repair: "
    "an oracle rollout showed initial +-15% candidates moved true outcomes "
    "by under 1% (below the noise floor), so candidates were widened; and "
    "the Sionna determinism fix, without which 'matched' runs were not "
    "matched. Definitive result on 8 matched heavy-load runs: predictive "
    "serves -0.16% traffic with +3.8% violations versus reactive — a "
    "statistical tie, with the optimizer consistently selecting "
    "reactive-like allocations from a set containing bold alternatives. "
    "Under unpredictable 3x bursts both policies tie as well. Backlog-"
    "driven allocation every 20 ms is near throughput-optimal [4]; there "
    "is little 'future' for prediction to exploit. The experiment's value "
    "is the proof: the gains are in the slice structure (Section V), not "
    "the controller.")

# ---------------- VII ----------------
h1("VII. Limitations and Future Work")
bullets([
    "Five seeds per layout over two layouts is stronger than pilot-grade "
    "but still narrow; publication-grade claims need more morphologies "
    "with confidence intervals.",
    "Poisson arrivals carry no predictable temporal structure; diurnal or "
    "periodic demand would test proactive control more fairly.",
    "The dimensioning grid covers global guaranteed shares; per-layer "
    "structures (e.g., URLLC pinned to the macro coverage layer) are the "
    "natural next search dimension.",
    "All UEs are outdoor (a Sionna indoor edge case is documented and "
    "excluded); uplink and inter-cell coordination are future work.",
    "The predictor-as-surrogate for dimensioning search is implemented "
    "only as per-epoch control; closing that loop is next."])

# ---------------- VIII ----------------
h1("VIII. Conclusion")
para(
    "We built an urban RAN digital twin that answers the carrier's "
    "structural question: given the network you own, what is the correct "
    "slice? Fed with standard inputtable topology, it derives coverage, "
    "capacity, and KPI trade-offs per morphology — and on heavy-load "
    "urban layouts the answer is concrete: the URLLC guarantee binds, the "
    "eMBB floor does not, borrowing does the rest. Per-epoch predictive "
    "control, honestly measured, ties reactive control: structure beats "
    "scheduling. The twin is a laboratory for slicing decisions, and its "
    "product is the derived slice.")

h1("References")
refs = [
    "[1] J. Hoydis et al., \"Sionna: An open-source library for next-"
    "generation physical layer research,\" arXiv:2203.11854, 2022.",
    "[2] O-RAN Alliance, \"O-RAN architecture description,\" v14.0, 2024.",
    "[3] M. Polese et al., \"OpenRAN Gym: An open toolbox for data-driven "
    "RAN control,\" IEEE INFOCOM WKSHPS, 2022.",
    "[4] L. Tassiulas and A. Ephremides, \"Stability properties of "
    "constrained queueing systems and scheduling policies for maximum "
    "throughput in multihop radio networks,\" IEEE Trans. Autom. Control, "
    "1992.",
    "[5] Survey of RAN slice dimensioning and optimization "
    "(representative works to be completed).",
    "[6] 3GPP TR 38.901, \"Study on channel model for frequencies from 0.5 "
    "to 100 GHz,\" v18.0.0, 2024.",
]
for r in refs:
    p = doc.add_paragraph(r)
    p.paragraph_format.space_after = Pt(4)

h1("Appendix: Verification")
para(
    "Every empirical claim in this paper resolves to workspace code and "
    "data outside the publication: the simulator "
    "(src/urtwin/sim/baseline.py, channel.py, hetnet.py), carrier topology "
    "import (src/urtwin/scenarios/topology.py, docs/input_formats.md, "
    "examples/carrier_topology_example.csv), the corpora "
    "(data/corpus_v2-v4 with manifests), the predictors "
    "(src/urtwin/learning/models_structured.py, train_structured.py; "
    "checkpoints/predictor_v2.pt, predictor_v5.pt), the controllers and "
    "comparison harness (src/urtwin/control/mpc.py, compare.py, "
    "oracle_test.py), the comparison results "
    "(data/stage_d_comparison_v7.json), and the dimensioning study "
    "(src/urtwin/control/dimension.py, data/stage_e_dimensioning.json). "
    "Each step replays from a single master seed.")

out = "paper/Urban_RAN_Digital_Twin_Paper_DRAFT.docx"
doc.save(out)
print("saved", out)
