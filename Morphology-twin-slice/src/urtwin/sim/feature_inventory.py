"""Stage A: probe installed Sionna for built-in vs must-build capabilities.

Checks the features Brief v2 depends on and reports which exist in the
installed Sionna version. Run: .venv/bin/python src/urtwin/sim/feature_inventory.py
"""
import importlib
import sys

CHECKS = [
    # (label, import path or attribute probe)
    ("sionna.sys module (system-level simulation)", "sionna.sys"),
    ("sionna.rt module (ray tracing)", "sionna.rt"),
    ("3GPP 38.901 channel models (UMa/UMi)", "sionna.channel.tr38901.UMa"),
    ("Multicell topology generators", "sionna.sys"),  # refined below
    ("OFDM / NR frame handling", "sionna.ofdm"),
    ("LDPC / Polar FEC", "sionna.fec.ldpc"),
]

ATTR_CHECKS = [
    ("PHY abstraction in sys", "sionna.sys", "PhyAbstraction"),
    ("Link adaptation in sys", "sionna.sys", "LinkAdaptation"),
    ("Power control in sys", "sionna.sys", "PowerControl"),
]


def probe_module(path):
    try:
        m = importlib.import_module(path)
        ver = getattr(m, "__version__", "?")
        return True, f"present (sionna {ver})" if path == "sionna" else "present"
    except Exception as e:  # noqa: BLE001
        return False, f"missing ({type(e).__name__}: {e})"


def probe_attr(modpath, attr):
    try:
        m = importlib.import_module(modpath)
        ok = hasattr(m, attr)
        return ok, "present" if ok else "module present, attribute missing"
    except Exception as e:  # noqa: BLE001
        return False, f"module missing ({type(e).__name__})"


def main():
    try:
        import sionna  # noqa: F401
        print(f"sionna version: {importlib.import_module('sionna').__version__}")
    except Exception as e:  # noqa: BLE001
        print(f"FATAL: sionna not importable: {e}")
        sys.exit(2)

    print("\n=== Module checks ===")
    for label, path in CHECKS:
        ok, note = probe_module(path)
        print(f"[{'OK ' if ok else 'MISS'}] {label}: {note}")

    print("\n=== Attribute checks (sionna.sys capabilities) ===")
    for label, modpath, attr in ATTR_CHECKS:
        ok, note = probe_attr(modpath, attr)
        print(f"[{'OK ' if ok else 'MISS'}] {label}: {note}")

    print("\n=== Known must-build items (per Brief v2 §2) ===")
    for item in [
        "Packet queues per UE/slice",
        "Mixed mobility + handover behavior",
        "HetNet macro/small-cell association",
        "FDD+TDD carrier aggregation",
        "PRACH / mIoT access congestion",
        "Slice PRB budgets + borrowing",
        "Cluster coordination of budgets",
    ]:
        print(f"[BUILD] {item}: no built-in found; implement in src/urtwin/sim/")

    # List what's actually inside sionna.sys for the record
    try:
        import sionna.sys as s
        names = sorted(n for n in dir(s) if not n.startswith("_"))
        print(f"\n=== sionna.sys exports ({len(names)}) ===")
        print(", ".join(names[:40]))
    except Exception as e:  # noqa: BLE001
        print(f"\ncould not list sionna.sys: {e}")


if __name__ == "__main__":
    main()
