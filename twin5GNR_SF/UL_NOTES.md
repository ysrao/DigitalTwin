# Uplink (UL) results for the SF mesh twin — modelled

**Everything here is MODELLED, not measured.** The UL layer is computed from the same stored ray-traced path gains as the downlink (channel reciprocity), closed with the twin’s UL link-budget layer (the v3 link-budget method — see `../twin5GNR/UL_METHOD.md`): UE 23 dBm handset / 26 dBm rooftop CPE (23 dBm at 28 GHz), BS noise figure 3 dB + rise-over-thermal 4 dB, UE antenna from the link-budget workbook fit capped at +2.14 dBi, TDD DDDSU uplink fraction, single-user, unloaded.

| Band | Receiver | DL mean (Mbps) | UL mean (Mbps) | DL/UL ratio | UL served % | UL median, served (Mbps) |
|---|---|---|---|---|---|---|
| 0.7 GHz | Rooftop CPE | 127.3 | 35.3 | 3.60 | 97.4 | 40.7 |
| 0.7 GHz | Outdoor handset | 38.9 | 16.7 | 2.33 | 80.2 | 19.1 |
| 2.5 GHz | Rooftop CPE | 490.9 | 42.4 | 11.58 | 95.8 | 56.9 |
| 2.5 GHz | Outdoor handset | 213.2 | 16.0 | 13.32 | 79.5 | 15.5 |
| 3.6 GHz | Rooftop CPE | 314.6 | 27.1 | 11.59 | 95.5 | 36.2 |
| 3.6 GHz | Outdoor handset | 136.7 | 10.3 | 13.32 | 79.2 | 10.0 |
| 7.125 GHz | Rooftop CPE | 687.4 | 54.8 | 12.54 | 95.2 | 73.3 |
| 7.125 GHz | Outdoor handset | 212.4 | 15.0 | 14.17 | 76.0 | 7.2 |
| 28 GHz | Rooftop CPE | 691.2 | 84.8 | 8.15 | 94.1 | 117.2 |
| 28 GHz | Outdoor handset | 86.0 | 10.6 | 8.08 | 57.8 | 1.9 |

Machine-readable form: `UL_summary.json` in this folder (per band and receiver class, with the computation note embedded). UL maps were not rendered for the SF scene; the published maps remain the downlink RSRP and throughput sets.
