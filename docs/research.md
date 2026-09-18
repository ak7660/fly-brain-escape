# Research notes

Collected 2026-09-17 while building this project. Numbers under "Data facts" were measured on the local MaleCNS files; anything marked *unverified* was not.

## Scientific grounding
| Topic | Source | What we take from it |
|---|---|---|
| LPLC2 detects looming and drives the Giant Fiber escape | Ache et al. 2019, *Curr Biol*, "Neural basis for looming size and velocity encoding in the Drosophila Giant Fiber escape pathway" (PMID 30827912) | Input neurons = LPLC2 (looming size) + LC4 (looming velocity); output includes the Giant Fiber (`DNp01`) |
| LPLC2 is ultra-selective for looming | Klapoetke et al. 2017, *Nature*, "Ultra-selective looming detection from radial motion opponency" (PMC7457385) | Justifies driving LPLC2 with angular size of an approaching disc |
| Whole-brain LIF model from the connectome | Shiu et al. 2024, *Nature*, "A Drosophila computational brain model reveals sensorimotor processing" (PMC11446845) | Signs from predicted neurotransmitter (ACh +, GABA/Glu −); one global synaptic weight scale |
| Connectome-constrained trained networks | Lappalainen et al. 2024, *Nature*, "Connectome-constrained networks predict neural activity across the fly visual system" (PMC11525180) | Train a few per-cell-type parameters (pair scales, τ, resting potential) on top of fixed wiring |
| Whole-CNS male connectome | Male CNS connectome, *Cell* 2026 (doi:10.1016/j.cell.2026.08.015); data at https://male-cns.janelia.org/download/ | Dataset used here (CC-BY 4.0) |

## Prior-art projects
| Project | Approach | Lesson |
|---|---|---|
| [nfly](https://github.com/zhengxuyu/nfly) | MaleCNS visual subgraph (138k neurons) as a rate RNN, per-edge gains, PPO on games | Supervised readout worked; RL through the frozen connectome failed |
| [FLYCNS](https://github.com/IONOFIELD/FLYCNS) | Brian2 LIF of all 166k MaleCNS neurons, untrained | Needed a 0.3× weight rescale; network is silent without tonic input |
| [Fly Dino](https://github.com/cobanov/flyjump) | 80 frozen neurons → 16 DNs, readout trained in a browser worker | Small, readable circuit works for a live web demo |
| [Swat](https://github.com/hrook1/Swat) | LC16 → MDN retreat circuit, Three.js + worker | UI pattern: click a neuron to see its identity |
| [FlyPong](https://github.com/jonatasperaza/FlyPong) | LIF + dopamine STDP | Did not learn — avoid spiking + local plasticity |
| [train-your-fly](https://github.com/eudald-seeslab/train-your-fly) | Trainable connectome message passing | Random wiring did about as well → shuffled-wiring control is mandatory |
| Curated list | [awesome-fly](https://github.com/cobanov/awesome-fly) | Index of the above |

## Data facts (measured on local files)
- Neurons: `superclass.notna()` → 166,700; `status == 'Traced'` → 165,122.
- Inputs: `type == 'LPLC2'` 185 (side from `instance` suffix `_L`/`_R`), `type == 'LC4'` 126.
- Outputs: `superclass == 'descending_neuron'` 1,314; Giant Fiber `type == 'DNp01'` 2; `type == 'MDN'` 4; motor `superclass in {vnc_motor, cb_motor}` 815; jump muscle `TTMn`.
- `consensus_nt` values are full lowercase names (`acetylcholine`, `glutamate`, `gaba`, `histamine`, `dopamine`, `octopamine`, `serotonin`, `unclear`). Among traced neurons: acetylcholine 103,692 · glutamate 29,295 · gaba 22,051 · histamine 5,904 · unclear 2,957 · dopamine 392 · octopamine 101 · serotonin 48 · missing 166.
- Traced neurons (`superclass` not null and `status == 'Traced'`): 164,606.
- Traced-only weights: 25.56M edges (6.24M with weight ≥ 5); LPLC2 → GF = 4,862 synapses (185 LPLC2 → both GFs; same in traced-only, significant-only and full weights and in traced-only syn-partners; the earlier figure 4,836 counts only neuron pairs with ≥5 synapses, which is exactly what the circuit keeps). Within circuit v1 the largest Giant Fiber inputs are LC4 (6,362 synapses) then LPLC2 (4,836) — see docs/figures/p1_gf_inputs.png.
- Path subgraph LPLC2 → DN/MN at w ≥ 5: ≤3 hops ≈ 3.5k neurons / 100k edges; ≤4 hops ≈ 23k / 1.26M. (Early estimate; the ≤3-hop figure did not reproduce with the P1 selection rule, see the next line.)
- DNs reachable from LPLC2+LC4 within 3 hops (w ≥ 5, signed edges, no motor-neuron relays): 1,308 of 1,314. Keeping all of them left only ~2.4k hidden slots, so `DN_CAP` = 300 keeps the top DNs by forward flow (both GFs reserved). Top by flow: DNp01, DNp04, DNp103, DNg40, DNp06, DNp02, DNp11; the 4 MDNs rank ~267–291.
- Circuit v1 (`data/cache/circuit_v1.npz`): 4,296 neurons = 311 input, 498 hidden d=1, 2,886 hidden d≥2, 300 DN, 301 MN; 149,232 edges. All caps bind: DN (1,308 candidates), node (9,775 path neurons → 4,000), edge (253,993 induced → hidden→hidden edges need weight ≥ 15). 801 edges from sign-0 neurons dropped; 5 neurons removed as unconnected after pruning. 202 kept neurons have sign 0 (197 MNs, 5 DNs) and so no outgoing edges.
- Caveat: Giant Fiber → TTMn (jump muscle) is mainly an electrical synapse (gap junctions); chemical-synapse weights understate it (v1 has GF → TTMn weights of only 70 and 20). Don't read the jump pathway's strength from W0.
- Positions: `somaLocation` missing for sensory neurons → use synapse centroids (8 nm voxel units; stream `syn-partners-*-traced-only` with column projection, ~31 s).
- Brain shell: `gs://flyem-male-cns/rois/pointcloud-shells/JRCFIB2022M_brain.ply` (32,724 vertices, 66,360 faces, nm) and `JRCFIB2022M_vnc.ply` (18,021 vertices, 36,638 faces, nm).
- Centroids (`data/cache/centroids.npz`, 17 s, 3.2 GB peak RSS): 164,403 / 164,606 traced neurons have positions. The 203 without any traced-only synapse rows have NaN centroids (cb_sensory 112, ENS 42, vnc_sensory 26, ol_intrinsic 6, ol_sensory 5, vnc_sensory_tbc 5, sensory_ascending 4, visual_projection 1, cb_endocrine 1, vnc_intrinsic 1). **Exports must drop these neurons or give them a fallback position.**
- `syn-partners` has one row per pre→post pair, so pre-site centroids are weighted by partner count: a presynaptic site with k partners counts k times.
- Axes: voxel × 8 = shell nm with the same axis order. Among positioned neurons, ol_intrinsic + cb_intrinsic sit 99.99% inside the brain-shell bbox; vnc_intrinsic sits 100% inside the VNC-shell bbox and 0% inside the brain bbox. Every axis permutation fails at least one of these. The fly's LEFT is larger x (`LEFT_IS_POSITIVE_X = True`): LPLC2 and LC4 left/right post centroids don't overlap (gap ≥ 32,800 voxels).
- Pitfall: don't split sides at the pooled median x when L/R counts differ. LC4 has 71 L / 55 R, so a pooled-median split misclassified 8 left cells. Use the midpoint of the per-side medians.

## Visualization stack
- three.js 0.186.0 via jsdelivr importmap, no build step; `Points` + custom shaders for neurons, `LineSegments` with shader pulses for synapses, `UnrealBloomPass` at half resolution, device pixel ratio capped at 1.25 for Intel Iris Xe.
- The network runs live in JS (CSR sparse matrix-vector product, ~100k edges × 4 substeps per frame).
- *Unverified:* frame-rate budgets on Iris Xe are estimates, to be measured in P6.

## Calibration findings (P2, measured on circuit v1)
Rate model at τ = 20 ms, all type gains 1, probed with a 40 ms l/v loom from the left, front and right.

| gain / bias | DNs recruited (|Δr| > 0.01) L / F / R | notes |
|---|---|---|
| 1 / 0.02 | 5% / 10% / 5% | mean DN response 0.0012; passes gate v1 only via its 1e-3 floor |
| 2 / 0.02 | 16% / 24% / 14% | |
| 4 / 0.02 | 48% / 62% / 42% | stable, no saturation, front-selective, 91% tonically active |
| 8 / 0.02 | 70% / 72% / 68% | recruitment no longer direction-selective |
| ≥16 / >0 | — | unstable: 3–18% of neurons saturate or oscillate |

- The Giant Fiber prefers frontal looms at every stable gain (e.g. gain 8: peak 2.52 front vs 1.61 left / 1.28 right).
- **Gate revision (disclosed):** the pre-registered gate v1 (mean-DN response ≥ max(3·noise SD, 1e-3)) passed at gain 1, where almost no DNs respond, because DN noise is also tiny. Gate v2 keeps v1 and adds the stated purpose of calibration: tonic bias > 0 (a silent network cannot be released by disinhibition, as FLYCNS found) and ≥ 25% of DNs recruited in every direction. v1 and v2 results are both stored in `results/calibration.json`. Gain remains trainable; calibration only sets the starting point.
