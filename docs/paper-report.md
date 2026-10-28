# Experimental report: seven chapters

Industrial motivation tables are intended for the main paper; matching Office motivation tables are intended for the appendix. Main, ablation and motivation results use TEST. Office sensitivity uses VALID.

Seed42. Tail items have1..6 TRAIN next-target events; head items have at least12. Overall includes middle and zero-frequency items. R is item Recall (%); N is NDCG (raw value). EHC is effective hit coverage: distinct correctly retrieved group targets divided by distinct group targets observed in the evaluation split. EC is exposure coverage: distinct group items shown across all queries divided by catalogue group size. Both coverage values are percentages.

The main method uses parallel6x16 SID, beam20, lambda=.5, beta=.4, mask<=3. Comparison methods use beam200. Best values are bold and second-best use combining underlines in main tables; ablation tables bold the best only. Ties follow displayed precision.

## Chapter 1. Motivation analyses

### Industrial: main paper

#### 1. Does offline construction affect tail retrieval?

Compare serial residual quantization and parallel subspace quantization at a fixed configuration using raw-GR exact matching, beam20. The nominal spaces are equal:256^3=16^6=2^24.

| SID | @50 tail R/N | @100 tail R/N | @200 tail R/N |
| --- | --- | --- | --- |
| serial3x256 | 6.30 / 0.02240 | 7.91 / 0.02497 | 8.62 / 0.02602 |
| parallel3x256 | 8.20 / 0.02751 | 11.30 / 0.03251 | 13.02 / 0.03500 |
| serial6x16 | 5.29 / 0.01666 | 6.78 / 0.01917 | 6.78 / 0.01917 |
| parallel6x16 | 6.36 / 0.02332 | 8.62 / 0.02700 | 9.75 / 0.02865 |

| SID | @50 tail EHC/EC (%) | @100 tail EHC/EC (%) | @200 tail EHC/EC (%) |
| --- | --- | --- | --- |
| serial3x256 | 5.73 / 41.02 | 6.76 / 42.53 | 7.45 / 42.68 |
| parallel3x256 | 7.22 / 46.72 | 9.62 / 51.72 | 10.54 / 52.77 |
| serial6x16 | 3.89 / 29.67 | 4.70 / 30.93 | 4.70 / 30.93 |
| parallel6x16 | 6.76 / 35.52 | 8.59 / 37.39 | 9.16 / 37.79 |

Parallel construction improves tail R@100/200 and effective hit coverage within both configurations on these datasets.

#### 2. Is usable local information retained after exact SID failure?

Compare the target to each complete generated beam and take the minimum Hamming distance. Matching coordinates are evaluated within the same complete beam. Reachability is measured before ranking or truncation.

| Maximum mismatches | Head cumulative reach (%) | Tail cumulative reach (%) |
| --- | --- | --- |
| 0 | 41.32 | 9.75 |
| 1 | 44.97 | 14.80 |
| 2 | 48.20 | 22.89 |
| 3 | 59.77 | 41.56 |

| Configuration | @50 tail R/N | @100 tail R/N | @200 tail R/N |
| --- | --- | --- | --- |
| Parallel3x256 exact | 8.20 / 0.02751 | 11.30 / 0.03251 | 13.02 / 0.03500 |
| Parallel6x16 mask0..0 | 6.36 / 0.02332 | 8.62 / 0.02700 | 9.75 / 0.02865 |
| Parallel6x16 mask0..1 | 7.07 / 0.02464 | 11.00 / 0.03103 | 14.80 / 0.03638 |
| Parallel6x16 mask0..2 | 7.07 / 0.02464 | 12.90 / 0.03403 | 19.56 / 0.04335 |
| Parallel6x16 mask0..3 | 7.07 / 0.02464 | 12.90 / 0.03403 | 20.39 / 0.04447 |

| Configuration | @50 tail EHC/EC (%) | @100 tail EHC/EC (%) | @200 tail EHC/EC (%) |
| --- | --- | --- | --- |
| Parallel3x256 exact | 7.22 / 46.72 | 9.62 / 51.72 | 10.54 / 52.77 |
| Parallel6x16 mask0..0 | 6.76 / 35.52 | 8.59 / 37.39 | 9.16 / 37.79 |
| Parallel6x16 mask0..1 | 7.79 / 46.17 | 12.03 / 52.22 | 13.75 / 53.28 |
| Parallel6x16 mask0..2 | 7.79 / 48.89 | 15.23 / 75.03 | 20.96 / 80.78 |
| Parallel6x16 mask0..3 | 7.79 / 48.89 | 15.23 / 75.68 | 22.45 / 96.47 |

Incomplete long-tail SIDs retain local agreement that can recover item recall and effective coverage through partial matching.

#### 3. When do exact prefixes and legal tolerant completions disappear?

Exact survival asks whether the true prefix is still in the free beam. Tolerant survival asks whether any retained prefix can complete to a legal catalogue SID within3 target mismatches.

| Group | Quantity | Depth1 (%) | Depth2 (%) | Depth3 (%) | Depth4 (%) | Depth5 (%) | Depth6 (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| head | Exact target prefix | 100.00 | 60.34 | 49.61 | 45.34 | 42.52 | 41.32 |
| head | Legal radius3 completion | 100.00 | 91.40 | 74.41 | 65.19 | 61.23 | 59.77 |
| tail | Exact target prefix | 100.00 | 37.51 | 23.19 | 15.87 | 10.76 | 9.75 |
| tail | Legal radius3 completion | 100.00 | 90.25 | 64.57 | 49.64 | 43.52 | 41.56 |

Many tail targets lose every legal tolerant completion before decoding ends. A post-beam mask cannot recover evidence outside the retained neighborhood; this motivates protecting relevant tail prefixes during beam search as well.

### Office: appendix

#### 1. Does offline construction affect tail retrieval?

Compare serial residual quantization and parallel subspace quantization at a fixed configuration using raw-GR exact matching, beam20. The nominal spaces are equal:256^3=16^6=2^24.

| SID | @50 tail R/N | @100 tail R/N | @200 tail R/N |
| --- | --- | --- | --- |
| serial3x256 | 4.76 / 0.01654 | 4.76 / 0.01654 | 4.76 / 0.01654 |
| parallel3x256 | 4.98 / 0.01534 | 5.69 / 0.01651 | 5.69 / 0.01651 |
| serial6x16 | 2.84 / 0.00914 | 2.84 / 0.00914 | 2.84 / 0.00914 |
| parallel6x16 | 5.76 / 0.01636 | 6.18 / 0.01707 | 6.18 / 0.01707 |

| SID | @50 tail EHC/EC (%) | @100 tail EHC/EC (%) | @200 tail EHC/EC (%) |
| --- | --- | --- | --- |
| serial3x256 | 5.73 / 47.79 | 5.73 / 47.79 | 5.73 / 47.79 |
| parallel3x256 | 6.73 / 49.87 | 7.50 / 54.93 | 7.50 / 54.93 |
| serial6x16 | 3.86 / 36.53 | 3.86 / 36.53 | 3.86 / 36.53 |
| parallel6x16 | 6.84 / 44.96 | 7.39 / 47.84 | 7.39 / 47.84 |

Parallel construction improves tail R@100/200 and effective hit coverage within both configurations on these datasets.

#### 2. Is usable local information retained after exact SID failure?

Compare the target to each complete generated beam and take the minimum Hamming distance. Matching coordinates are evaluated within the same complete beam. Reachability is measured before ranking or truncation.

| Maximum mismatches | Head cumulative reach (%) | Tail cumulative reach (%) |
| --- | --- | --- |
| 0 | 32.58 | 6.18 |
| 1 | 35.35 | 10.52 |
| 2 | 41.01 | 19.40 |
| 3 | 55.10 | 39.80 |

| Configuration | @50 tail R/N | @100 tail R/N | @200 tail R/N |
| --- | --- | --- | --- |
| Parallel3x256 exact | 4.98 / 0.01534 | 5.69 / 0.01651 | 5.69 / 0.01651 |
| Parallel6x16 mask0..0 | 5.76 / 0.01636 | 6.18 / 0.01707 | 6.18 / 0.01707 |
| Parallel6x16 mask0..1 | 5.97 / 0.01674 | 9.67 / 0.02270 | 10.52 / 0.02395 |
| Parallel6x16 mask0..2 | 5.97 / 0.01674 | 10.73 / 0.02439 | 17.34 / 0.03368 |
| Parallel6x16 mask0..3 | 5.97 / 0.01674 | 10.73 / 0.02439 | 17.63 / 0.03406 |

| Configuration | @50 tail EHC/EC (%) | @100 tail EHC/EC (%) | @200 tail EHC/EC (%) |
| --- | --- | --- | --- |
| Parallel3x256 exact | 6.73 / 49.87 | 7.50 / 54.93 | 7.50 / 54.93 |
| Parallel6x16 mask0..0 | 6.84 / 44.96 | 7.39 / 47.84 | 7.39 / 47.84 |
| Parallel6x16 mask0..1 | 7.06 / 52.37 | 11.14 / 60.59 | 11.80 / 61.55 |
| Parallel6x16 mask0..2 | 7.06 / 53.39 | 12.68 / 77.39 | 20.62 / 86.61 |
| Parallel6x16 mask0..3 | 7.06 / 53.39 | 12.68 / 77.81 | 21.06 / 96.59 |

Incomplete long-tail SIDs retain local agreement that can recover item recall and effective coverage through partial matching.

#### 3. When do exact prefixes and legal tolerant completions disappear?

Exact survival asks whether the true prefix is still in the free beam. Tolerant survival asks whether any retained prefix can complete to a legal catalogue SID within3 target mismatches.

| Group | Quantity | Depth1 (%) | Depth2 (%) | Depth3 (%) | Depth4 (%) | Depth5 (%) | Depth6 (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| head | Exact target prefix | 100.00 | 51.87 | 37.47 | 34.54 | 33.35 | 32.58 |
| head | Legal radius3 completion | 100.00 | 93.38 | 72.28 | 60.22 | 56.80 | 55.10 |
| tail | Exact target prefix | 100.00 | 34.19 | 14.93 | 8.60 | 6.97 | 6.18 |
| tail | Legal radius3 completion | 100.00 | 87.21 | 60.20 | 46.13 | 41.15 | 39.80 |

Many tail targets lose every legal tolerant completion before decoding ends. A post-beam mask cannot recover evidence outside the retained neighborhood; this motivates protecting relevant tail prefixes during beam search as well.

## Chapter 2. Main comparison

### industrial

Main paper: deep-K retrieval and coverage@200.

| Group | Method | @50 R/N | @100 R/N | @200 R/N | EHC@200 (%) | EC@200 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | RQ-VAE | 18.20 / 0.08812 | 25.79 / 0.10043 | 34.24 / 0.11225 | 25.16 | 85.08 |
| Overall | TIGER-style | 20.10 / **0.10799** | 25.79 / **0.11722** | 33.09 / **0.12741** | 24.93 | 87.30 |
| Overall | OneRec-style | **22.48** / 0̲.̲1̲0̲6̲1̲2̲ | **28.52** / 0̲.̲1̲1̲5̲9̲0̲ | **36.66** / 0̲.̲1̲2̲7̲2̲5̲ | 30.60 | 85.73 |
| Overall | CRAB | 19.72 / 0.10596 | 24.99 / 0.11445 | 32.03 / 0.12428 | 24.99 | 88.09 |
| Overall | Ghost | 19.26 / 0.10343 | 23.54 / 0.11035 | 31.63 / 0.12159 | 3̲2̲.̲0̲3̲ | 9̲4̲.̲9̲0̲ |
| Overall | SA2CRQ | 20.41 / 0.08817 | 27.18 / 0.09917 | 3̲5̲.̲5̲4̲ / 0.11083 | 24.76 | 50.05 |
| Overall | Ours | 2̲2̲.̲0̲2̲ / 0.10254 | 2̲7̲.̲4̲7̲ / 0.11132 | 33.77 / 0.12013 | **33.01** | **99.40** |
| Tail | RQ-VAE | 3.27 / 0.01153 | 5.05 / 0.01443 | 7.79 / 0.01821 | 6.99 | 74.52 |
| Tail | TIGER-style | 1.01 / 0.00278 | 1.96 / 0.00429 | 4.58 / 0.00792 | 5.50 | 78.71 |
| Tail | OneRec-style | 6.78 / 0.01940 | 9.81 / 0.02426 | 14.68 / 0.03103 | 14.43 | 75.93 |
| Tail | CRAB | 0.89 / 0.00233 | 1.84 / 0.00389 | 4.70 / 0.00783 | 6.30 | 80.58 |
| Tail | Ghost | 7̲.̲3̲1̲ / 0̲.̲0̲2̲1̲1̲1̲ | 9̲.̲9̲3̲ / 0̲.̲0̲2̲5̲3̲6̲ | 1̲6̲.̲1̲1̲ / 0̲.̲0̲3̲3̲9̲3̲ | 2̲1̲.̲4̲2̲ | 9̲2̲.̲8̲4̲ |
| Tail | SA2CRQ | 2.91 / 0.00924 | 4.82 / 0.01234 | 7.31 / 0.01578 | 6.07 | 29.97 |
| Tail | Ours | **9.22** / **0.02968** | **14.03** / **0.03749** | **21.22** / **0.04756** | **25.20** | **99.09** |

Appendix: shallow-K retrieval and coverage@100.

| Group | Method | @5 R/N | @10 R/N | @20 R/N | EHC@100 (%) | EC@100 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | RQ-VAE | 8.67 / 0.06618 | 10.54 / 0.07242 | 11.93 / 0.07594 | 17.71 | 75.37 |
| Overall | TIGER-style | 10.10 / **0.08405** | 12.40 / **0.09141** | 14.93 / **0.09781** | 18.62 | 77.56 |
| Overall | OneRec-style | **10.70** / 0.07791 | 1̲3̲.̲0̲8̲ / 0.08554 | **16.72** / 0.09470 | 22.64 | 77.40 |
| Overall | CRAB | 9.84 / 0̲.̲0̲8̲2̲2̲9̲ | 12.05 / 0̲.̲0̲8̲9̲2̲8̲ | 14.89 / 0̲.̲0̲9̲6̲4̲3̲ | 18.97 | 80.01 |
| Overall | Ghost | 9.62 / 0.07999 | 11.96 / 0.08751 | 14.82 / 0.09467 | 2̲3̲.̲2̲7̲ | 9̲2̲.̲7̲3̲ |
| Overall | SA2CRQ | 7.81 / 0.05992 | 9.66 / 0.06586 | 11.96 / 0.07157 | 18.40 | 37.36 |
| Overall | Ours | 1̲0̲.̲2̲6̲ / 0.07375 | **13.50** / 0.08444 | 1̲6̲.̲0̲2̲ / 0.09083 | **26.02** | **93.35** |
| Tail | RQ-VAE | 1.13 / 0̲.̲0̲0̲6̲6̲9̲ | 1.49 / 0.00787 | 1.72 / 0.00844 | 4.47 | 57.27 |
| Tail | TIGER-style | 0.12 / 0.00085 | 0.18 / 0.00106 | 0.42 / 0.00166 | 3.21 | 61.50 |
| Tail | OneRec-style | 1̲.̲1̲9̲ / 0.00598 | 2.20 / 0.00918 | 4.34 / 0.01461 | 10.08 | 62.06 |
| Tail | CRAB | 0.12 / 0.00063 | 0.24 / 0.00100 | 0.36 / 0.00128 | 2.98 | 66.20 |
| Tail | Ghost | 1.13 / 0.00616 | 2̲.̲5̲0̲ / 0̲.̲0̲1̲0̲5̲0̲ | 4̲.̲4̲6̲ / 0̲.̲0̲1̲5̲3̲6̲ | 1̲3̲.̲8̲6̲ | 8̲8̲.̲9̲5̲ |
| Tail | SA2CRQ | 0.65 / 0.00404 | 1.01 / 0.00520 | 1.49 / 0.00645 | 4.24 | 18.31 |
| Tail | Ours | **2.38** / **0.01420** | **3.45** / **0.01760** | **5.11** / **0.02184** | **16.95** | **90.21** |

### office

Main paper: deep-K retrieval and coverage@200.

| Group | Method | @50 R/N | @100 R/N | @200 R/N | EHC@200 (%) | EC@200 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | RQ-VAE | 21.13 / 0.11744 | 26.96 / 0.12688 | 34.18 / 0.13697 | 27.64 | 89.48 |
| Overall | TIGER-style | 21.74 / **0.12327** | 27.15 / 0̲.̲1̲3̲2̲0̲0̲ | 34.03 / 0̲.̲1̲4̲1̲6̲1̲ | 28.20 | 91.21 |
| Overall | OneRec-style | **22.96** / 0̲.̲1̲2̲3̲1̲8̲ | **28.55** / **0.13223** | **36.42** / **0.14322** | 31.99 | 88.44 |
| Overall | CRAB | 20.65 / 0.11994 | 26.26 / 0.12896 | 32.41 / 0.13755 | 28.30 | 92.31 |
| Overall | Ghost | 18.17 / 0.11575 | 22.52 / 0.12279 | 30.44 / 0.13380 | 3̲4̲.̲5̲4̲ | 9̲6̲.̲6̲2̲ |
| Overall | SA2CRQ | 21.37 / 0.11092 | 2̲7̲.̲1̲7̲ / 0.12026 | 3̲5̲.̲1̲2̲ / 0.13132 | 23.95 | 45.33 |
| Overall | Ours | 2̲1̲.̲8̲0̲ / 0.11595 | 26.24 / 0.12319 | 31.52 / 0.13058 | **36.28** | **99.54** |
| Tail | RQ-VAE | 1.92 / 0.00636 | 2.63 / 0.00751 | 4.34 / 0.00989 | 6.17 | 81.81 |
| Tail | TIGER-style | 1.56 / 0.00646 | 2.20 / 0.00750 | 4.48 / 0.01061 | 5.84 | 84.85 |
| Tail | OneRec-style | 3.20 / 0.01547 | 5.12 / 0.01858 | 8.32 / 0.02299 | 10.25 | 79.73 |
| Tail | CRAB | 1.85 / 0.00749 | 2.70 / 0.00884 | 4.83 / 0.01180 | 6.28 | 86.99 |
| Tail | Ghost | 5̲.̲9̲0̲ / 0̲.̲0̲1̲9̲6̲3̲ | 8̲.̲6̲0̲ / 0̲.̲0̲2̲4̲0̲2̲ | 1̲3̲.̲5̲0̲ / 0̲.̲0̲3̲0̲8̲4̲ | 1̲7̲.̲7̲5̲ | 9̲4̲.̲8̲3̲ |
| Tail | SA2CRQ | 1.28 / 0.00394 | 2.06 / 0.00518 | 3.41 / 0.00706 | 4.19 | 26.88 |
| Tail | Ours | **8.10** / **0.02342** | **13.65** / **0.03246** | **20.54** / **0.04210** | **25.03** | **99.31** |

Appendix: shallow-K retrieval and coverage@100.

| Group | Method | @5 R/N | @10 R/N | @20 R/N | EHC@100 (%) | EC@100 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | RQ-VAE | 11.39 / 0.09419 | 13.44 / 0.10081 | 16.24 / 0.10782 | 21.60 | 82.10 |
| Overall | TIGER-style | 1̲2̲.̲0̲8̲ / 0̲.̲1̲0̲0̲5̲9̲ | 13.89 / **0.10637** | 16.44 / 0̲.̲1̲1̲2̲8̲2̲ | 21.95 | 83.93 |
| Overall | OneRec-style | **12.31** / 0.09772 | **14.39** / 0.10433 | **17.76** / **0.11290** | 24.41 | 81.01 |
| Overall | CRAB | 11.43 / 0.09800 | 13.40 / 0.10427 | 15.89 / 0.11055 | 22.52 | 85.08 |
| Overall | Ghost | 11.76 / **0.10064** | 13.01 / 0̲.̲1̲0̲4̲6̲6̲ | 14.67 / 0.10884 | 2̲5̲.̲3̲8̲ | 9̲4̲.̲6̲2̲ |
| Overall | SA2CRQ | 10.62 / 0.08538 | 12.68 / 0.09206 | 15.82 / 0.10002 | 18.42 | 35.62 |
| Overall | Ours | 11.04 / 0.08917 | 1̲3̲.̲9̲1̲ / 0.09853 | 1̲7̲.̲0̲0̲ / 0.10639 | **28.35** | **95.26** |
| Tail | RQ-VAE | 0.43 / 0.00276 | 0.78 / 0.00395 | 1.14 / 0.00485 | 3.75 | 68.53 |
| Tail | TIGER-style | 0.57 / 0.00405 | 0.85 / 0.00489 | 1.14 / 0.00561 | 2.76 | 71.57 |
| Tail | OneRec-style | **1.71** / **0.01207** | 2.06 / **0.01319** | 2.20 / 0.01354 | 6.28 | 66.77 |
| Tail | CRAB | 0.50 / 0.00419 | 0.92 / 0.00554 | 1.14 / 0.00611 | 3.53 | 74.13 |
| Tail | Ghost | 1̲.̲4̲9̲ / 0̲.̲0̲0̲9̲1̲1̲ | 2̲.̲3̲5̲ / 0.01184 | 3̲.̲7̲0̲ / 0̲.̲0̲1̲5̲2̲1̲ | 1̲1̲.̲5̲8̲ | 9̲1̲.̲1̲5̲ |
| Tail | SA2CRQ | 0.28 / 0.00162 | 0.50 / 0.00231 | 0.71 / 0.00282 | 2.43 | 17.23 |
| Tail | Ours | 1.28 / 0.00733 | **2.84** / 0̲.̲0̲1̲2̲2̲1̲ | **4.48** / **0.01639** | **16.98** | **92.37** |

## Chapter 3. Office component ablation

Keep one table in the paper, or convert it to a figure if space is limited. The first three removals reuse the frozen parallel model. Serial Full changes offline construction to serial6x16, uses its corresponding frozen GR and the same TRAIN-only linear projection recipe for8 epochs.

| Group | Method | @50 R/N | @100 R/N | @200 R/N | EHC@200 (%) | EC@200 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | Full | 21.80 / 0.11595 | **26.24** / 0.12319 | **31.52** / 0.13058 | **36.28** | 99.54 |
| Overall | Full without prior | 21.19 / 0.11346 | 26.08 / 0.12138 | 31.30 / 0.12866 | 34.19 | 98.64 |
| Overall | Full without semantics | 20.16 / 0.11107 | 24.97 / 0.11884 | 30.05 / 0.12593 | 34.54 | 99.77 |
| Overall | Full without mask | 20.80 / 0.11409 | 21.13 / 0.11465 | 21.13 / 0.11465 | 20.42 | 77.74 |
| Overall | Full with serial SID | **22.13** / **0.11976** | 25.05 / **0.12450** | 29.47 / **0.13065** | 33.83 | **99.86** |
| Tail | Full | **8.10** / **0.02342** | **13.65** / **0.03246** | **20.54** / **0.04210** | **25.03** | 99.31 |
| Tail | Full without prior | 6.47 / 0.01930 | 12.58 / 0.02914 | 18.76 / 0.03777 | 22.71 | 97.81 |
| Tail | Full without semantics | 7.46 / 0.02118 | 13.57 / 0.03105 | 19.26 / 0.03898 | 23.81 | 99.63 |
| Tail | Full without mask | 6.82 / 0.02106 | 6.97 / 0.02131 | 6.97 / 0.02131 | 8.60 | 63.73 |
| Tail | Full with serial SID | 7.68 / 0.01996 | 11.37 / 0.02594 | 16.99 / 0.03376 | 21.39 | **99.79** |

Parallel Full improves all tail Recall/NDCG cutoffs relative to serial Full; serial Full has stronger overall shallow retrieval and slightly higher overall NDCG.

## Chapter 4. Office parameter sensitivity

Use figures in the paper; these tables supply plotting data. All panels use VALID. Vary one parameter around(.5,.4,3), holding the other two fixed.

### TRAIN prior lambda: beta=.4, mask<=3

| Group | Method | @50 R/N | @100 R/N | @200 R/N | EHC@200 (%) | EC@200 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | 0.3 | 28.03 / 0.16066 | 32.00 / 0.16708 | 37.16 / 0.17426 | 38.58 | 99.48 |
| Overall | 0.4 | 28.07 / 0.16140 | 31.98 / 0.16771 | 37.14 / 0.17492 | 38.84 | 99.54 |
| Overall | 0.5 | 27.91 / 0.16061 | 32.00 / 0.16718 | 36.68 / 0.17372 | 39.06 | 99.68 |
| Overall | 0.6 | 27.81 / 0.16010 | 31.89 / 0.16670 | 36.60 / 0.17328 | 39.37 | 99.77 |
| Overall | 0.7 | 27.85 / 0.16064 | 31.73 / 0.16692 | 36.52 / 0.17362 | 39.53 | 99.94 |
| Tail | 0.3 | 7.92 / 0.02728 | 13.54 / 0.03635 | 19.79 / 0.04511 | 24.13 | 99.20 |
| Tail | 0.4 | 8.39 / 0.02803 | 13.38 / 0.03605 | 20.19 / 0.04557 | 24.48 | 99.31 |
| Tail | 0.5 | 8.87 / 0.02885 | 14.09 / 0.03721 | 20.35 / 0.04597 | 24.83 | 99.52 |
| Tail | 0.6 | 8.71 / 0.02926 | 13.94 / 0.03767 | 20.67 / 0.04712 | 25.17 | 99.68 |
| Tail | 0.7 | 9.11 / 0.03070 | 14.17 / 0.03887 | 20.43 / 0.04768 | 25.29 | 99.95 |

### Semantic beta: lambda=.5, mask<=3

| Group | Method | @50 R/N | @100 R/N | @200 R/N | EHC@200 (%) | EC@200 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | 0.2 | 27.60 / 0.15971 | 31.73 / 0.16636 | 36.87 / 0.17353 | 38.95 | 99.60 |
| Overall | 0.3 | 27.76 / 0.16031 | 31.85 / 0.16691 | 36.91 / 0.17398 | 39.06 | 99.65 |
| Overall | 0.4 | 27.91 / 0.16061 | 32.00 / 0.16718 | 36.68 / 0.17372 | 39.06 | 99.68 |
| Overall | 0.5 | 28.11 / 0.16084 | 32.08 / 0.16728 | 36.85 / 0.17395 | 39.32 | 99.68 |
| Overall | 0.6 | 28.46 / 0.16135 | 32.06 / 0.16718 | 36.68 / 0.17365 | 39.16 | 99.71 |
| Tail | 0.2 | 8.55 / 0.02828 | 13.94 / 0.03691 | 20.51 / 0.04611 | 25.06 | 99.41 |
| Tail | 0.3 | 8.63 / 0.02836 | 13.78 / 0.03660 | 20.27 / 0.04570 | 24.71 | 99.47 |
| Tail | 0.4 | 8.87 / 0.02885 | 14.09 / 0.03721 | 20.35 / 0.04597 | 24.83 | 99.52 |
| Tail | 0.5 | 8.95 / 0.02903 | 14.17 / 0.03744 | 20.59 / 0.04644 | 25.41 | 99.52 |
| Tail | 0.6 | 9.11 / 0.02931 | 14.09 / 0.03733 | 20.51 / 0.04636 | 25.29 | 99.52 |

### Maximum masks: lambda=.5, beta=.4

| Group | Method | @50 R/N | @100 R/N | @200 R/N | EHC@200 (%) | EC@200 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | 0 | 26.84 / 0.15863 | 27.33 / 0.15946 | 27.33 / 0.15946 | 24.11 | 77.83 |
| Overall | 1 | 27.66 / 0.16016 | 29.59 / 0.16332 | 29.68 / 0.16344 | 27.77 | 82.48 |
| Overall | 2 | 27.91 / 0.16061 | 31.96 / 0.16712 | 35.20 / 0.17169 | 36.72 | 94.77 |
| Overall | 3 | 27.91 / 0.16061 | 32.00 / 0.16718 | 36.68 / 0.17372 | 39.06 | 99.68 |
| Overall | 4 | 27.91 / 0.16061 | 32.00 / 0.16718 | 36.68 / 0.17372 | 39.06 | 99.68 |
| Tail | 0 | 7.13 / 0.02557 | 7.28 / 0.02585 | 7.28 / 0.02585 | 8.86 | 63.68 |
| Tail | 1 | 8.63 / 0.02841 | 10.77 / 0.03191 | 10.85 / 0.03203 | 13.17 | 71.36 |
| Tail | 2 | 8.87 / 0.02885 | 14.01 / 0.03709 | 18.45 / 0.04336 | 22.49 | 91.84 |
| Tail | 3 | 8.87 / 0.02885 | 14.09 / 0.03721 | 20.35 / 0.04597 | 24.83 | 99.52 |
| Tail | 4 | 8.87 / 0.02885 | 14.09 / 0.03721 | 20.35 / 0.04597 | 24.83 | 99.52 |

## Chapter 5. Relative inference efficiency

Within each dataset, normalize latency and throughput to Ours=1. Latency is batch1 time per request (lower is better); throughput is batch16 requests per second (higher is better). Historical CPU FP32 measurements use identical64 requests and3 serial rounds, including semantic work, matching and sorting.

| Method | Beam | Industrial latency | Industrial throughput | Office latency | Office throughput |
| --- | --- | --- | --- | --- | --- |
| RQ-VAE | 200 | 2.00 | 0.28 | 2.11 | 0.26 |
| TIGER-style | 200 | 2.85 | 0.17 | 3.06 | 0.17 |
| OneRec-style | 200 | 2.00 | 0.26 | 2.05 | 0.25 |
| CRAB | 200 | 2.87 | 0.18 | 2.98 | 0.17 |
| Ghost | 200 | 8.01 | 0.06 | 8.31 | 0.06 |
| SA2CRQ | 200 | 3.72 | 0.14 | 3.86 | 0.13 |
| Ours | 20 | 1.00 | 1.00 | 1.00 | 1.00 |

The table reports end-to-end CPU inference with each method using its configured SID length, vocabulary and beam budget.

## Chapter 6. Dataset popularity statistics

Appendix. Counts use TRAIN next-target events only. Middle and zero-frequency items remain in the catalogue totals.

| Dataset | Items | TRAIN events | Tail items (%) | Tail events (%) | Head items (%) | Head events (%) |
| --- | --- | --- | --- | --- | --- | --- |
| industrial | 3686 | 36259 | 53.77 | 22.44 | 20.56 | 56.46 |
| office | 3459 | 38924 | 54.21 | 20.20 | 22.67 | 62.54 |

Tail items exceed half the catalogue yet receive only about20..22% of TRAIN target events, while roughly one fifth of items receive more than half the events.

## Chapter 7. Baseline descriptions

Appendix; one sentence per method.

- **RQ-VAE:** three residual vector-quantization codes without an extra identity suffix; preserve aliases.
- **TIGER-style SID:** append a stable collision-disambiguation token to the same three residual codes.
- **OneRec-style SID:** construct residual k-means codes with balanced per-cluster item allocations.
- **CRAB:** split popular semantic parent tokens and adapt the shared retriever with structural regularization.
- **Ghost:** let tail items inherit head prefixes and add longer suffixes, together with tail unlikelihood training.
- **SA2CRQ:** combine adaptive code lengths with anchored head codebooks and tail extensions.

Baselines use their SID construction and training mechanisms on the shared ordinary GR backbone, with method-specific vocabularies and decoding lengths. Sources and implementation details are in README.md and NOTICE.
