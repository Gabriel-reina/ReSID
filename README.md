# Reproduction README

This anonymous English release contains the prepared Industrial and Office inputs,
frozen weights, source code and numerical references. The companion document is the
[seven-chapter experimental report](docs/paper-report.md). Prepared embeddings and
checkpoints are included.

## Quick start and fixed protocol

Python 3.9+. From the extracted project root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -B -m sid_retrieval evaluate --verify --output runs/main
```

Seed42 throughout. Ours uses parallel6x16 SID, beam20, lambda=.5, beta=.4,
mask0..3 and a200-item budget. Six comparison methods use beam200. Serial/parallel
describes offline SID construction; the shared ordinary Transformer GR predicts
causally from scratch. Baselines use their SID construction and training mechanisms
on this shared GR backbone.

TRAIN target counts define tail1..6, head>=12, middle7..11 and zero0. Overall
includes all groups. Item Recall measures target hits, NDCG discounts target rank,
effective hit coverage counts distinct correctly retrieved target items over observed
targets, and exposure coverage counts shown items over the group catalogue.
Candidate reachability measures target membership in the expanded pool before ranking.


Run from the extracted project root. All output directories must be new. Every
experiment uses seed42. The primary CLI accepts --root before the command when
using code installed outside the artifact directory. Installation as a wheel
contains code only; pass the separate prepared-data root explicitly in that case.

## 1. Frozen main results and serial Full

```sh
python -B -m sid_retrieval evaluate --verify --output runs/main
python -B -m sid_retrieval evaluate --dataset office --method serial_full --verify --output runs/serial-full
python -B -m sid_retrieval evaluate --dataset office --method ghost --beam 20 --output runs/ghost-beam20
```

The last command evaluates Ghost at beam20. Ghost/SA2CRQ beam20 references are
in references/. Main tables use beam200 for baselines. Use --verify with the
complete frozen configuration; use --limit for a subset. The prediction API takes
history item IDs.

## 2. Training on supplied SID maps

```sh
python -B -m sid_retrieval.training --dataset industrial --method ours --output runs/industrial-gr
python -B -m sid_retrieval train --dataset office --method tiger --output runs/office-tiger
python -B -m sid_retrieval.training --dataset office --method rqvae --output runs/office-rqvae
python -B -m sid_retrieval.training --dataset office --method onerec --output runs/office-onerec
python -B -m sid_retrieval.training --dataset office --method ghost --output runs/office-ghost
python -B -m sid_retrieval.training --dataset office --method sa2crq --output runs/office-sa2crq
python -B -m sid_retrieval.training --dataset office --method crab --output runs/office-crab
python -B -m sid_retrieval.training --dataset industrial --method projection --artifact runs/industrial-gr --output runs/industrial-projection
```

GR training uses TRAIN, checkpoint selection uses VALID, and projection training uses TRAIN.
Ours preserves its global-generator DataLoader/LambdaLR convention. EOS-based baselines use
their explicit generator and manual warmup schedule. Ghost uses15 NLL +4 AUO epochs;
CRAB uses10 adapter epochs initialized from the frozen TIGER GR. Pass
--tiger-artifact to initialize CRAB from a newly trained TIGER instead.
Checkpoints use minimum VALID token NLL; projection training uses8 epochs.

For an independently trained baseline, evaluate with --artifact NEW_GR_DIRECTORY
and the matching --method/--dataset, without --verify. To assemble a new Ours artifact,
place its trained projection.pt and text_features.npz next to its new model.pt and
codes.json. Save each trained artifact in its own output directory.

Serial Full projection can be fitted with the identical recipe:

```sh
python -B -m sid_retrieval.training --dataset office --method projection --artifact artifacts/office/serial_full --output runs/serial-projection
```

## 3. Offline SID construction

```sh
python -m pip install -r requirements-tokenizers.txt
python -B -m sid_retrieval.tokenization --dataset industrial --method controlled --output runs/industrial-four-sids
python -B -m sid_retrieval.tokenization --dataset office --method tiger --output runs/tiger-sid
python -B -m sid_retrieval.tokenization --dataset office --method rqvae --output runs/rqvae-sid
python -B -m sid_retrieval.tokenization --dataset office --method onerec --output runs/onerec-sid
python -B -m sid_retrieval.tokenization --dataset office --method ghost --output runs/ghost-sid
python -B -m sid_retrieval.tokenization --dataset office --method sa2crq --output runs/sa2crq-sid
python -B -m sid_retrieval.tokenization --dataset office --method crab --output runs/crab-sid
```

Controlled tokenizers train3000 full-catalogue epochs; classic RQ-VAE trains20000.
The supplied embeddings are used as input. Controlled initialization uses the
supplied Torch Lloyd implementation;
classic RQ-VAE uses the original sklearn convention. Initialized codebook flags are
restored when loading. Ghost uses the constrained solver in requirements-tokenizers.txt.

To train GR on rebuilt SID maps, pass --artifact NEW_SID_DIRECTORY.

## 4. Analyses, ablation and parameters

```sh
python -B -m sid_retrieval.analysis --dataset industrial --analysis representation --output runs/industrial-representation
python -B -m sid_retrieval.analysis --dataset industrial --analysis diagnosis --output runs/industrial-diagnosis
python -B -m sid_retrieval.analysis --dataset office --analysis representation --output runs/office-representation
python -B -m sid_retrieval.analysis --dataset office --analysis diagnosis --output runs/office-diagnosis
python -B -m sid_retrieval.sweeps --dataset office --experiment ablation --output runs/office-ablation
python -B -m sid_retrieval.sweeps --dataset office --experiment parameters --output runs/office-parameters
python -B -m sid_retrieval.sweeps --dataset industrial --experiment parameters --output runs/industrial-parameters
python -B -m sid_retrieval.popularity --output runs/popularity
```

Office ablation emits the six original diagnostic cases plus serial_full; the paper
uses Full and four ablation rows. Office sensitivity uses VALID by default;
--split test evaluates TEST. Lambda/beta/mask each vary alone about(.5,.4,3),
with mask0..4.

Representation uses original raw-GR beam20 and exact matching, with cached mask
expansion tables. Diagnosis freezes fixed-pattern comparators on VALID before TEST.
Diagnosis compares target SIDs with the generated beams after decoding.

## 5. Efficiency, tests and immutable inputs

```sh
python -B -m sid_retrieval.benchmark --dataset industrial --batch-size 1 --output runs/latency
python -B -m sid_retrieval.benchmark --dataset office --batch-size 16 --output runs/throughput
python -B -m unittest discover -s tests -v
python -B -m sid_retrieval.integrity
```

Benchmark uses serial isolated processes,64 fixed requests and3 rounds. Measure
batch1 latency and batch16 throughput with other heavy jobs paused.

## 6. Render the seven-chapter report

```sh
python -B -m sid_retrieval.paper_report --output runs/paper-tables
```

This renders the English tables from numerical references.

## Method and fixed protocol

### 1. Offline representation

A content autoencoder maps the available 2560-dimensional item text embeddings to
96 dimensions. For the main method, split this latent vector into six disjoint
subspaces and quantize each with a 16-entry trainable VQ. Train catalogue-only
reconstruction MSE plus the existing VQ loss. Parallel refers to offline subspace
quantization. Four frozen representation controls are
included: serial/parallel 3x256 and serial/parallel 6x16.

### 2. Ordinary generative retriever

From-scratch Transformer, hidden128, 4 heads, encoder2/decoder2, FF384, dropout .1,
last10 history items. Positional SID embeddings are concatenated into item features.
Online decoding is causal and uses catalogue Trie constraints. Base training uses
teacher-forced token cross-entropy. Train for15 epochs and select the checkpoint
by full VALID token NLL.

### 3. TRAIN conditional prior correction

For each exact catalogue prefix p and token v, count TRAIN target transitions only.
Use Laplace+1 across the complete per-position vocabulary, including currently
Trie-forbidden classes. The transition score is

`log_softmax(logits(u,p) - lambda * log P_TRAIN(v|p))`, lambda=.5.

Normalize over the full vocabulary first, then apply the Trie mask. Lambda=0
recovers the original generation score.

### 4. Frozen user-semantic completion guidance

Mean-pool nonpadding frozen GR encoder states, map them with a trained linear128
projection and L2 normalize. Catalogue text is centered, randomized-PCA128 and
normalized. Projection training uses one full-catalogue contrastive/softmax CE at
fixed temperature .07 on TRAIN only; the frozen projection is epoch8.

For catalogue items completing prefix p, define

`Phi_u(p) = log mean_{i: SID_i starts with p} exp(cos(q_u,text_i)/.07)`.

Each expansion adds `beta * (Phi_u(child)-Phi_u(parent))`, beta=.4. The increments
telescope along a path. The mean avoids rewarding a prefix merely for containing
more items. This prefix energy guides beam expansion. Catalogue scoring includes
all items and SID aliases.

### 5. Partial matching and deterministic ranking

For each item, find its minimum Hamming distance to any finite generated beam SID.
Retain items at distance<=3, equivalent to enumerating every position subset with
0..3 masked slots. Rank lexicographically by:

1. smaller minimum mismatch count;
2. earlier generated beam attaining this minimum;
3. ascending stable item ID.

Return at most200 unique items, retaining SID aliases. Evaluate each prediction
against the target item ID. TRAIN popularity determines transition priors.

### Motivation analyses

1. Compare serial and parallel SID construction at3x256 and6x16 using raw-GR
   beam20 and exact matching.
2. Measure the minimum Hamming distance between each target SID and a complete
   generated beam, and evaluate recovery through partial matching.
3. Track individual-coordinate evidence, jointly matching position subsets and
   legal tolerant completions through beam search.

## Efficiency measurement

Ours uses beam20 and partial matching to expand generated SIDs into ranked lists
of up to200 items. Baselines use beam200.

Timing starts with prepared history SID tensors and ends with the final item lists.
It includes history encoding, semantic potentials, prior correction, beam generation,
matching, alias expansion and sorting. Model loading and input preparation precede
timing. Batch1 measures latency; batch16 measures throughput.

Reference measurements use Apple M4,16GiB, macOS15.5, CPU FP32, PyTorch2.8,
intra-op2/inter-op1, BLAS2,64 shared requests and3 serial rounds. Peak RSS is
process lifetime high-water memory. Numerical timings are in
references/efficiency-historical.json.

## Data and artifact provenance

The datasets are the public Reviews2018 categories Industrial and Scientific and
Office Products. Prepared TRAIN/VALID/TEST sequences use numeric item IDs and
pseudonymous user IDs, with the original split membership and row order preserved.

| Dataset | Items | TRAIN | VALID | TEST | Embedding width |
|---|---|---|---|---|---|
| Industrial | 3686 | 36259 | 4532 | 4533 | 2560 |
| Office | 3459 | 38924 | 4866 | 4866 | 2560 |

Prepared splits and2560-dimensional Qwen text embeddings come from MiniOneRec
SID resources. Embedding rows, SID rows and CSV item IDs share the same ordering.
Tokenizer training uses catalogue content. GR training and popularity counts use
TRAIN; checkpoint selection uses VALID.

Artifacts contain model states, SID maps, tokenizers, semantic projections and
reference predictions. Configurations are in configs/frozen.json.

Sources:

- Reviews 2018 dataset information: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
- MiniOneRec SID resources: https://github.com/AaronHeee/MiniOneRec

## Verification

Main replay covers14 method/dataset cases and65,793 requests. Verification records
also include the complete4,866-request Office serial-Full replay,13 Office VALID
parameter configurations,28 tests, installation checks and relocated execution.
Numerical evidence is in verification/; immutable file hashes are in
MANIFEST.sha256.json.

## License

Project code is Apache-2.0. See LICENSE and NOTICE for project and third-party
notices. Data and content-derived artifacts follow the upstream resource terms.
# ReSID
