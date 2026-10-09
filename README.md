# RIPPLe

RIPPLe is a provenance-first pipeline for preparing Rubin observations for
strong-lens research. It keeps survey retrieval, deterministic numerical work,
model qualification, and agent decisions behind separate typed interfaces.

The repository contains two data-access paths:

- the original LSST Science Pipelines/Butler utilities under `ripple.data_access`;
- a lightweight external Rubin DP2 path under `ripple.dp2`, authenticated only
  through the local `RSP_TOKEN` environment variable.

## Current implementation

| Area | State |
| --- | --- |
| DP2 discovery and cutout retrieval | Implemented for one bounded `LSST.DP2` r-band deep-coadd request |
| Observation package | Image, variance, mask, WCS, calibration metadata, identity, checksums, and explicit PSF status |
| Scientific preprocessing | Deterministic 64 x 64 single-channel Mriganka adapter with a saved tensor, manifest, and QA preview |
| Mriganka inference | Blocked pending the exact encoder/classifier checkpoints and missing training-domain metadata |
| Researcher repository analysis | Read-only PydanticAI route over a bounded, immutable code/config snapshot |
| Simulation and training smoke | Typed ten-image SLSim campaign, architecture selection, remote worker execution, and technical report |
| LensCat | Deterministic final-stage catalog matching; not used as a classifier |

The provisional Mriganka transform is available for integration work, but its
output is not a scientifically qualified classifier input yet. The checked-in
manifest therefore permits preprocessing and rejects model execution.

## Layout

```text
ripple/
  dp2/             Rubin DP2 client, retrieval, and observation packaging
  preprocessing/   deterministic transforms and artifact verification
  modeling/        model manifests, adapter registry, and source-analysis agent
  scientist/       typed routes, agents, tools, workflows, and worker entry point
  data_access/     original Butler-based data access
configs/scientist/ portable request and smoke-campaign examples
```

The scientist subsystem follows a small control-plane pattern: Pydantic models
define the contracts, ordinary Python tools perform side effects, PydanticAI
agents choose only allowlisted analysis actions, and workflow code owns state,
budgets, hashes, and stopping conditions. Researcher-supplied code is treated as
untrusted text and is not imported or executed by the repository-analysis route.

## Environments

Python 3.12 is the supported control-plane interpreter. Dependencies are split
by execution boundary:

```bash
python3.12 -m venv .venv-agent
source .venv-agent/bin/activate
python -m pip install -r requirements-scientist-agent.txt
```

- `requirements-dp2.txt`: external DP2 retrieval and package construction
- `requirements-m3.txt`: DP2 plus preprocessing and visualization
- `requirements-agent.txt`: OpenAI-backed onboarding support
- `requirements-scientist-agent.txt`: DP2/M3 plus Bedrock-backed orchestration
- `requirements-scientist-worker.txt`: offline numerical worker dependencies

The worker image must supply a compatible PyTorch/CUDA installation. Its fully
resolved Python 3.12 Linux lock is stored at
`ripple/scientist/locks/requirements-worker-linux-x86_64-py312.lock`.

## DP2 to preprocessing

Credentials stay outside the repository. Export `RSP_TOKEN` in the current
shell, then run:

```bash
python -m ripple.dp2.cli --output-root outputs/dp2_smoke
python -m ripple.dp2.package_cli --output-root outputs/dp2_m2
python -m ripple.preprocessing.cli \
  --package outputs/dp2_m2/<run-id>/package.json \
  --output-root outputs/m3_mriganka
python -m ripple.modeling.cli list
```

All output roots are ignored by Git. Retrieval receipts deliberately exclude
the token, authorization headers, and access URLs.

## Agentic routes

The dispatcher exposes three request types:

1. `mriganka_dp2` verifies an existing observation package and runs the
   registered preprocessing adapter. It stops at the closed inference gate.
2. `researcher_model` snapshots supplied code/config files, lets an agent plan
   and investigate through read-only tools, and returns evidence-linked
   integration findings.
3. `simulation_training` runs the bounded SLSim and GPU-worker smoke campaign.

Inspect a request without performing network, model, or remote work:

```bash
python -m ripple.scientist.router_cli plan \
  --request configs/scientist/researcher-model.example.json \
  --researcher-output-root outputs/researcher-runs
```

For a live researcher run, copy the example into the ignored `configs/local/`
directory, set its absolute source path, configure the non-secret
`RIPPLE_BEDROCK_PROFILE`, `RIPPLE_BEDROCK_REGION`, and
`RIPPLE_BEDROCK_MODEL_ID` values, and use the `run` subcommand.

The simulation examples contain placeholder SSH deployment values. Copy them
to `configs/local/`, replace the host and dedicated worker paths, and validate
before execution:

```bash
python -m ripple.scientist.campaign_cli validate \
  --configuration configs/local/campaign.json
```

## Scientific boundaries

- Agent output cannot override a model manifest, preprocessing recipe,
  checkpoint identity, or qualification gate.
- A sharper image, higher score, or catalog association is not confirmation of
  a gravitational lens.
- LensCat is consulted only after real candidate evidence exists and no-match is
  never interpreted as a non-lens label.
- Synthetic smoke metrics are integration evidence, not survey-performance
  measurements.
- Every promoted classifier still needs held-out, Rubin-compatible calibration
  and selection-effect evaluation.

Vendored scientific sources and the design precedents are recorded in
`ripple/scientist/THIRD_PARTY_NOTICES.txt` and
`ripple/scientist/locks/source-revisions.json`.
