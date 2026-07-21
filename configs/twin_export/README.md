# configs/twin_export — artifacts consumed by downstream repos

Hand-authored files that downstream repos (`UPF_NDT`, `UpfRLControllers`) depend
on but which are **not** emitted by `dvc.yaml`. They previously existed only in
downstream gitignored `data/external/profiling_twin/` directories, i.e. on two
developer machines and nowhere else. They are git-tracked here so they survive a
machine wipe and have a citable revision.

Both files are byte-for-byte the copies that were in use downstream (verified by
md5 against the downstream `.dvc` stubs). Do not edit the measurement values.

## Files

### `switching_costs.yaml`
RAPL package-0 activation-cost measurements (100 iterations per variant):
dpdk activation 24.0 s, usr 3.3 s, plus the `spike_wh ≈ P_steady × t / 3600`
scaling rule. Drives the switching-energy term `L_SW` in the RL reward.

Provenance caveat, preserved verbatim in the file's own header: these are raw
measurements from **a colleague's machine, not this profiling campaign's host**.
Absolute Wh figures are therefore not portable; only the activation durations
and the scaling rule transfer across machines. The file lives here because it is
campaign-adjacent measurement data with no better home — not because this
pipeline produced it.

### `params.yaml`
The config snapshot downstream pins. **This is not the repo-root `params.yaml`.**
Root `params.yaml` is the DVC pipeline config (ingest/merge/features/train
stages); this one is a small hand-authored summary for twin consumers holding
`capacity`, `operating_ranges`, and `layer1_targets`. The two files share a name
and nothing else, which is why this one is namespaced in a subdirectory rather
than living at the repo root. Its `operating_ranges` values are not derivable
from anything else in this repo.

## How downstream should fetch these

Replace the local `dvc add` stubs with imports from this repo:

    dvc import https://github.com/Shima-Af/UpfProfilingCampaign.git \
        configs/twin_export/switching_costs.yaml \
        -o data/external/profiling_twin/switching_costs.yaml \
        --rev <tag>

    dvc import https://github.com/Shima-Af/UpfProfilingCampaign.git \
        configs/twin_export/params.yaml \
        -o data/external/profiling_twin/params.yaml \
        --rev <tag>

`dvc import` resolves Git-tracked files as well as DVC outputs, so these need no
`dvc push` and no S3 credentials — the content travels in the Git object itself.
Downstream local paths are unchanged, so `configs/digital_twin_paths.yaml` needs
no edit.

`models/` is unaffected: it remains a cached DVC output of the `train` stage and
downstream continues to import it normally.
