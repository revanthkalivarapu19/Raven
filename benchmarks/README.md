# Raven Verification Accuracy Benchmark

Run the real Raven graph on a labeled dataset:

```powershell
python -m benchmarks.verification data\claims.csv
python -m benchmarks.verification data\claims.jsonl --output-prefix results\raven
```

The default output prefix is beside the input file and uses its stem plus
`_benchmark`. The harness writes `<prefix>.json` with per-example results and
aggregate metrics, and `<prefix>.csv` with one row per example.

## Dataset format

CSV files need `claim` and `ground_truth_label` headers. JSONL files need one
object per line with the same required keys. Optional fields such as `id`,
`domain`, `evidence`, and `source` are preserved in `dataset_annotations`.
Optional evidence/source data is reference annotation only; Raven continues to
retrieve and verify evidence through the existing production graph.

```csv
claim,ground_truth_label,domain,evidence,source
"The agency reduced its policy rate by 0.50 percent.",SUPPORTED,Finance,"Optional reference evidence","Optional source label"
```

```json
{"claim":"The agency reduced its policy rate by 0.50 percent.","ground_truth_label":"SUPPORTED","domain":"Finance","evidence":"Optional reference evidence","source":"Optional source label"}
```

Accepted ground-truth aliases include `true`/`real` for `SUPPORTED`,
`false`/`fake` for `CONTRADICTED`, `unknown`/`insufficient_evidence` for
`UNVERIFIED`, and `conflict`/`conflicting_evidence` for `CONFLICTING`.

## Metrics and interpretation

The JSON reports accuracy; fixed-four-class macro and per-class precision,
recall, and F1; a confusion matrix; UNVERIFIED rate; abstention accuracy;
correct and safe abstention counts; incorrect confident prediction count; and
average runtime and linearly interpolated p50/p95 runtime. Macro metrics
average across all four fixed categories; a category absent from the dataset
has zero precision, recall, and F1.

`abstention_accuracy` is the share of Raven's UNVERIFIED predictions whose
ground truth is also UNVERIFIED. `correct_abstention_rate` is the share of
UNVERIFIED ground-truth examples Raven abstained on.

An abstention on a supported, contradicted, or conflicting ground truth is
reported as a `safe_abstention`, not an incorrect confident prediction.
`incorrect_confident_prediction_count` includes only wrong non-UNVERIFIED
predictions. Pipeline invocation exceptions remain in the per-example outputs
as `pipeline_error` and are counted separately; they are excluded from
classification metrics rather than treated as Raven abstentions. Their
`raven_verdict` is empty because no verdict was produced.

## Initial real-data smoke-test dataset

[`data/claims.csv`](../data/claims.csv) contains 20 labeled examples selected
from the FEVER 1.0 shared-task development dataset. FEVER is a publicly
available, human-annotated fact extraction and verification dataset. Each CSV
row retains the FEVER record ID and original label in `source`; `evidence`
preserves the original evidence-set tuples, including Wikipedia page IDs and
sentence IDs when FEVER supplies them. FEVER does not provide a domain field,
so `domain` is intentionally blank rather than inferred.

Sources:

- [Official FEVER dataset description and format](https://fever.ai/dataset/fever.html)
- [Labeled FEVER shared-task development data](https://fever.ai/download/fever/shared_task_dev.jsonl)
- [FEVER data license notice](https://fever.ai/download/fever/license.html)
- Thorne et al., [FEVER: a large-scale dataset for Fact Extraction and VERification](https://arxiv.org/abs/1803.05355), NAACL-HLT 2018.

The original FEVER labels map directly by meaning:

| Original FEVER label | Raven benchmark label | Why this mapping is valid |
|---|---|---|
| `SUPPORTS` | `SUPPORTED` | FEVER annotators found evidence supporting the claim. |
| `REFUTES` | `CONTRADICTED` | FEVER annotators found evidence refuting the claim. |
| `NOT ENOUGH INFO` | `UNVERIFIED` | FEVER annotators could not establish support or refutation from the available evidence; this is an abstention, not a false label. |

The 20-row smoke-test sample is balanced across the available FEVER categories:
7 `SUPPORTED`, 7 `CONTRADICTED`, and 6 `UNVERIFIED`. The FEVER license notice
states that annotations incorporate Wikipedia material and must be used under
the applicable Wikipedia license terms or, where those are unavailable, the
Creative Commons Attribution-ShareAlike 3.0 terms.

Run this dataset with:

```powershell
python -m benchmarks.verification data\claims.csv
```

## Domain-specific E2E and verifier datasets

[`data/raven_e2e_claims.csv`](../data/raven_e2e_claims.csv) and
[`data/raven_verifier_claims.csv`](../data/raven_verifier_claims.csv) use the
same 20 claims from the SciFact 1.0 labeled development split. The records come
from `claims_dev.jsonl`; the verifier evidence text comes from the matching
abstracts in `corpus.jsonl`. The official release is
[data.tar.gz](https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz);
the source schemas are documented in the
[SciFact data documentation](https://github.com/allenai/scifact/blob/master/doc/data.md).

This is a fixed, label-stratified smoke-test cohort: 7 `SUPPORT`, 7
`CONTRADICT`, and 6 empty-evidence records. It is curated for biomedical and
health-related claims and is **not a statistically representative sample**.
SciFact does not provide domain labels; `Medical` is an explicit manual
benchmark annotation for each selected claim, not a label inferred by the
runner. All 20 rows use that domain.

The source records encode rationale labels inside the `evidence` mapping:

| SciFact dev record | Raven benchmark label | Mapping |
|---|---|---|
| Rationale label `SUPPORT` | `SUPPORTED` | The source annotates cited abstract sentence(s) as supporting the claim. |
| Rationale label `CONTRADICT` | `CONTRADICTED` | The source annotates cited abstract sentence(s) as contradicting the claim. |
| Empty `evidence` mapping | `UNVERIFIED` | SciFact supplies no annotated support/refutation rationale for the claim. |

Each CSV `source` value records the SciFact claim ID, original annotation,
cited S2ORC document IDs, and rationale sentence indices when present. In the
verifier dataset, `evidence` contains the exact annotated abstract sentences
for `SUPPORT` and `CONTRADICT` examples. For empty-evidence examples it contains
the complete abstract(s) of the claim's cited document(s); those are real
source text, not labeled rationales, and the source `UNVERIFIED` label remains
unchanged. Only whitespace between source sentences is normalized.

SciFact states that claims and evidence annotations are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); corpus abstracts are
from S2ORC and licensed under
[ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/). Refer to those
licenses when redistributing the dataset files.

Run the two modes with:

```powershell
python -m benchmarks.verification data\raven_e2e_claims.csv --supported-domains-only
python -m benchmarks.verification data\raven_verifier_claims.csv --mode verifier-only
```
