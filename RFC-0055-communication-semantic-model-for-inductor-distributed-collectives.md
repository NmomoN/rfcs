# RFC-0055: Communication Semantic Model for Inductor Distributed Collectives

## Summary

This RFC proposes introducing a small **communication semantic model** for distributed collectives in Inductor. The goal is to let distributed compiler passes reason about what a collective semantically represents, rather than inferring intent only from graph structure.

The initial model contains two core concepts:

* `CommRole`: the semantic purpose of a collective, for example `fsdp.param_allgather`, `fsdp.grad_reducescatter`, `fsdp.grad_allreduce`, or future roles such as `tp.shard_to_replicate` and `cp.kv_rotate`.
* `CommUrgency`: a semantic hint describing the scheduling flexibility of the collective, for example `critical_path`, `prefetch`, `deferred`, or `unknown`.

The first implementation will use current FSDP-related Inductor collectives as the first producer/consumer of this metadata. FSDP passes will first read `CommRole` / `CommUrgency` when present and fall back to existing structural heuristics when metadata is absent. Successful FSDP structural classification can then annotate canonical metadata for downstream passes.

Phase 1 is intentionally behavior-preserving. `CommUrgency` is **diagnostics-only** in phase 1: `OverlapScheduler` may read it for logging, counters, verifier checks, and coverage reporting, but it must not change scheduling priority, collective ordering, exposed-time accounting, memory budgeting, or any other scheduling decision based on urgency. Urgency-aware scheduling is discussed only as a future phase.

This RFC does **not** propose a stable public API, a full hardware cost model, a complete constraint solver, or a strategy registry in the first phase. Those are discussed as long-term extensions toward mixed-parallel communication optimization.

## Motivation

### Current distributed passes infer intent from structure

Inductor has useful distributed optimization passes that operate on functional collective nodes when the relevant distributed optimization options are enabled. For example, FSDP bucketing identifies FSDP all-gathers and reduce-scatters, micro-pipeline TP fuses hardcoded `all_gather + matmul` and `matmul + reduce_scatter` patterns, and overlap scheduling reasons about schedulable `(collective_start, wait_tensor)` pairs.

The problem is not that these passes exist. The problem is that most of the intent is inferred from structural patterns in the FX graph.

| Area | Current behavior | Limitation |
| --- | --- | --- |
| FSDP all-gather classification | Recognize all-gathers by structural ancestry, such as a single-placeholder parameter ancestry heuristic. | A TP or future all-gather may have a similar graph shape. |
| FSDP reduce-scatter classification | Recognize waits whose results flow to graph outputs through conservative unary chains. | The structural pattern is FSDP-specific and conservative. |
| FSDP all-reduce bucketing | Infer FSDP groups from all-gather / reduce-scatter recognition, then use group-name transitivity. | Group membership is not itself a semantic role. |
| FSDP reduce-scatter dedup | Match patterns like `add(wait(RS(a)), wait(RS(b)))` and rewrite to one `RS(add(a, b))` when operation-level constraints are satisfied. | The pass can preserve semantics for FSDP-only cases, but cross-role merges need a richer semantic representation. |
| Micro-pipeline TP | Match specific `all_gather + matmul` and `matmul + reduce_scatter` structures and replace them with `symm_mem.fused_*` operators. | After fusion, the original collective is no longer a normal schedulable collective node, but the fused op still contains a communication effect. |
| Overlap scheduling | Discover schedulable collectives and waits without explicit communication-role metadata. | It cannot distinguish an urgent TP collective from a flexible FSDP prefetch using semantic intent alone. |

When a graph contains a single distributed parallelism dimension, these heuristics are often sufficient. In mixed-parallel graphs, however, `_c10d_functional.*` collectives can represent very different semantic purposes even when their local graph shapes are similar. An all-gather can be an FSDP parameter prefetch, a TP shard-to-replicate transition, or part of a future context-parallel or expert-parallel schedule. Without explicit semantics, each pass has to guess.

### Why explicit communication semantics

The compiler needs a small contract that answers two questions for each communication node:

1. **What is this communication for?**
2. **How flexible is it for scheduling and diagnostics?**

For example:

* An FSDP parameter all-gather is often a **prefetch** communication. It can be moved earlier within a safe scheduling window to overlap with previous compute.
* An FSDP gradient reduce-scatter is often **deferred** communication. It can overlap with later backward compute, subject to dependency and memory constraints.
* A TP shard-to-replicate transition can be on the **critical path** of a matmul and should not be treated as the same kind of communication as a flexible prefetch.
* A future CP KV exchange can also be on the **critical path** of attention computation.

This RFC starts with the smallest semantic model that appears useful: `CommRole` and `CommUrgency`.

### Why this RFC is scoped to distributed collectives, not scheduler changes

The long-term motivation is to support mixed-parallel communication optimization, including FSDP + TP + CP cases. However, changing scheduling behavior in the same first step would make the RFC much harder to evaluate because reviewers would need to reason about correctness, memory lifetime, and performance policy at the same time.

For phase 1, the proposed acceptance scope is therefore limited to:

* defining the metadata contract,
* validating it on current FSDP collectives,
* preserving and invalidating it soundly across graph transforms,
* exposing diagnostics that measure coverage and mismatches.

The title uses **Distributed Collectives** rather than **Distributed Scheduling** to reflect this narrower first step. The semantic model is intended to serve future scheduling work, but phase 1 does not change scheduling decisions.

### Why FSDP first

FSDP is a good first migration target because:

* There are already FSDP-specific Inductor passes that classify and bucket collectives.
* The existing heuristics provide a safe fallback and a bootstrapping mechanism for annotation.
* A first implementation can be behavior-preserving: read metadata when present, otherwise use the current structure-based logic.
* The FSDP path gives a concrete way to validate metadata preservation through bucketing, deduplication, and overlap diagnostics before expanding the contract to TP / CP / mixed parallelism.

## Goals

* Introduce an explicit communication semantic model for Inductor distributed collectives.
* Start with two minimal concepts: `CommRole` and `CommUrgency`.
* Represent the first version as private / experimental compiler metadata on FX nodes.
* Keep `node.meta["comm_semantics"]` as the single metadata namespace for this RFC.
* Migrate FSDP-related Inductor passes to consume the metadata first and fall back to existing structural heuristics.
* Let existing FSDP heuristics annotate metadata after successful classification, so later passes can consume a unified contract.
* Define preservation, transfer, and invalidation rules for metadata across graph transforms such as clone, stable sort, bucketing, deduplication, fusion, decomposition, and lowering.
* Explicitly handle merged collectives and fused communication-compute operators in the schema without introducing a second top-level metadata key.
* Add diagnostics and verifier checks for coverage and correctness.
* Keep existing behavior unchanged by default.
* Use this first step to discuss whether the same contract can later support FSDP + TP + CP mixed-parallel communication scheduling.

## Non-goals

* This RFC does not propose a stable public user API in phase 1.
* This RFC does not require all DTensor / TP / CP collectives to be annotated immediately.
* This RFC does not introduce a full hardware topology model or resource contention cost model in phase 1.
* This RFC does not introduce a complete constraint solver or scheduling-window model in phase 1.
* This RFC does not replace the `post_grad.py` distributed pass pipeline with a strategy registry in phase 1.
* This RFC does not change `OverlapScheduler` priority, collective ordering, exposed-time accounting, memory budgeting, or scheduling policy in phase 1.
* This RFC does not introduce cross-role reduce-scatter deduplication in phase 1.
* This RFC does not require a performance improvement for every FSDP-only workload in phase 1. The first phase is primarily a refactor and semantic-contract validation.

## Proposed Implementation

### Acceptance scope for this RFC

This RFC asks the community to discuss and, if accepted, agree to the following first step:

* Inductor distributed passes should start moving from purely structural collective recognition toward explicit communication semantics.
* `CommRole` and `CommUrgency` are the proposed minimal first semantic split.
* FSDP collectives are the first producer/consumer of the metadata contract.
* Metadata must have clear preservation, transfer, and invalidation rules so downstream passes can trust it.
* The contract should be designed so future DTensor / TP / CP producers can use it without baking FSDP-specific assumptions into Inductor.
* Phase 1 diagnostics can expose how metadata would classify collectives, but no scheduling behavior changes are made.

### Relationship to the long-term distributed graph optimizer

This RFC should be read as the first concrete, independently landable step toward a broader distributed graph optimizer for Inductor. The long-term architecture may include richer resource profiles, hardware-contention and cost models, explicit scheduling constraints, and a strategy registry for distributed graph transformations.

This RFC does not ask the community to accept those later components now. Instead, it validates the shared semantic contract that those components would need in order to reason about distributed collectives soundly. Before a future cost model, scheduling policy, or strategy registry can make reliable decisions, Inductor needs a pass-preserved way to distinguish whether a collective represents FSDP parameter prefetch, FSDP gradient synchronization, TP shard-to-replicate communication, CP KV exchange, or a fused communication-compute region.

For that reason, the proposed phase-1 implementation focuses on `node.meta["comm_semantics"]`, `CommRole`, `CommUrgency`, FSDP migration, metadata preservation / transfer / invalidation rules, and diagnostics-only integration. Future phases can build on the same contract without requiring phase 1 to change scheduling priority, introduce a hardware cost model, or replace the current pass pipeline.

### Metadata namespace

Add a small private helper module, for example:

```text
torch/_inductor/fx_passes/comm_semantics.py
```

The exact location is open to discussion. The metadata is attached to FX nodes under a private key:

```python
COMM_SEMANTICS_KEY = "comm_semantics"
node.meta[COMM_SEMANTICS_KEY] = CommSemantics(...)
```

This RFC intentionally keeps a **single top-level metadata key**. It does not introduce a separate `node.meta["comm_effects"]` namespace. Instead, `CommSemantics` is shaped so it can represent:

1. an ordinary atomic collective,
2. a merged collective,
3. an opaque fused communication-compute operator.

### Metadata schema

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

COMM_SEMANTICS_KEY = "comm_semantics"


class CommUrgency(Enum):
    CRITICAL_PATH = "critical_path"
    PREFETCH = "prefetch"
    DEFERRED = "deferred"
    UNKNOWN = "unknown"


class CommSemanticsKind(Enum):
    COLLECTIVE = "collective"
    MERGED_COLLECTIVE = "merged_collective"
    FUSED_COMM_COMPUTE = "fused_comm_compute"
    INVALIDATED = "invalidated"


class CommRoles:
    # FSDP roles in phase 1.
    FSDP_PARAM_ALLGATHER = "fsdp.param_allgather"
    FSDP_GRAD_REDUCESCATTER = "fsdp.grad_reducescatter"
    FSDP_GRAD_ALLREDUCE = "fsdp.grad_allreduce"

    # Illustrative future roles, not required in phase 1.
    TP_SHARD_TO_REPLICATE = "tp.shard_to_replicate"
    TP_PARTIAL_TO_SHARD = "tp.partial_to_shard"
    TP_PARTIAL_TO_REPLICATE = "tp.partial_to_replicate"
    CP_KV_ROTATE = "cp.kv_rotate"
    CP_ACTIVATION_RESHARD = "cp.activation_reshard"
    PP_STAGE_SEND = "pp.stage_send"
    PP_STAGE_RECV = "pp.stage_recv"
    EP_TOKEN_DISPATCH = "ep.token_dispatch"
    EP_TOKEN_COMBINE = "ep.token_combine"


@dataclass(frozen=True)
class CommEffect:
    """One atomic communication effect represented inside CommSemantics.

    For an ordinary collective, the scalar fields on CommSemantics are usually
    enough. For a merged collective or fused communication-compute op, effects
    records the underlying atomic communication intents.
    """

    role: str
    collective_kind: str                    # "all_gather", "reduce_scatter", "all_reduce", ...
    urgency: CommUrgency = CommUrgency.UNKNOWN
    group_name: Optional[str] = None
    mesh_dim: Optional[str] = None
    reduce_op: Optional[str] = None
    scatter_dim: Optional[int] = None
    dtype: Optional[str] = None
    provenance: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CommSemantics:
    """Private compiler metadata for distributed communication semantics."""

    version: int
    kind: CommSemanticsKind
    source: str

    # Fast path for an ordinary collective with a single scalar role.
    role: Optional[str] = None
    urgency: CommUrgency = CommUrgency.UNKNOWN
    collective_kind: Optional[str] = None
    group_name: Optional[str] = None
    mesh_dim: Optional[str] = None

    # General form for merged collectives or fused comm-compute operators.
    effects: tuple[CommEffect, ...] = field(default_factory=tuple)

    # Whether this node is still an ordinary collective-start node that can be
    # paired with wait_tensor and considered as a schedulable collective.
    schedulable: bool = True

    # Which pass consumed or transformed the original collective, if any.
    optimized_by: Optional[str] = None

    # Names of original FX nodes or other debug provenance.
    provenance: tuple[str, ...] = field(default_factory=tuple)
```

The schema has a scalar fast path because most phase-1 users only need to ask whether a node is an FSDP all-gather, reduce-scatter, or all-reduce. The `effects` field is included so the same metadata namespace can later represent merged collectives and fused communication-compute operators.

The `effects` field does not need to be fully consumed by phase-1 FSDP passes. The important phase-1 rule is that a transform must not blindly copy a scalar `CommRole` to a node that no longer represents the same atomic collective.

### `CommRole`

`CommRole` classifies the semantic purpose of a collective. This RFC recommends representing roles as **namespaced strings** rather than a closed enum.

Examples:

```python
"fsdp.param_allgather"
"fsdp.grad_reducescatter"
"fsdp.grad_allreduce"
"tp.shard_to_replicate"
"tp.partial_to_shard"
"tp.partial_to_replicate"
"cp.kv_rotate"
```

Built-in roles can be provided as constants for typo avoidance, but the metadata contract should not require every future TP / CP / PP / EP role to modify a central enum before experimentation is possible.

Phase 1 only requires FSDP roles. Future roles are shown to validate that the namespacing scheme does not bake FSDP-specific assumptions into Inductor.

### `CommUrgency`

`CommUrgency` describes the expected scheduling flexibility of a communication effect:

| Urgency | Meaning |
| --- | --- |
| `critical_path` | The dependent compute is expected to block soon after this communication. |
| `prefetch` | The communication can often be issued earlier within a safe window. |
| `deferred` | The communication can often be overlapped with later compute. |
| `unknown` | No semantic urgency is known. Consumers must fall back to existing behavior. |

`CommUrgency` is **not** a cost model and is **not** a phase-1 scheduling policy. It is a semantic classification signal.

In phase 1:

* `OverlapScheduler` may read `CommUrgency` only for diagnostics, counters, verifier checks, and coverage reporting.
* `OverlapScheduler` must not use `CommUrgency` as a priority key.
* `OverlapScheduler` must not change collective ordering, exposed-time accounting, memory budgeting, or any existing scheduling decision because of `CommUrgency`.

Urgency-aware scheduling is a future phase after the metadata contract is validated on current FSDP graphs.

Illustrative defaults:

| Role | Default urgency | Phase-1 behavior |
| --- | --- | --- |
| `fsdp.param_allgather` | `prefetch` | Diagnostics only. |
| `fsdp.grad_reducescatter` | `deferred` | Diagnostics only. |
| `fsdp.grad_allreduce` | `deferred` or `unknown` | Unresolved; diagnostics only. |
| `tp.shard_to_replicate` | `critical_path` | Future producer; diagnostics only if present. |
| `tp.partial_to_shard` | `critical_path` | Future producer; diagnostics only if present. |
| `cp.kv_rotate` | `critical_path` | Future producer; diagnostics only if present. |

### Phase 1A: FSDP consumers read metadata first

FSDP-related helpers can read semantic metadata before falling back to the current structural heuristics.

```python
def get_comm_semantics(node: torch.fx.Node) -> CommSemantics | None:
    sem = node.meta.get(COMM_SEMANTICS_KEY)
    if isinstance(sem, CommSemantics) and sem.kind is not CommSemanticsKind.INVALIDATED:
        return sem
    return None


def has_comm_role(node: torch.fx.Node, role: str) -> bool:
    sem = get_comm_semantics(node)
    return sem is not None and sem.role == role


def is_fsdp_all_gather(node: torch.fx.Node) -> bool:
    sem = get_comm_semantics(node)
    if sem is not None and sem.role is not None:
        return sem.role == CommRoles.FSDP_PARAM_ALLGATHER

    # Fallback: existing structural heuristic.
    return _is_fsdp_all_gather_by_structure(node)
```

The fallback is important for backward compatibility and for incremental rollout.

### Phase 1B: FSDP heuristics annotate canonical metadata

When existing FSDP heuristics classify a collective, the pass can attach canonical metadata for downstream passes.

```python
if _is_fsdp_all_gather_by_structure(node):
    node.meta[COMM_SEMANTICS_KEY] = CommSemantics(
        version=1,
        kind=CommSemanticsKind.COLLECTIVE,
        source="heuristic.fsdp",
        role=CommRoles.FSDP_PARAM_ALLGATHER,
        urgency=CommUrgency.PREFETCH,
        collective_kind="all_gather",
        group_name=get_group_name(node),
        mesh_dim="dp",
        effects=(
            CommEffect(
                role=CommRoles.FSDP_PARAM_ALLGATHER,
                urgency=CommUrgency.PREFETCH,
                collective_kind="all_gather",
                group_name=get_group_name(node),
                mesh_dim="dp",
                provenance=(node.name,),
            ),
        ),
        schedulable=True,
        provenance=(node.name,),
    )
```

This makes the first implementation behavior-preserving: the same heuristic still decides the classification when metadata is absent, but later passes can use a unified semantic contract.

### Phase 1C: Private / experimental annotation source

This RFC does not propose a stable public annotation API in phase 1.

An internal or private experimental path may be useful for testing manual collectives, but it should not be presented as a stable `torch.distributed` API yet. For example, this is illustrative only:

```python
from torch.distributed._experimental import comm_semantics

with comm_semantics(role="fsdp.param_allgather", mesh_dim="dp"):
    full_param = torch.distributed.all_gather_tensor(shard, dim=0, group=dp_group)
```

The exact namespace and shape of any user-facing API are out of scope for phase 1.

### Phase 1D: Preservation helpers and verifier

Add small helper functions so passes do not mutate metadata ad hoc:

```python
def set_comm_semantics(node: torch.fx.Node, sem: CommSemantics) -> None:
    node.meta[COMM_SEMANTICS_KEY] = sem


def invalidate_comm_semantics(node: torch.fx.Node, reason: str) -> None:
    old = node.meta.get(COMM_SEMANTICS_KEY)
    node.meta[COMM_SEMANTICS_KEY] = CommSemantics(
        version=1,
        kind=CommSemanticsKind.INVALIDATED,
        source=f"invalidated:{reason}",
        schedulable=False,
        provenance=(node.name,),
    )


def verify_comm_semantics(gm: torch.fx.GraphModule) -> None:
    ...
```

The verifier can be enabled in debug / verbose modes and should check for obvious inconsistencies, for example:

* `fsdp.param_allgather` should not be attached to a reduce-scatter op.
* `fsdp.grad_reducescatter` should not be attached to an all-gather op.
* `CommSemantics(kind=FUSED_COMM_COMPUTE)` should have `schedulable=False` unless a future fused-op scheduler explicitly supports it.
* `CommSemantics(kind=MERGED_COLLECTIVE, role=None)` should not be consumed by scalar-role-only FSDP helpers as if it were an FSDP-only collective.

### Phase 1E: OverlapScheduler diagnostics only

In phase 1, `OverlapScheduler` can optionally read `CommRole` and `CommUrgency` for diagnostics only.

Examples of allowed phase-1 diagnostics:

* Count how many schedulable collective-start nodes have `comm_semantics`.
* Count unknown / invalidated / missing semantics by collective kind.
* Print a debug summary of roles and urgency classes encountered by the scheduler.
* Compare structural FSDP classification with metadata classification and warn on mismatches in verbose mode.
* Record whether a collective became an opaque fused communication-compute node before scheduling.

Examples of disallowed phase-1 behavior:

* Sorting ready collectives by `CommUrgency`.
* Raising or delaying collectives because of `CommUrgency`.
* Changing exposed-time accounting because of `CommUrgency`.
* Changing memory budget decisions because of `CommUrgency`.
* Treating `FUSED_COMM_COMPUTE` nodes as ordinary `(collective_start, wait_tensor)` pairs.

Urgency-aware scheduling is deferred to a future phase after correctness and performance policies are discussed separately.

## Metadata preservation, transfer, and invalidation rules

Communication semantic metadata is a compiler contract. A graph transform must do one of the following:

1. **Preserve** scalar semantics when the transformed node still represents the same atomic communication intent.
2. **Transfer** the original semantics into a richer representation when the transform merges collectives or fuses communication with compute.
3. **Invalidate** the metadata when neither preservation nor transfer is sound.

A transform should not arbitrarily choose one input role when the output no longer has a single role.

### Canonical location

For an ordinary functional collective, metadata is attached to the collective-start node, for example `_c10d_functional.all_gather_into_tensor`, `_c10d_functional.reduce_scatter_tensor`, or `_c10d_functional.all_reduce`.

`wait_tensor` nodes can inspect the metadata on their input collective. This avoids duplicating mutable metadata across both the start node and wait node.

For an opaque fused communication-compute operator, metadata is attached to the fused op node with `kind=FUSED_COMM_COMPUTE` and `schedulable=False`.

### Clone, copy, and stable topological sort

Graph clone, node copy, and stable topological sort may copy metadata by value.

Because the proposed dataclasses are frozen, transforms should create new metadata objects rather than mutating shared objects in place.

### Bucketing same-role collectives

Bucketing multiple collectives may preserve a scalar role only when the merged collectives have compatible communication semantics and operation-level constraints.

For example, bucketing FSDP parameter all-gathers can preserve:

```python
role = "fsdp.param_allgather"
kind = CommSemanticsKind.MERGED_COLLECTIVE
```

only when all inputs are compatible with the same scalar role and collective behavior, including:

* same or compatible `role`,
* same collective kind,
* compatible process group / group name,
* compatible mesh dimension,
* compatible dtype / layout / payload shape constraints,
* compatible scheduling constraints.

The bucketed node can record the original operations in `effects` and `provenance`.

### Reduce-scatter deduplication

For phase 1, reduce-scatter deduplication preserves metadata only when all input reduce-scatter nodes have the same scalar role, for example multiple `fsdp.grad_reducescatter` nodes.

This same-role rule is a conservative **metadata preservation rule**, not a fundamental algebraic requirement. A cross-role reduce-scatter merge may be mathematically valid if operation-level constraints are compatible, such as:

* same collective kind: `reduce_scatter`,
* compatible process group / group name,
* compatible `reduce_op`, usually a linear op such as sum or average,
* compatible scatter dimension and output layout,
* compatible input/output shape,
* compatible dtype and numerical policy,
* compatible dependency and single-use pattern.

However, if two reduce-scatter nodes come from different parallelism strategies, the resulting node no longer has a single role. For example, merging an FSDP gradient reduce-scatter with a TP partial-to-shard reduce-scatter should not produce a node that claims to be only `fsdp.grad_reducescatter` or only `tp.partial_to_shard`.

Phase 1 does not introduce cross-role reduce-scatter deduplication. If an existing transform cannot soundly preserve a scalar role, it must either skip the transform or invalidate the scalar metadata rather than choosing one role arbitrarily.

A future extension could represent a cross-role merged collective as:

```python
merged_rs.meta[COMM_SEMANTICS_KEY] = CommSemantics(
    version=1,
    kind=CommSemanticsKind.MERGED_COLLECTIVE,
    source="future.cross_role_reduce_scatter_dedup",
    role=None,
    urgency=CommUrgency.CRITICAL_PATH,  # conservative join of the effects
    collective_kind="reduce_scatter",
    group_name=shared_group_name,
    mesh_dim=None,
    effects=(
        CommEffect(
            role="fsdp.grad_reducescatter",
            collective_kind="reduce_scatter",
            urgency=CommUrgency.DEFERRED,
            group_name=shared_group_name,
            mesh_dim="dp",
            provenance=(rs_fsdp.name,),
        ),
        CommEffect(
            role="tp.partial_to_shard",
            collective_kind="reduce_scatter",
            urgency=CommUrgency.CRITICAL_PATH,
            group_name=shared_group_name,
            mesh_dim="tp",
            provenance=(rs_tp.name,),
        ),
    ),
    schedulable=True,
    provenance=(rs_fsdp.name, rs_tp.name),
)
```

Such a transform should be controlled by a future cost / policy decision because cross-role deduplication can reduce communication count while also tying together dependencies that might otherwise be scheduled independently. For example, merging a critical-path TP reduce-scatter with a deferred FSDP reduce-scatter may delay the TP communication until the FSDP input is ready.

### Fused communication-compute operators

Some passes replace a communication + compute pattern with an opaque fused operator. For example, micro-pipeline TP can replace:

```text
all_gather + matmul
matmul + reduce_scatter
```

with symmetric-memory fused operators such as:

```text
symm_mem.fused_all_gather_matmul
symm_mem.fused_all_gather_scaled_matmul
symm_mem.fused_matmul_reduce_scatter
symm_mem.fused_scaled_matmul_reduce_scatter
```

These are communication-compute fused operators, not ordinary compute-only nodes and not ordinary schedulable `(collective_start, wait_tensor)` pairs.

For such transforms, the original collective metadata must not be blindly copied to the matmul output or fused output as a scalar `CommRole`. Instead, the fused node should carry the original communication intent under the same `node.meta["comm_semantics"]` key, using `kind=FUSED_COMM_COMPUTE` and `effects`.

For `all_gather + matmul`:

```python
fused_node.meta[COMM_SEMANTICS_KEY] = CommSemantics(
    version=1,
    kind=CommSemanticsKind.FUSED_COMM_COMPUTE,
    source="micro_pipeline_tp",
    role=None,
    urgency=CommUrgency.CRITICAL_PATH,
    collective_kind=None,
    group_name=tp_group_name,
    mesh_dim="tp",
    effects=(
        CommEffect(
            role="tp.shard_to_replicate",
            collective_kind="all_gather",
            urgency=CommUrgency.CRITICAL_PATH,
            group_name=tp_group_name,
            mesh_dim="tp",
            provenance=(ag_node.name, wait_node.name),
        ),
    ),
    schedulable=False,
    optimized_by="micro_pipeline_tp",
    provenance=(ag_node.name, wait_node.name, *[mm.name for mm in matmul_nodes]),
)
```

For `matmul + reduce_scatter`:

```python
fused_node.meta[COMM_SEMANTICS_KEY] = CommSemantics(
    version=1,
    kind=CommSemanticsKind.FUSED_COMM_COMPUTE,
    source="micro_pipeline_tp",
    role=None,
    urgency=CommUrgency.CRITICAL_PATH,
    group_name=tp_group_name,
    mesh_dim="tp",
    effects=(
        CommEffect(
            role="tp.partial_to_shard",
            collective_kind="reduce_scatter",
            urgency=CommUrgency.CRITICAL_PATH,
            group_name=tp_group_name,
            mesh_dim="tp",
            reduce_op=reduce_op,
            scatter_dim=scatter_dim,
            provenance=(matmul_node.name, rs_node.name, wait_node.name),
        ),
    ),
    schedulable=False,
    optimized_by="micro_pipeline_tp",
    provenance=(matmul_node.name, rs_node.name, wait_node.name),
)
```

In phase 1, `OverlapScheduler` may read these fused semantics only for diagnostics and coverage reporting. It must not treat a fused communication-compute operator as a normal schedulable collective pair.

A future cost model may use `FUSED_COMM_COMPUTE` metadata to account for communication resource usage that is hidden inside a fused operator.

### Mixed-role merge and scalar-role consumers

If a transform merges collectives with different roles, the resulting node must not expose a misleading scalar `role`.

Options are:

1. Skip the transform in phase 1.
2. Invalidate scalar semantics.
3. Use `CommSemantics(kind=MERGED_COLLECTIVE, role=None, effects=(...))` in a future phase.

Scalar-role-only consumers, such as a phase-1 FSDP helper, must only trust `sem.role` when `sem.role` is not `None` and `sem.kind` is compatible with the consumer.

### Decomposition and lowering

A lowering or decomposition pass may preserve metadata only when the new node still has equivalent communication semantics.

If a collective is lowered into a backend-specific communication op that remains a schedulable collective-start node, the metadata may be preserved with updated `collective_kind` / provenance.

If a collective is lowered into a fused or opaque implementation, the metadata should be transferred into `effects` and marked `schedulable=False` if it is no longer visible as a normal collective pair.

If a pass cannot prove preservation or transfer is sound, it must invalidate the metadata.

## Diagnostics and validation

Phase-1 diagnostics should help answer whether the model is useful without changing behavior.

Potential diagnostics:

* Number and percentage of distributed collectives with valid `comm_semantics`.
* Number of FSDP collectives classified by metadata vs fallback heuristic.
* Number of metadata / heuristic mismatches.
* Number of missing, unknown, invalidated, merged, and fused communication semantics.
* Distribution of `CommRole` and `CommUrgency` values.
* Count of fused communication-compute nodes and their underlying `effects`.
* Debug dump of role / urgency / group / mesh_dim / provenance for distributed collectives.

Potential verifier checks:

* Role and collective kind are consistent.
* `schedulable=True` is only used for ordinary schedulable collective-start nodes.
* `FUSED_COMM_COMPUTE` nodes have `schedulable=False` in phase 1.
* Cross-role merged collectives do not expose a misleading scalar role.
* Metadata objects are versioned and use known schema versions.

## Long-term direction and future phases

The long-term direction remains a more general distributed communication optimization framework for Inductor. This RFC intentionally starts with a small semantic model because it is the minimum shared contract needed before richer scheduling and cost modeling can be built.

### Future phase 2: DTensor / TP / CP producers

Future producers can annotate collectives when communication is emitted from distributed tensor placement transitions or parallelism-specific transformations.

Examples:

| Source | Example role |
| --- | --- |
| DTensor `Shard -> Replicate` | `tp.shard_to_replicate` |
| DTensor `Partial -> Shard` | `tp.partial_to_shard` |
| DTensor `Partial -> Replicate` | `tp.partial_to_replicate` |
| Context parallel KV exchange | `cp.kv_rotate` |
| Pipeline stage communication | `pp.stage_send`, `pp.stage_recv` |
| Expert parallel routing | `ep.token_dispatch`, `ep.token_combine` |

### Future phase 3: urgency-aware mixed-parallel scheduling policy

A later phase can evaluate whether `CommUrgency` should influence scheduling decisions.

Possible policy questions:

* Should critical-path TP / CP communication be prioritized over flexible FSDP prefetches?
* Should FSDP prefetches be delayed when they would contend with critical-path communication on the same physical resource?
* Should fused communication-compute operators contribute to resource-usage accounting even though they are not normal schedulable collective pairs?
* How should priority interact with memory limits and dependency constraints?

This phase should be proposed and evaluated separately, likely behind a config flag, with correctness and performance data.

### Future phase 4: scheduling windows, resource profile, and cost model

Once role/urgency semantics are validated, the schema can grow richer fields, such as:

* scheduling windows,
* estimated bytes per rank,
* physical resource / NIC affinity,
* inter-node vs intra-node classification,
* memory overhead,
* resource contention estimates.

These fields can feed a hardware cost model, but they are not required for phase 1.

### Future phase 5: strategy registry / generalized distributed graph optimizer

The original long-term architecture can be revisited after the semantic contract is proven useful. In that future design, existing passes such as FSDP bucketing, TP communication-compute fusion, CP ring pipelining, low-contention collectives, and PP / EP optimizations could register as composable strategies over the semantic model.

This RFC does not ask the community to accept that full architecture now. It asks whether the semantic contract is the right first step.

## Metrics

Phase-1 metrics should focus on behavior preservation and metadata quality rather than runtime speedups.

1. **Behavior equivalence**
   * Existing FSDP-only tests should pass unchanged.
   * Metadata-based FSDP classification should match existing structural classification when both are available.
   * Existing graph rewrites should remain enabled / disabled under the same configs as before.

2. **Runtime behavior**
   * Phase 1 should not change runtime scheduling behavior.
   * Existing FSDP-only runtime should not regress because of scheduling changes, since no scheduling policy changes are made.
   * Urgency-aware scheduling is measured only in future phases, not as part of phase-1 acceptance.

3. **Compilation overhead**
   * Metadata annotation, validation, and diagnostics should add minimal compile-time overhead.
   * The verifier should be debug / verbose gated if it is expensive.

4. **Coverage**
   * Percentage of FSDP all-gather / reduce-scatter / all-reduce nodes with valid `comm_semantics`.
   * Number of fallback heuristic classifications that were converted into metadata.
   * Number of invalidated metadata entries by transform.

5. **Debuggability**
   * Ability to produce a clear per-collective summary: role, urgency, collective kind, group name, mesh dimension, source, provenance, and whether it is schedulable.

Future phases can add throughput, exposed communication time, cost-model accuracy, and mixed-parallel overlap metrics.

## Drawbacks

1. **Additional compiler metadata surface.** The proposal introduces a new private metadata contract. Even if the first version is small, it creates a maintenance responsibility for producers, consumers, and transforms.

2. **Annotation quality matters.** Incorrect metadata is worse than missing metadata. This is why phase 1 preserves fallback heuristics, adds diagnostics, and requires explicit invalidation when semantics cannot be preserved.

3. **Potential confusion around `CommUrgency`.** The name may suggest immediate scheduler behavior. This RFC mitigates that by making `CommUrgency` diagnostics-only in phase 1 and deferring urgency-aware scheduling to a future phase.

4. **Schema complexity for future-proofing.** Including `CommSemanticsKind` and `effects` makes the schema slightly larger than a minimal `role + urgency` pair. The benefit is that it can represent merged collectives and fused communication-compute ops without introducing a second top-level metadata namespace later.

5. **Testing burden.** Metadata preservation must be tested across FSDP bucketing, reduce-scatter deduplication, cloning, sorting, fusion, and lowering paths.

## Alternatives

### Alternative 1: Keep extending structural heuristics

We could continue adding specialized pattern matching for each distributed optimization. This is simpler for individual features, but it does not address mixed-parallel composition. Similar graph structures can represent different communication roles, and each pass must keep guessing intent.

### Alternative 2: Propose the full Distributed Graph Optimizer now

The broader design could include semantic modeling, cost modeling, constraint modeling, scheduling windows, and a strategy registry in one RFC. That matches the long-term direction, but it makes the first acceptance scope too large. This RFC proposes to validate the smallest useful semantic contract first.

### Alternative 3: Only add `CommRole` and `CommUrgency` without a broader `CommSemantics` model

This would make the first patch smaller, but it does not explain how metadata should survive graph transforms. It also leaves open questions for merged collectives and communication-compute fused operators. The proposed `CommSemantics` schema keeps the phase-1 required fields small while reserving structured space for future transforms.

### Alternative 4: Add a separate `node.meta["comm_effects"]` namespace

A separate top-level metadata key could represent fused communication-compute effects. This RFC does not choose that approach because it fragments the contract. Instead, `node.meta["comm_semantics"]` remains the single entry point, and `CommSemantics.effects` represents underlying atomic communication effects when needed.

### Alternative 5: Use a stable public annotation API immediately

A public API such as `torch.distributed.comm_semantics(...)` could help advanced users annotate manual collectives, but it would require stronger naming and compatibility guarantees. The proposed first phase keeps any API private / experimental.

### Alternative 6: Change `OverlapScheduler` priority in phase 1

We could immediately use `CommUrgency` to sort ready collectives. This is tempting because it is the eventual motivation, but it would mix a semantic-contract RFC with a scheduling-policy RFC. The proposed phase 1 is diagnostics-only so scheduler behavior remains unchanged while the metadata contract is validated.

## Prior Art

* **PyTorch DTensor.** DTensor placement transitions already encode distributed intent at a higher level than raw collectives. A future producer could preserve this intent as `CommRole` metadata when collectives are emitted.
* **Existing Inductor FSDP passes.** Current FSDP passes already classify and transform FSDP collectives, but primarily through graph structure and group-name transitivity. This RFC proposes a semantic contract that can coexist with those heuristics during migration.
* **Existing micro-pipeline TP pass.** The current TP pass recognizes `all_gather + matmul` and `matmul + reduce_scatter` patterns and replaces them with symmetric-memory fused operators. This RFC proposes how such fused operators should preserve communication semantics without pretending to be ordinary collectives.
* **XLA / GSPMD.** Sharding annotations give the compiler explicit distributed intent, and lower-level scheduling can reason from that intent. This RFC follows the same broad principle at Inductor's FX-level distributed collective layer.
* **Megatron-style hand scheduling.** Large-scale training systems often hand-code communication/computation overlap for tensor, context, and pipeline parallelism. A semantic model could provide a compiler-level contract for expressing similar intent.
* **Alpa-style planning.** Fully automatic parallelism planning uses cost models and search. This RFC is much smaller: it proposes the semantic metadata layer that could later feed richer scheduling or search.

## How we teach this

### For users

No user-visible API changes are required for phase 1. Existing FSDP users should see no behavior change.

If a future public or experimental API is introduced, it should be documented separately and should make clear that communication semantics are advanced compiler hints, not required for ordinary PyTorch distributed usage.

### For Inductor developers

The main concept is:

> If a pass needs to know what a distributed collective means, it should first look for `node.meta["comm_semantics"]`. If metadata is missing, phase-1 FSDP code may fall back to existing structural heuristics. If a pass transforms a collective, it must preserve, transfer, or invalidate the metadata explicitly.

Developer documentation should include:

* how to attach `CommSemantics` to an atomic collective,
* how to preserve it when bucketing same-role collectives,
* how to represent fused communication-compute operators,
* when to invalidate metadata,
* how to run the verifier / diagnostics.

## Unresolved questions

1. **Role representation.** Should `CommRole` remain a namespaced string, or should PyTorch define a string enum for built-in roles while still allowing custom roles?

2. **Exact metadata location.** Should the helper live under `torch/_inductor/fx_passes/comm_semantics.py`, a new distributed subpackage, or another internal namespace?

3. **Initial FSDP urgency defaults.** Should `fsdp.grad_allreduce` default to `deferred`, `critical_path`, or `unknown` for diagnostics?

4. **Metadata producer location.** Should phase 1 only annotate after FSDP heuristics classify collectives, or should metadata be produced earlier when FSDP / DTensor emits collectives?

5. **Cross-role reduce-scatter deduplication.** Should future phases allow cross-role deduplication when operation-level constraints are valid? If yes, should the resulting metadata use `role=None` with `effects`, or another representation?

6. **Fused communication-compute semantics.** Is `CommSemantics(kind=FUSED_COMM_COMPUTE, effects=(...), schedulable=False)` the right way to represent `symm_mem.fused_*` operators, or should these remain unannotated until a future pass consumes them?

7. **Diagnostics surfaced by `OverlapScheduler`.** What phase-1 counters and debug dumps would be useful without changing scheduling behavior?

8. **Future urgency-aware scheduling.** If a later RFC uses `CommUrgency` for scheduling, what correctness and performance evidence should be required?

9. **Schema stability.** How should versioning work if future fields such as scheduling windows, resource profiles, or topology hints are added?

