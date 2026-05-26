# Megaprop

Cross-repo control repository for the FEATURE_GRAM matrix optimizer integration.

This repository intentionally uses Git submodules, not copied source trees. Each component keeps its own upstreamable Git history while this superproject pins one coherent cross-repo state.

## Pinned Components

| Component | Fork branch | Purpose |
| --- | --- | --- |
| Megatron-LM | `plugyawn/Megatron-LM@codex/feature-gram-matrix-optimizers` | Megatron Core metadata, config, native collection, optimizer routing |
| Emerging-Optimizers | `plugyawn/Emerging-Optimizers@codex/feature-gram-matrix-optimizers` | Newton-Muon/LocoProp-S rules, TP apply helpers, FEATURE_GRAM kernels |
| TransformerEngine | `plugyawn/TransformerEngine@codex/feature-gram-matrix-optimizers` | TE extra wgrad helper and fused-module FEATURE_GRAM collection |

## Checkout

```bash
git clone --recurse-submodules https://github.com/plugyawn/Megaprop.git
```

To refresh to the branch tips declared in `.gitmodules`:

```bash
git submodule update --init --remote --recursive
```

The superproject commit pins exact SHAs for reproducibility.
