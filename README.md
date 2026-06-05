# Megaprop
Large-scale distributed MoE training with some powerful right-preconditioned optimizers!
<img width="3252" height="1930" alt="image" src="https://github.com/user-attachments/assets/dee281dd-a3df-4ad1-82de-ddb7c04f8107" />
Figure: *Locoprop-S* beats *AdamW* on wallclock time on a TP=2 sweep on a Megatron GPT over a FineWeb Edu set, 2000 steps, two sweeps, with the feature gram matrix refreshed every 8 steps.

## Background
Newton-Muon and Locoprop are part of an evolving line of optimizers that study right-preconditioning on the gradient. However, well-known limitation has been some of them require local activation info at the optimization step.

However, 
- Newton-Muon can be expressed as $msign(G f(C))$, where C is the feature gram matrix. (NM chooses $f(C)$ to be $C^{-1}$ but it could be something else, from what I understand).
- Locoprop can also be expressed purely as a function of $(G,C)$ through the following construction:

<img width="611" height="585" alt="Image" src="https://github.com/user-attachments/assets/81aac8a8-7711-4595-861d-78aa8e55cb5c" />

Hence, by adding support  to route the  `_feature_gram_` ($X^T X$) _*beside*_ the `main_grad` ($dY^TX$), via a series of changes into `TransformerEngine`, `Megatron-LM`, and `Emerging-Optimizers`, with the possibility for a neat abstraction:

<img width="539" height="166" alt="Image" src="https://github.com/user-attachments/assets/7304e3ac-00d3-495e-917e-e7efa9fdcb59" />

## Experiments

On a TP=2 sweep on a Megatron GPT over a FineWeb Edu set, 2000 steps, two sweeps, with the feature gram matrix refreshed every 8 steps, Locoprop beats AdamW on wallclock time:

<img width="3252" height="1942" alt="Image" src="https://github.com/user-attachments/assets/5f49d908-ad78-47ef-bf85-7aa369257647" />

It's wasteful to materialize the full gram matrix, so with Locoprop I tried a diagonal approximation and a block diagonal approximation. 
The diagonal approximation seems to do well! NM appears to be slower at the moment due to the polar iteration maybe, I need to check.

<img width="3080" height="1928" alt="image" src="https://github.com/user-attachments/assets/33a06797-83fe-4fd0-8487-7ab3e82b1e8c" />



I think there should be a few more AdamW LRs checked, but the initial results look promising, and not streaming the activations seem to work. I double-checked to make sure that the calculations come out to be equivalent.

I've also attached a design doc here for reference. 
The diff excl. test files is not that huge:
```
repo                 | non-test diff
---------------------|-----------------------
Megatron-LM          | 13 files, +1336/-2
Emerging-Optimizers  | 6 files, +1140/-3
TransformerEngine    | 5 files, +281/-0
Total                | 24 files, +2757/-5
```

CC and thanks to: @mkhona-nvidia for his help!

[feature_gram_matrix_optimizers_design.pdf](https://github.com/user-attachments/files/28427984/feature_gram_matrix_optimizers_design.1.pdf)

TL;DR: Cross-repo control repository for the FEATURE_GRAM matrix optimizer integration.

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
