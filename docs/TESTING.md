# TrainPilot — Testing

## Reference scripts and expected result

| Example | Situation | Expected verdict | Expected recommendation |
|---|---|---|---|
| 01 CIFAR CNN | slow loading | waiting for data | num_workers, pin_memory |
| 02 TinyStories | FP32 | GPU busy | mixed precision (FP16) |
| 03 ResNet-50 | heavy model | GPU busy | mixed precision (FP16) |

## Reference results on a T4

| Example | Step before (ms) | Step after (ms) | Speedup | Verdict correct? | Loss OK? |
|---|---|---|---|---|---|
| 01 | 11.4 | 8.0 | 1.44 | yes | yes |
| 02 | 493.4 | 151.5 | 3.26 | yes | yes |
| 03 | 372.0 | 152.0 | 2.45 | yes | yes |

Short run on a Tesla T4. The verdict matched the expectation for all three: 01 waiting for data, 02 and 03 GPU busy. Loss did not diverge, no NaN.

01, cell 10: 89.4% of the step is spent on the next batch. "Before/after" in row 01 is GPU compute only, after the batch is already in memory: 11.4 ms and 8.0 ms. Recommendation: num_workers and pin_memory.

02, cell 9: FP16 made the step 3.26× faster. Recommendation: mixed precision.

03, cell 9: GPU busy, wait was 2.2% in cell 4. FP16 made the step 2.45× faster. Recommendation: mixed precision. Write "little is left" only when the short test itself showed no clear speedup.

## Real scripts on a T4 (v0.1.0)

Five third-party scripts, Colab, Tesla T4. Every verdict was correct. Loss did not diverge, no NaN.

| Script | Verdict | Data wait | FP32 step (ms) | AMP step (ms) | Speedup | Loss OK? |
|---|---|---|---|---|---|---|
| CIFAR ResNet-18 | gpu busy | 0.3% | 116.8 | 46.8 | 2.5x | yes |
| DistilBERT IMDB | gpu busy | 0.2% | 310.4 | 96.3 | 3.22x | yes |
| nanoGPT Shakespeare | gpu busy | 0.6% | 183.5 | 58.6 | 3.13x | yes |
| Heavy augmentations, `num_workers=0` | waiting for data | 71.2% | — | — | — | yes |
| Heavy augmentations, `num_workers=2` | waiting for data | 65.0% | — | — | — | yes |
| Script already using AMP | gpu busy | 0% | 24.5 | 25.5 | ~1.04x | yes |

Also, nanoGPT + `torch.compile` on FP32 (v0.1.0 still compared compile to FP32): prepare 19.0 s, step 186.5 ms vs FP32 196.6 ms (~1.05x). That run had no compile verdict.

On the `num_workers=2` script, v0.1.0 suggested `num_workers=2` again. After the fix, if the loader already has 2 workers and `pin_memory`, the advice is not repeated: Colab/Kaggle have two CPU cores, so the next step is cheaper loading. On the script that already used AMP, the report said "Little is left" and offered no snippet.

Later Colab numbers (including compile on AMP) are in the README table.

## Manual timing check

- [ ] TrainPilot step time ≈ hand `time.time()` with synchronize (±5%)
- [ ] A second run gives similar numbers (±10%)
- [x] On CPU the library does not crash; it writes "GPU not found"
