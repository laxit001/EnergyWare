# EnergyAware AI: measuring LLM routing across model sizes

Can a router that reads a question pick a smaller model and save GPU energy without losing accuracy?
This project measures real per-query accuracy, latency and **GPU energy** for three open-weight models and tests learned routers against simple fixed policies.

**Result:** a measurement study with a negative result on router gains. No learned router significantly beat the best fixed policy, even with a much better difficulty predictor.

Final-year project by Laxit (B.Tech CSE, AI Specialization, Shoolini University). Everything runs on free Colab or Kaggle GPUs.

## Key findings

- **Math generation is almost all of the energy.** A GSM8K answer costs about 865 J on the large model versus about 18 to 22 J for a multiple-choice answer, so GSM8K is about 98% of the energy in this 800-query mix.
- **The 3B model is a strong fixed baseline.** It keeps 94.4% of the 7B model's accuracy and uses 9.5% less raw GPU energy, but only 1.6% less net (above-idle) energy, because the 4-bit 7B model is not cheaper to run on this GPU.
- **An oracle could save about 28% net energy** (it sees the answers, so this is only an upper bound).
- **Hidden-state probes predict failure far better than text features.** On GSM8K the mean AUC is about 0.73 (probe of a 0.5B model) versus about 0.62 (text features), at a cost of 0.62 J per question.
- **That did not turn into an energy win.** At 95% of the large model's accuracy, the probe router saves 6.8% raw energy versus 7.2% for the best fixed mix (difference -0.4 points, 95% bootstrap interval -4.9 to +4.7). The intervals include zero at every accuracy level tested.

| Tier | Accuracy (all) | GSM8K accuracy | Raw energy (J/query) | Net energy (J/query) | GSM8K raw energy (J) |
|---|---|---|---|---|---|
| Small (Qwen2.5-0.5B) | 0.401 | 0.390 | 192.9 | 44.6 | 381 |
| Medium (Qwen2.5-3B) | 0.754 | 0.780 | 400.3 | 230.4 | 794 |
| Large (Qwen2.5-7B, 4-bit) | 0.799 | 0.828 | 442.5 | 234.1 | 865 |

800 queries: 400 GSM8K, 200 MMLU, 200 ARC-Challenge. 2,400 (query, tier) records.

## Demo

`docs/index.html` is a self-contained page that replays the 400 GSM8K questions. Move a slider to trade accuracy for energy and compare the router with fixed policies.
After enabling GitHub Pages (Settings -> Pages -> Deploy from branch -> `main` -> `/docs`) it is served at `https://laxit001.github.io/EnergyWare/`.

## Repository layout

| Path | What it is |
|---|---|
| `notebooks/01_collect_data.ipynb` | Runs every query through all three tiers, logging correctness, latency and NVML GPU energy (resumable) |
| `notebooks/02_router_baselines.ipynb` | Sentence-embedding router, baselines (always-tier, length rule, random mix), oracle, in-domain and out-of-domain tests |
| `notebooks/03_probe_router.ipynb` | Hidden-state probe router, expected-utility policy, probe overhead, bootstrap confidence intervals |
| `docs/index.html` | The interactive demo |
| `data/` | Demo data and notes on the result files |

## Reproduce

1. Open `notebooks/01_collect_data.ipynb` in Colab with a **GPU runtime** and run it from the top (about 5 hours in total across sessions; it resumes after disconnects). Results go to `MyDrive/energyaware`.
2. Run `02_router_baselines.ipynb` (CPU is fine), then `03_probe_router.ipynb` (GPU for the first run; features are cached afterwards).
3. No paid services or API keys are needed. Models and datasets download from Hugging Face.

If the install cell is followed by a `sympy` import error, restart the session once and run again.

## Limitations

- 400 math queries and 200 each of MMLU and ARC; one GPU family; GPU-only energy (CPU and memory not counted).
- Idle GPU power varied between about 27 W and 35 W across sessions, which moves raw energy by up to about 10% for the smallest model; net energy is less affected.
- Greedy decoding only. The probe layer was chosen on the same data used for evaluation, which is slightly optimistic.
- The oracle is an upper bound, not an achievable result.

## References

- Ong et al., "RouteLLM: Learning to Route LLMs with Preference Data," arXiv:2406.18665, 2024.
- Chen, Zaharia and Zou, "FrugalGPT," arXiv:2305.05176, 2023.
- Ding et al., "Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing," ICLR 2024.
- Aggarwal et al., "AutoMix: Automatically Mixing Language Models," NeurIPS 2024.
- Samsi et al., "From Words to Watts: Benchmarking the Energy Costs of LLM Inference," IEEE HPEC 2023.

## License

MIT. See `LICENSE`.
