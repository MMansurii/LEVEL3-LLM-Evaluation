# LEVEL3-LLM-Evaluation

# Week 1
# 🦜 On the Dangers of Stochastic Parrots — Summary

This document provides a concise overview of the risks, costs, and possible mitigation strategies related to ever-larger language models (LLMs).  
It is based on the paper *"On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?"* (Bender et al., 2021).  

---

## 📊 Risks and Mitigation

| **Risks** | **Consequences** | **Examples from Source** | **Proposed Strategies** |
|-----------|------------------|---------------------------|--------------------------|
| **Unmanageable training data** | Reproduction of social biases | Reddit (67% US users male, 64% aged 18–29), Wikipedia (8.8–15% women), LGBTQ spaces filtered | Intentional dataset selection, source transparency |
| **Static data / Value lock** | Reinforcing outdated, less-inclusive views | Post-BLM: more coverage of shootings of Black people, but still selective focus | Curated retraining, community input |
| **Bias** | Discrimination against social groups | Blodgett et al. 2020 (bias overview), unreliable automated probing | Documentation, participatory evaluation |
| **Environmental & financial costs** | High CO₂ emissions, inequitable access | Transformer training = 284t CO₂ (≈ 56 years of avg. human emissions), +0.1 BLEU = $150k compute | Efficient hardware, renewables, energy reporting |
| **Synthetic language harms** | Hate speech, stereotyping, extremist propaganda | Extremist recruiting (McGuffie & Newhouse 2020), MT errors blamed on humans, PII extraction (Carlini 2020), hidden bias in search (Noble 2018) | Watermarking text, policy regulation |
| **Narrow research trajectories** | Over-focus on benchmarks (SOTA) | LMs succeed via spurious artifacts (Niven & Kao 2019; Bras et al 2020), no access to meaning (Bender & Koller 2020) | Value-sensitive design, pre-mortem analysis |

---

## 💡 Key Questions to Consider
- Are ever larger language models inevitable or necessary?  
- What costs are associated with this research direction?  
- Do NLP researchers or the public actually need larger LMs?  
- If yes, how can we pursue this direction while mitigating risks?  
- If not, what alternative research paths should we take?  
