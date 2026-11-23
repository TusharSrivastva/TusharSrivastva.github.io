---
layout: post
title: "Knowledge Compilation into Neural Network Weights: Can LLMs Build Classifiers Without Training Data?"
date: 2026-11-22
redirect_from:
  - /research/knowledge-compilation/
math: true
image: /assets/img/wsp-results.png
description: "Can an LLM write a neural network's weights without seeing any data? I tested MIT CSAIL's weight-space programming idea on a small classifier, with and without the rule that generated the data."
---

## Abstract

Neural networks are usually trained by gradient descent on examples. This study asks whether a large language model (LLM) can instead write a network's weights directly from a description of the problem, with no training data. Using Claude Opus 5.5 and a 513-parameter classifier on synthetic data, I compare gradient descent with LLM-written and LLM-edited weights, with and without the rule that generated the data. Given the rule, the LLM's networks reached 93.0% test accuracy against 79.2% for gradient descent. Without it, they reached 64.6%, below always guessing the majority class. Weight-space programming works as a way to compile knowledge into weights, not as a substitute for it.

## 1. Introduction

In *Sparks of Weight-Space Programming* (MIT CSAIL, Nov 2026), Takehi, Teoh, Zhang and Khattab show frontier LLMs reading and writing neural-network weights directly. Given the rules and source code of MinAtar games, but no gameplay data, an LLM wrote a policy network in one reply that beat a PPO-trained baseline. The LLM wasn't learning from experience; it was turning a description into weights.

Games are fully described by their rules. Most real problems aren't. This study tests the idea on a small tabular classifier and asks two questions:

- **RQ1:** Can an LLM understand, construct and modify neural-network weights directly?
- **RQ2:** If so, does it need to know how the data was generated?

## 2. Method

**Data.** The data is a synthetic task in an invented domain, so the LLM cannot have seen it: predicting whether batches of fictional "velmorite" crystals fail a stress test.

- There are 569 rows. Each batch has 10 true properties, each measured three ways with noise (`_mean`, `_peak`, `_spread`), giving 30 features.
- Failure comes from five of the properties:

$$\ell = 1.4\,z_{\text{impurity}} + 0.9\,z_{\text{grain}} - 1.1\,z_{\text{lattice}} + 3.5\max(z_{\text{anneal}} - 1,\,0) + 2.0\,z_{\text{impurity}}\,z_{\text{cooling}} - 1.3$$

- A batch fails when $$\ell$$ plus small logistic noise is above zero. The rule has linear effects, a threshold and an interaction.
- The data is split 60/20/20 into training (341 rows), validation (114) and test (114). The test set is used once, at the end.

**Model.** A multilayer perceptron, 30 → 16 (ReLU) → 1 (sigmoid). It has 513 parameters, few enough to fit in a prompt as JSON.

**Arms.**

| Arm | What happens |
|---|---|
| 1. Gradient descent (GD) | Adam with early stopping on validation loss: the baseline |
| 2. LLM from scratch | The LLM writes all 513 weights in one reply, with no data and no feedback |
| 3. GD → LLM modification | The LLM edits a GD-trained network over 10 rounds, seeing only validation metrics after each round |

Each LLM arm runs under two conditions:

- **Rule known:** the prompt includes the exact generating rule, the classifier equivalent of game rules and source code.
- **Rule unknown:** the prompt includes only the task description, the feature names and the architecture.

**LLM.** Claude Opus 5.5, in isolated calls with no tools and no other context. The LLM never sees the data rows and never runs code.

**Calibration.** Each final model is also scored after temperature scaling. This rescales the output logit, $$p = \sigma(z/T)$$, with one number $$T$$ fitted on the validation set. It changes log loss but not accuracy or AUC, which separates real improvement from recalibration.

**Repetitions.** There are 3 GD seeds and 3 from-scratch trials per condition, and each GD model is modified once per condition. Results are means over the three runs.

## 3. Results

| Approach (test set) | Accuracy | AUC | Log loss | Log loss after temperature scaling |
|---|---|---|---|---|
| Always predict "passes" | 0.667 | 0.500 | – | – |
| Gradient descent | 0.792 | 0.814 | 0.628 | 0.547 |
| LLM from scratch, rule known | **0.930** | 0.983 | 0.212 | 0.187 |
| GD → LLM, rule known | 0.924 | **0.986** | **0.163** | 0.164 |
| LLM from scratch, rule unknown | 0.646 | 0.722 | 0.773 | 0.650 |
| GD → LLM, rule unknown | 0.798 | 0.813 | 0.549 | 0.554 |

**With the rule, the LLM out-built gradient descent with no data.** All three trials reached 93.0% accuracy, against 79.2% for gradient descent trained on 341 examples. The LLM's reasoning reads like a compiler at work:

- It used only the cleanest measurement (`_mean`) of each relevant property and gave every other input zero weight.
- Two ReLU units carry the linear terms, and one hinge unit, $$\text{ReLU}(z_{\text{anneal}} - 1)$$, carries the threshold.
- A ReLU network can't multiply, so it rewrote the interaction as $$2ab = \tfrac{1}{2}[(a+b)^2 - (a-b)^2]$$ and approximated each square with six ReLUs.

Each network came from a single reply costing about $0.11 (list-price equivalent). Its only flaw was underconfidence, which temperature scaling fixes.

**With the rule, editing became rebuilding:**

| GD seed | Accuracy | AUC | Log loss |
|---|---|---|---|
| 0 | 0.772 → 0.921 | 0.798 → 0.984 | 0.694 → 0.174 |
| 1 | 0.798 → 0.930 | 0.826 → 0.987 | 0.553 → 0.156 |
| 2 | 0.807 → 0.921 | 0.818 → 0.986 | 0.636 → 0.160 |

In every run the LLM eventually replaced the trained network with one built from the rule, changing 482 to 513 of the 513 weights. It then spent its remaining rounds on refinements a statistician would make: tuning the output scale, weighting measurements by how noisy they are, and smoothing the threshold. These were the best models in the study.

**Without the rule, construction failed.** The networks scored 64.6%, below always predicting "passes". The LLM built a plausible story from the feature names, and some of its directions were right, but it also gave weight to properties with no effect.

**Without the rule, modification was only calibration:**

| GD seed | Accuracy | AUC | Log loss vs temperature-scaled GD |
|---|---|---|---|
| 0 | 0.772 → 0.789 | 0.798 → 0.802 | −1.8% |
| 1 | 0.798 → 0.798 | 0.826 → 0.818 | +2.0% |
| 2 | 0.807 → 0.807 | 0.818 → 0.819 | +0.8% |

The edits were output rescaling, bias shifts and shrinking features the LLM guessed were weak. Ten LLM rounds matched what one fitted temperature does.

## 4. Discussion

**The LLM as a weight programmer.** Given a precise specification, the LLM did what a programmer does: it implemented the computation directly, without running it on thousands of examples first. The weights became an executable form of the supplied knowledge.

**Knowledge, not optimization, made the difference.** Gradient descent had to infer the structure from 341 examples. The LLM was handed it. The gap is therefore not evidence that LLMs are better optimizers. It shows that explicit structural knowledge can be worth more than a small dataset, and the rule-unknown control shows the LLM can't produce good weights without that knowledge.

**Editing can turn into reconstruction.** Once it had the rule, the LLM treated the trained network as a starting point, not something to preserve. Direct editing could be useful for revising existing models when new knowledge arrives.

**A possible small-data workflow.** Describe the known relationships, let an LLM write the network, then calibrate and check it on held-out data. This could suit engineering and scientific problems, where experts understand the process but labelled data is scarce.

## 5. Limitations

- **Synthetic data:** real problems have hidden variables and relationships that are hard to state explicitly.
- **Exact rule:** the LLM got the true coefficients, the strongest possible knowledge. Real domain knowledge is usually partial.
- **Small scale:** one LLM, a 513-parameter network and three runs per approach. Larger networks raise questions of context length and precision.
- **Limited baselines:** there is no comparison with symbolic regression or decision trees.
- **Arm 3 is not data-free:** it saw validation metrics after each round.

## 6. Conclusion

With the generating rule, an LLM wrote a classifier that beat gradient descent without seeing any training data, and rebuilt trained networks into better ones. Without the rule, it could do no more than calibrate. Weight-space programming is best understood as compiling knowledge into neural-network weights, not as a replacement for learning from data. Where the relationships are already known, weights need not always be learned; they can be programmed.

**Code, notebook and every LLM prompt and reply:** [github.com/TusharSrivastva/weight-space-classifier](https://github.com/TusharSrivastva/weight-space-classifier)

## References

1. Takehi, Teoh, Zhang and Khattab (2026). *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights.* MIT CSAIL blog post. [rikiyat.github.io/blog/weight-space-programming](https://rikiyat.github.io/blog/weight-space-programming/)
2. Kingma and Ba (2015). Adam: A Method for Stochastic Optimization. *ICLR.*
3. Guo, Pleiss, Sun and Weinberger (2017). On Calibration of Modern Neural Networks. *ICML.*
