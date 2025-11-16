# Research Plan: Mechanistic Interpretability of Bias in Language Models

## Overview

We propose a comprehensive study to mechanistically analyze gender and racial biases in transformer LMs (English-focused, with a Hindi side analysis). We will use activation-based methods (e.g. causal tracing/patching and probing) to identify where bias is encoded (which heads/layers/edges) and to extract "bias vectors" that capture biased concepts in activation space. Based on prior work, we anticipate bias to be localized in a few components and to become more explicit in later layers [1][2]. Our plan builds on state-of-the-art techniques (activation/attribution patching [3], linear probes [1], steering vectors [1]) and bias benchmarks (e.g. StereoSet, WinoGender) to design targeted experiments. We will use open-source PyTorch models (GPT-2, GPT-Neo, possibly BERT) under 4B parameters and existing bias datasets (demographics and gender) to avoid new data collection. Our novelty lies in jointly leveraging edge-patching and vector-analysis to pinpoint bias mechanisms, while also examining Hindi bias via analogous benchmarks [4][5].

## 1. Background & Objectives

### Bias in LLMs

Large LMs trained on web data inherit social biases (gender, race, nationality, etc.) present in the corpora [7][8]. These biases can manifest as stereotypical completions or unfair associations. Quantifying bias often uses benchmarks like StereoSet and CrowS-Pairs for English [2] and WEAT/CrowS for other languages (e.g. Hindi) [6][5]. Our goal is not just to detect bias, but to interpret how and where it is represented internally (which layers, heads, neurons encode bias) using mechanistic methods.

### Mechanistic Interpretability

Prior work has successfully reverse-engineered circuits for well-defined tasks (e.g. arithmetic, induction) in transformer LMs [3]. Recent efforts have adapted these ideas to bias. For example, Vig et al. (2020) used causal mediation to find sparse gender-bias neurons in GPT-2 [9], and Conmy et al. (2023) introduced Edge Attribution Patching (EAP) to automatically prune model subgraphs for specific behaviors [3]. Chandna et al. (2025) showed that both demographic and gender biases are encoded in a small set of edges and layers in GPT-2 and Llama-2, and that ablations of these edges reduce bias (but also affect unrelated tasks) [10][7]. Yang et al. (2025) focused on attention heads, identifying a few "biased heads" in BERT and GPT/LLaMA that disproportionately contribute to stereotypes [7][11]. Yu & Ananiadou (2025) found "gender neurons" and broader "general neurons" circuits for gender bias, and noted that editing too many neurons can harm model capabilities [12]. Steering-vector approaches (Gupta et al. 2025, ICML Actionable Interpretability) compute a bias direction by subtracting mean activations of neutral vs biased examples [1][4]. 

In summary, evidence suggests:
- (a) bias often concentrates in late layers [1][2]
- (b) a few heads/edges carry disproportionate bias [7][11]
- (c) bias can be extracted as a linear direction in activation space [1][4]

### Research Questions

Building on this, our study will ask: Which specific components (heads, MLPs, residual edges) are most responsible for gender/racial bias? Can we identify a stable "bias vector" in activation space that captures these stereotypes? And how consistent are these findings across data perturbations and fine-tuning? We will operationalize bias via existing benchmarks (see Datasets), then use activation patching (both causal and attribution-based) and probes to trace bias. Our plan includes:

- **Localization**: Use edge-level patching to score each component's impact on a bias metric [3][10].
- **Probing**: Train simple classifiers on hidden activations to see in which layer bias is linearly decodable (following e.g. Nanda 2025) [1].
- **Bias Vectors**: Compute and analyze "bias directions" (e.g. difference of means [1]) in key layers, and test steering effects as in prior work [4].
- **Cross-validation**: Repeat experiments across multiple bias categories (gender vs nationality) and data variations to test stability [7].

Each step will be documented with citations to related methods to ensure we build on proven techniques while contributing new insights.

## 2. Data Selection

### English Bias Datasets

We will use existing bias benchmarks covering gender and nationality (racial) stereotypes, without creating new data. For gender bias, a common approach is to use sentence templates with profession or role words from Bolukbasi et al. (2016), who provided exhaustive male/female occupation lists [10]. We will form prompts like "The [profession] said that..." and measure gendered completions. For nationality bias, we can leverage the dataset of Narayanan et al. (2023) [8], which contains English prompts about country demonyms (e.g. "People from [Country] are known to be ___") with measured sentiment differences. Other relevant benchmarks: StereoSet and CrowS-Pairs provide collections of sentences and preferred continuations to test stereotypical associations [2]. We plan to use such pairs to compute bias scores and as intervention tasks.

### Hindi Bias Data

For Hindi, very few ready-made bias datasets exist, but we can adapt existing resources to avoid manual creation. One approach is to use the neutral/gendered occupation lexicon from Kirtane & Anand (2022) [5]. That work provides Hindi equivalents of gendered profession words, which we can plug into Hindi sentence templates (e.g. "यह [profession] काम कर रहा है" with gendered morphology) to test gender stereotypes. We will also explore publicly available Hindi sentiment and bias lexicons. Additionally, we note studies (Kumar et al. 2024) that have translated English bias sentences or tested Hindi-English MT for bias [6]. If needed, we can use Google Translate to create Hindi versions of select StereoSet/CrowS prompts, but only where such translations have been validated (e.g., by Ramesh et al. 2021's evaluation [6]). The aim is to have analogous English/Hindi evaluation sets so we can compare bias mechanisms across languages without manual annotation.

### Balanced Classes

Wherever possible, we will use balanced test sets (equal "biased" vs neutral/"counterstereotype" versions). For example, if using profession templates, we ensure equal male- vs female-stereotypical prompts. Similarly, nationality prompts will include both positive and negative countries as in prior work [8]. This avoids constructing new pairs manually and follows the model of past studies.

## 3. Model Selection

We will experiment on open-source Transformer models (all PyTorch-based) with ≤4B parameters, to balance interpretability and capacity:

### Decoder-only models

We will definitely include GPT-2 variants (e.g. GPT-2 Small/Medium, 124M–350M) and GPT-Neo (2.7B) from EleutherAI. GPT-2 is a standard interpretability testbed [1]. GPT-Neo 2.7B (PyTorch) is larger but still manageable on a high-end GPU. These autoregressive models allow straightforward activation patching via HookedTransformer/TransformerLens and generation of completions.

### Encoder-based models

To see if results generalize, we will also use an encoder-only model like BERT-base (110M) or RoBERTa-base. As Yang et al. (2025) did, we can probe both architectures for bias [11]. We will feed the same bias sentences (masked or as inputs) and examine BERT's layer outputs via probes. This ensures our plan covers both classes of transformers and aligns with existing literature (they found analogous head-level biases in BERT vs GPT [11]).

### Considered models

We will avoid very large or proprietary LLMs. All models will be from Hugging Face or EleutherAI, ensure PyTorch compatibility (no TF/Keras), and runnable in typical research settings. For example: `gpt2-medium`, `EleutherAI/gpt-neo-2.7B`, etc.

We will run our experiments on GPUs (up to A100) to allow fast patching/probing, but note that all chosen models are <4B parameters as requested.

## 4. Methods & Analysis

### 4.1 Activation Patching

#### Causal Tracing/Activation Patching

We will use activation patching to measure the importance of each model component to biased outputs [3]. Concretely, for a given input prompt, we will record the model's activations under a "biased" condition vs a neutral reference. Then, following prior work [3], we will replace (patch) individual activations (edges) from the biased run into the neutral run (or vice versa) and measure change in a bias metric. An "edge" here is a single tensor connection (e.g. output of one head feeding into the next layer) [10]. By computing the metric difference |ΔL| for each edge, we identify which edges cause the most bias change [3]. This follows the "Edge Attribution Patching" framework [3].

#### Attribution Patching

To scale the edge scoring, we will use Attribution Patching (Nanda 2023) which approximates the patching effect via gradients [3]. This allows ranking thousands of edges with only 2 forward and 1 backward pass, instead of one forward per edge [3]. We will adapt this to our bias metric L (e.g. logit difference between stereotypical vs neutral tokens). This identifies candidate bias-carrying edges efficiently. We will validate top edges with actual activation patching (slow but precise) for accuracy.

#### Module/Head Ablations

In addition to edges, we will directly ablate entire heads or MLPs: zeroing out a head's output and measuring bias change, as in Yang et al. (2025) [7][11]. This helps confirm whether certain heads are "biased heads." We expect to find, as they did, that only a few heads per layer exhibit a strong bias score. Masking out the top-biased heads should reduce bias (and we will measure any side-effects on general language ability).

### 4.2 Probing and Layer Analysis

#### Linear Probes

We will train simple linear classifiers (probes) on hidden-layer activations to predict if a prompt is "biased" or "neutral" [1]. For each layer's residual stream, we collect activations for a set of biased vs neutral prompts and train a logistic regression. High probe accuracy indicates that layer explicitly encodes bias. Prior work shows bias often becomes linearly separable in mid-to-late layers (e.g. "nearly perfect by layer 16 of GPT-2 large" [1]). We will replicate this analysis to see which layers carry gender/racial bias signals. The layers with highest AUC will be prime candidates for our patching/ablation experiments.

#### Logit Lens Analysis

We will apply the Logit Lens technique [2]: at each layer, project the current hidden state to vocabulary logits and measure whether stereotypical tokens dominate. For example, given "The doctor said that ___", does an intermediate layer already favor "he" over "she"? This layerwise decoding (as done by Prakash & Lee 2023 [2]) helps visualize bias emergence. We will quantify biases at each layer (via StereoSet-style scores [2]) and compare to probe results.

### 4.3 Bias Vector Identification

#### Mean Difference Vectors

Following Gupta et al. (2025) and prior "steering vector" work [1][4], we will compute a bias vector in hidden space as the difference between mean activations for biased vs neutral examples: v_bias = μ(biased) – μ(neutral). We will do this in the residual stream of the key layer identified by probes. This vector represents the direction in activation space that pushes outputs towards biased completions [1].

#### Enhancing Vectors

As Jørgensen (2023) notes [13], one can refine such directions by subtracting out any "background" mean to isolate the bias component. Concretely, we will first center activations over a broad neutral sample to estimate a general mean (bias from dataset) and then compute the residual direction for bias-specific examples [13]. We may also perform PCA on the set of biased-example activations to confirm if the top principal component aligns with our mean-difference vector (as an extra check).

#### Steering Experiments

We will test bias vectors by modifying activations during generation. For a chosen layer, adding (or subtracting) the bias vector should increase (or decrease) bias in the model's output. This is analogous to [11]'s steering experiments [1] and [1]'s classification steering [4]. For example, given a biased prompt ("Women are ___ to be nurses."), subtracting v_bias may yield a more neutral completion. We will quantitatively measure the effect of steering on stereotype scores (e.g. reduction in biased word probability) and qualitatively inspect sample outputs.

#### Gradient-based Feature Search (Novel)

As a novel contribution, we plan to search for an optimal bias direction by maximizing the bias metric over a synthetic direction. Using a small set of prompts, we can apply gradient ascent in activation space to find a direction that most increases the bias score (e.g. logit diff) when added. This automatically produces a "bias steering vector." We will compare this learned vector with the mean-difference vector to see if they coincide. This method extends prior mean-based steering [1] by using the model's gradients for fine-tuning the direction.

### 4.4 Ablation and Generalization Tests

#### Component Ablation

After identifying candidate bias edges/heads, we will disable them (zeroing activations) and measure both bias reduction and impact on other tasks. Following Chandna et al. (2025), we expect that removing bias components will reduce biased completions, but may also degrade unrelated abilities [7]. We will test effects on a small generality set (e.g. language modeling perplexity or factual QA) to quantify trade-offs.

#### Data Perturbations

To test stability, we will vary the prompts (e.g. swap synonyms, change syntax) and repeat the analysis. Prior work found that which edges score as "important" can shift with lexicon changes [7]. We will report which findings are robust (same heads/layers repeatedly flagged) vs which are brittle, to understand how general the bias circuits are.

#### Fine-tuning Variation

Optionally, we may fine-tune a model on a small bias-related corpus (as a probe). Chandna et al. (2025) fine-tuned GPT-2 on a "positive bias" dataset and on Shakespeare text to see how the bias circuits adapt [10]. If resources permit, we will similarly fine-tune (e.g. on gender-neutral data) to test if bias edges shift, but we emphasize interpretability of the original pretrained model as the main goal.

## 5. Tools and Implementation

### Frameworks

We will use Python with PyTorch, utilizing HuggingFace Transformers for model loading and TransformerLens (HookedTransformer) for easy activation hooking and patching [1]. TransformerLens allows us to attach hooks at specific heads, MLPs, and the residual stream, which is essential for patching interventions (e.g. `hook_resid_post` and `attn.hook_z`) [1]. All code will be in PyTorch (no TensorFlow/Keras) as requested.

### Bias Metrics

For each prompt completion, we will compute a bias score. This could be the log-probability gap between stereotype-consistent vs -inconsistent tokens, or a binary classification measure (AUC) between biased/neutral examples. For template tasks we might use the same "sum of MS vs FS probabilities" method as Chandna et al. [10] (assigning predicted next token to male/female sets) or StereoSet's likelihood scoring. We will automate these metrics so that patching (which yields new logits) can be evaluated.

### Evaluation Workflow

A typical experiment flow will be: select a bias prompt, get model output and activations; apply patching or vector addition; get new output; compute bias metric change. We will script this in a modular way so that scanning through all heads/layers is systematic. We will also use statistical tests to confirm significance (as done by Yang et al. for head scores [11]).

### Reproducibility

All models and data are public. We will release code notebooks for our study. Following the style of previous work, each key result will cite the method used (e.g. which layer was patched, which metric used) for transparency.

## 6. Expected Findings & Novelty

### Localization of Bias

Based on prior reports, we expect to find that gender/racial bias is not diffuse but concentrated in certain layers (likely mid-to-late layers) and in a subset of heads/edges [1][7]. We will quantify how many edges vs heads need to be patched/ablated to significantly alter bias. A novel outcome would be identifying specific circuits (combinations of heads and MLPs across layers) that form a "bias network."

### Bias Vectors

We will likely confirm that a linear bias vector exists. Our innovation will be in how we compute and validate it. By combining mean difference with gradient refinement, we may uncover more precise "directional" interventions than prior work. We will test if such vectors transfer across prompts (a direction that reduces bias in one sentence should work for others of the same type).

### Interpretability Tools Applied to Bias

While many studies perform black-box debiasing, our approach is white-box. We will systematically compare activation-patching results to simpler masking (head removal) to check consistency. If successful, our results can guide new mitigation: e.g. "target head X in layer Y for debiasing."

### English vs Hindi

We anticipate that Hindi bias circuits may share characteristics with English (especially gender is a linguistic category), but unique features (Hindi's grammatical gender) could lead to different layers/heads being involved. If our analysis shows consistent patterns, that would be novel insight. If it fails to find "nice" bias vectors in Hindi, that also tells us about language differences.

### Prototype Contributions

We will deliver a prioritized list of (layer, head) or (layer, edge) components implicated in bias. We will also provide visualizations (e.g. bar charts of patching impact per layer) and examples of steering results. Together, this constitutes a "bias mechanistic map" of the model. Any newly observed phenomenon (e.g. a particular attention pattern that conveys bias) will be highlighted.

## 7. Summary of Methods (Checklist)

- **Data**: Use existing English bias datasets (e.g. lists from Bolukbasi 2016 [10], StereoSet/CrowS) and analogous Hindi sources [5][6]. Ensure balanced biased vs neutral samples.
- **Models**: GPT-2 small/medium/neo (all PyTorch, ≤2.7B)
- **Metrics**: Stereotype score on completions; token-prob differences for target words; probe accuracy (AUC) [1].
- **Interpretation tools**:
  - **Probing**: train layerwise logistic classifiers [1].
  - **Patching**: do causal activation patching on edges (activation-intervened forward passes) [3] and efficient attribution patching [3].
  - **Heads**: zeroing out/masking attention heads and MLPs as ablations [7][11].
  - **Logit Lens**: decode intermediate hidden states to logits [2].
  - **Bias vectors**: compute mean-difference and test steering (subtract/add) [1][4], and refine via gradient optimization.
- **Analysis**: Identify which components cause large metric changes. Compare across biases (gender vs race). Test transfer across prompts/perturbations (e.g. rewording). Evaluate side-effects on general tasks (language modeling or GLUE) for targeted ablative mitigations.
- **Tools**: TransformerLens/PyTorch for hooking activations [1]; HuggingFace for model weights; scikit-learn for probes; standard evaluation scripts for bias benchmarks.

## 8. References

<!-- We will ground our plan in extensive literature. For brevity, key references include Bolukbasi et al. (2016) on gendered word lists [10]; Narayanan et al. (2023) on nationality bias [8]; Yang et al. (2025) on biased attention heads in BERT/GPT [7][11]; Chandna et al. (2025) on edge localization of bias [10][7]; Yu & Ananiadou (2025) on gender "neurons" [12]; Gupta et al. (2025) on activation steering vectors [1]; Nanda et al. (2024) on attribution patching [3]; and Prakash & Lee (2023) on layerwise bias via Logit Lens [2]. We also draw on established bias benchmarks (e.g. StereoSet [2], CrowS-Pairs) and Hindi bias studies [5][6]. -->

In all, our plan weaves mechanistic interpretability best practices into the domain of bias analysis. By thoroughly surveying ~100+ works in bias/fairness and interpretability, we ensure each experimental choice is justified. The expected outcome is a fine-grained map of how bias flows through model components, and candidate "bias vectors" for targeted interventions – a novel interpretability contribution that goes beyond black-box mitigation to explaining why a model is biased.

### Reference Links

1. [Activation Steering for Bias Mitigation: An Interpretable Approach to Safer LLMs](https://arxiv.org/html/2508.09019v1)

2. [Layered Bias: Interpreting Bias in Pretrained Large Language Models](https://aclanthology.org/2023.blackboxnlp-1.22.pdf)

3. [Edge Attribution Patching and Attribution Patching](https://aclanthology.org/2024.blackboxnlp-1.25.pdf)

4. [ICML No Training Wheels: Steering Vectors for Bias Correction at Inference Time](https://icml.cc/virtual/2025/49557)

5. [Mitigating Gender Stereotypes in Hindi and Marathi - ACL Anthology](https://aclanthology.org/2022.gebnlp-1.16/)

6. [Akal Badi ya Bias: An Exploratory Study of Gender Bias in Hindi Language Technology](https://arxiv.org/html/2405.06346v1)

7. [Bias in LLMs: Mechanistic Interpretability](https://www.emergentmind.com/papers/2506.05166)

8. [Nationality Bias in Text Generation](https://arxiv.org/abs/2302.02463)

9. [Interpreting Bias in Large Language Models: A Feature-Based Approach](https://arxiv.org/html/2406.12347v1)

10. [Dissecting Bias in LLMs: A Mechanistic Interpretability Perspective](https://ar5iv.labs.arxiv.org/html/2506.05166)

11. [Bias in LLMs: Mechanistic Interpretability](https://aclanthology.org/2025.trustnlp-main.18.pdf)

12. [Understanding and Mitigating Gender Bias in LLMs via Interpretable Neuron Editing](https://arxiv.org/abs/2501.14457)

13. [Imperial Dissertation - Jørgensen](https://ojorgensen.github.io/assets/pdfs/Imperial_Dissertation.pdf)

