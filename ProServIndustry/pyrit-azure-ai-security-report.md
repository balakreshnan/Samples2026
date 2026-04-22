# Testing Azure AI Foundry / Azure OpenAI Endpoints for AI Security with Microsoft PyRIT

**Research date:** 2026-04-22
**Confidence overall:** HIGH (primary sources: Microsoft Learn, Microsoft Security Blog, microsoft/PyRIT repo, arXiv paper by PyRIT authors)

---

## Executive Summary

Microsoft's **PyRIT (Python Risk Identification Tool for generative AI)** is the officially-sanctioned, MIT-licensed framework for red-teaming Azure OpenAI and Azure AI Foundry deployments. Built and maintained by the **Microsoft AI Red Team (AIRT)**, PyRIT powers Microsoft's own internal red-team engagements (100+ operations including Copilots and Phi-3) and is **directly integrated into Azure AI Foundry as the "AI Red Teaming Agent" (preview)**, exposed through the `azure-ai-evaluation` Python SDK's `RedTeam` class ([Run AI Red Teaming Agent locally - Microsoft Learn](https://learn.microsoft.com/en-us/azure/foundry/how-to/develop/run-scans-ai-red-teaming-agent); [Announcing PyRIT - Microsoft Security Blog](https://www.microsoft.com/en-us/security/blog/2024/02/22/announcing-microsofts-open-automation-framework-to-red-team-generative-ai-systems/)).

For Azure OpenAI endpoints, the recommended approach is:

1. Use **`OpenAIChatTarget`** as the PyRIT prompt target (not a separate "AzureOpenAI" class — a single class handles both OpenAI and Azure OpenAI endpoints via the deployment-path URL) ([PyRIT OpenAI Chat Target documentation](https://deepwiki.com/Azure/PyRIT/5.2-openai-chat-and-completion-targets); [Stack Overflow verified example](https://stackoverflow.com/questions/79548081/how-to-call-azureopenai-api-with-pyrit)).
2. Load built-in datasets (HarmBench, AdvBench, XSTest, TDC23, JailbreakBench, SORRY-Bench, and ~30 others) via `SeedDatasetProvider.fetch_datasets_async()` ([PyRIT Loading Built-in Datasets](https://microsoft.github.io/PyRIT/code/datasets/loading-datasets/)).
3. Drive attacks with **`PromptSendingOrchestrator`** (single-turn), **`RedTeamingOrchestrator`** (multi-turn), or **`CrescendoOrchestrator`** (gradual escalation) ([PyRIT Orchestrators DeepWiki](https://deepwiki.com/Azure/PyRIT/9-development-guide)).
4. Score results with **`AzureContentFilterScorer`** and **`PromptShieldScorer`** (which call Azure AI Content Safety `/text:analyze` and `/text:shieldPrompt`) or the LLM-as-judge **`SelfAskTrueFalseScorer` / `SelfAskLikertScorer`** ([PyRIT Azure scorers](https://deepwiki.com/Azure/PyRIT/7.3-azure-and-api-based-scorers)).

**Critical caveat:** Microsoft allows red-teaming of **your own** Azure OpenAI deployments. Automated adversarial prompts directed at third-party APIs violate usage policies. For testing that must bypass content filters, request modification via Microsoft's Limited Access form ([Microsoft Foundry red-teaming guidance](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/red-teaming)).

---

## 1. What is PyRIT? Who maintains it? What is its purpose?

PyRIT ("Python Risk Identification Toolkit for generative AI") is an **open-source, MIT-licensed** Python framework released by the **Microsoft AI Red Team** on **22 February 2024** by Ram Shankar Siva Kumar (AIRT's "Data Cowboy") ([Announcing PyRIT - Microsoft Security Blog](https://www.microsoft.com/en-us/security/blog/2024/02/22/announcing-microsofts-open-automation-framework-to-red-team-generative-ai-systems/)).

The official repository is **[microsoft/PyRIT](https://github.com/microsoft/PyRIT)** (recently migrated from `Azure/PyRIT`). As of April 2026 the repo has 3.7k stars, MIT license, and is at version 0.13.0 ([microsoft/PyRIT GitHub](https://github.com/microsoft/PyRIT)).

**Purpose (from the authors' peer paper):**
> "A model- and platform-agnostic tool that enables red teamers to probe for and identify novel harms, risks, and jailbreaks in multimodal generative AI models. … The Microsoft AI Red Team has successfully utilized PyRIT in 100+ red teaming operations of GenAI models, including for Copilots and the release of Microsoft's Phi-3 models." ([Lopez Munoz et al., arXiv 2410.02828](https://arxiv.org/html/2410.02828v1))

**Core composable components**: targets (how to send prompts), converters (how to transform prompts — e.g., Base64, leetspeak), datasets (seed prompts), orchestrators (attack strategies), scorers (how to judge responses), memory (SQLite/Azure SQL for persistence) ([PyRIT documentation home](https://microsoft.github.io/PyRIT/)).

**Confidence:** HIGH — authorship, license, and purpose corroborated by the official Microsoft Security blog, the GitHub repository, and the peer-published arXiv paper authored by the PyRIT team.

---

## 2. Is PyRIT appropriate for testing Azure OpenAI / Azure AI Foundry? Microsoft's official recommendation

**Yes — and Microsoft officially recommends it.** Two pieces of direct evidence:

1. Microsoft's Foundry documentation states explicitly:
   > "The AI red teaming capabilities of Microsoft's open-source framework for Python Risk Identification Tool (PyRIT) are integrated directly into Microsoft Foundry." ([Run AI Red Teaming Agent locally - Microsoft Learn](https://learn.microsoft.com/en-us/azure/foundry/how-to/develop/run-scans-ai-red-teaming-agent))

2. The **`azure-ai-evaluation[redteam]`** pip extra installs PyRIT as a hard dependency for the `RedTeam` class ([azure.ai.evaluation.red_team.RedTeam API reference](https://learn.microsoft.com/en-us/python/api/azure-ai-evaluation/azure.ai.evaluation.red_team.redteam?view=azure-python); [azure-ai-evaluation PyPI](https://pypi.org/project/azure-ai-evaluation/)).

Microsoft's guidance document ["Planning red teaming for large language models"](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/red-teaming) explicitly recommends testing the LLM base model via API endpoint *and* your application via UI, iteratively, with and without mitigations — which is the exact workflow PyRIT automates.

**Confidence:** HIGH.

---

## 3. PyRIT Targets for Azure OpenAI / Azure AI Foundry

A quick note on nomenclature in the user's question: **there is no `AzureOpenAIGPT4OChatTarget` class in current PyRIT.** Azure OpenAI endpoints are handled through the generic `OpenAIChatTarget` by pointing `endpoint` at the Azure deployment URL. `AzureMLChatTarget` existed historically but has been largely superseded. Details below.

| Target class | Used for | Status |
|---|---|---|
| **`OpenAIChatTarget`** (`pyrit.prompt_target.OpenAIChatTarget`) | Primary target for **both OpenAI and Azure OpenAI** chat-completion endpoints. Supports multimodal (text, image, audio), JSON-schema responses, multi-turn. | Current / primary |
| **`OpenAICompletionTarget`** | Legacy `/completions` endpoint (single-turn). | Current (legacy use only) |
| **`OpenAIResponseTarget`** (`openai_response_target.py`) | Responses API / tool-calling for o1/o3/GPT-5-class models. | Current |
| **`OpenAIRealtimeTarget`** (`openai_realtime_target.py`) | Realtime/WebSocket voice-audio endpoints. | Current |
| **`HTTPTarget` / custom `PromptChatTarget`** | Arbitrary HTTP endpoints, AML managed online endpoints, MaaS, custom model deployments. | Current |
| `AzureMLChatTarget` | Historical Azure ML endpoint target. | Largely superseded by `HTTPTarget` and `OpenAIChatTarget` — check `pyrit.prompt_target.__init__` in your installed version |

Source: [OpenAI Chat and Completion Targets (DeepWiki, sourced from pyrit source code)](https://deepwiki.com/Azure/PyRIT/5.2-openai-chat-and-completion-targets); corroborated by [Stack Overflow verified AzureOpenAI-with-PyRIT example](https://stackoverflow.com/questions/79548081/how-to-call-azureopenai-api-with-pyrit).

The verified Azure OpenAI endpoint URL pattern for `OpenAIChatTarget`:
```
https://<openaiName>.openai.azure.com/openai/deployments/<deploymentName>/chat/completions?api-version=<api-version>
```

`OpenAIChatTarget` accepts either `api_key=...` (key auth) or `api_key=get_azure_token_provider("https://cognitiveservices.azure.com/.default")` for **Entra ID** (Azure AD) token-based auth ([PyRIT Azure Content Safety scorer example](https://github.com/microsoft/PyRIT/blob/main/doc/code/scoring/1_azure_content_safety_scorers.py)).

**Confidence:** HIGH for `OpenAIChatTarget` being the canonical Azure OpenAI target; MODERATE for the current status of `AzureMLChatTarget` (refactoring in the PyRIT repo is active).

---

## 4. PyRIT Built-in Datasets

PyRIT ships with a large catalog of **seed datasets** accessible via `SeedDatasetProvider.fetch_datasets_async()` (or the older module-level fetch functions). Some are local YAML/JSON; others are remote loaders that fetch from HuggingFace or GitHub (e.g., the HarmBench CSV is fetched directly from `centerforaisafety/HarmBench`) ([PyRIT HarmBench dataset loader source](https://github.com/microsoft/PyRIT/blob/main/pyrit/datasets/seed_datasets/remote/harmbench_dataset.py)).

Per the [PyRIT Loading Built-in Datasets documentation](https://microsoft.github.io/PyRIT/code/datasets/loading-datasets/), the shipped catalog includes:

| Dataset | Paper / Source | Focus |
|---|---|---|
| **HarmBench** | Mazeika et al., 2024 | 504 harmful behaviors, 7 semantic categories (cybercrime, CBRN, copyright, misinformation, harassment, illegal, general) |
| **AdvBench** | Zou et al., 2023 (2307.15043) | 500 harmful behaviors (universal / transferable adversarial attacks) |
| **JailbreakBench (JBB)** | Chao et al., 2024 | "Just-Be-Bad" behaviors across multiple harm categories |
| **XSTest** | Röttger et al., 2023 | Exaggerated safety / false refusal tests |
| **TDC23** | Mazeika et al., 2023 | Trojan Detection Challenge 2023 red-team prompts |
| **AILuminate** | Vidgen et al., 2024 | MLCommons AILuminate safety benchmark |
| **ALERT** | Tedeschi et al., 2024 | LLM safety red-team benchmark |
| **Aegis 2.0** | Ghosh et al., 2025 | Diverse AI safety dataset for guardrail alignment |
| **BeaverTails** | Ji et al., 2023 | Human-preference safety alignment |
| **Do Anything Now (DAN)** | Shen et al., 2023 | In-the-wild jailbreak prompts |
| **Do-Not-Answer** | Wang et al., 2023 | Prompts models should refuse |
| **HarmfulQA** | Chu et al., 2023 | Harmful question answering |
| **SORRY-Bench** | Xie et al., 2024 | Safety refusal benchmark |
| **SALAD-Bench** | Li et al., 2024 | Comprehensive LLM safety benchmark |
| **SimpleSafetyTests** | Vidgen et al., 2023 | Minimum-viable safety checks |
| **SOSBench** | Jiang et al., 2025 | Science-of-safety benchmark |
| **OR-Bench** | Cui et al., 2024 | Over-refusal benchmark |
| **PKU-SafeRLHF** | Ji et al., 2024 | Safety RLHF dataset |
| **ToxicChat** | Lin et al., 2023 | Toxicity in chat |
| **LLM-LAT** | Sheshadri et al., 2024 | Latent adversarial training prompts |
| **MedSafetyBench** | Han et al., 2024 | Medical safety |
| **EquityMedQA** | Pfohl et al., 2024 | Health equity / bias |
| **CBT-Bench** | Zhang et al., 2024 | Cognitive-behavioral therapy assistance |
| **DarkBench** | Apart Research, 2025 | Dark design patterns |
| **Multilingual Alignment (Prism)** | Aakanksha et al., 2024 | Cross-language alignment |
| **Multilingual Vulnerabilities** | Tang et al., 2025 | Multilingual jailbreaks |
| **Transphobia Awareness** | Scheuerman et al., 2025 | Bias re: transgender topics |
| **Red Team Social Bias** | Taylor, 2024 | Social bias probes |
| **PromptIntel** | Roccia, 2024 | Prompt-injection intelligence |
| **VLSU** | Palaskar et al., 2025 | Vision-language safety |
| Plus jailbreak templates from **garak** (Derczynski et al., 2024) | AART templates, and more | Template-based jailbreaks |

**DecodingTrust** (Wang et al., 2023) is a prominent benchmark for trustworthiness dimensions; it is *not* currently in PyRIT's built-in dataset list per the official documentation (as of April 2026). It can be plugged in as a custom `SeedDataset`.

**Example use:**
```python
from pyrit.datasets import SeedDatasetProvider

# List all built-in datasets
all_names = await SeedDatasetProvider.get_all_dataset_names_async()

# Fetch specific datasets
datasets = await SeedDatasetProvider.fetch_datasets_async(
    dataset_names=["harmbench", "xstest"]
)
for ds in datasets:
    for seed in ds.seeds:
        print(seed.value)
```

Source: [PyRIT Loading Built-in Datasets](https://microsoft.github.io/PyRIT/code/datasets/loading-datasets/); dataset loader source [harmbench_dataset.py](https://github.com/microsoft/PyRIT/blob/main/pyrit/datasets/seed_datasets/remote/harmbench_dataset.py); [HarmBench dataset homepage](https://www.harmbench.org/explore); [AdvBench Hugging Face dataset card](https://huggingface.co/datasets/walledai/AdvBench).

**Confidence:** HIGH for dataset list (direct Microsoft docs + source code); MODERATE for DecodingTrust's explicit inclusion (not found in current catalog — flag as absent/user-configurable).

---

## 5. Relationship: PyRIT vs Azure AI Content Safety / Prompt Shields / Foundry's "AI firewall"

These are **complementary**: PyRIT is the **offensive/testing** tool; Content Safety/Prompt Shields are the **defensive** runtime filters.

| Component | Role | API / endpoint |
|---|---|---|
| **Azure AI Content Safety – harm categories** | Defensive runtime moderation for hate, sexual, self-harm, violence (severity 0–7) | `POST /contentsafety/text:analyze` |
| **Prompt Shields** | Defensive detection of direct (jailbreak/UPIA) + indirect (XPIA / cross-domain) prompt injection | `POST /contentsafety/text:shieldPrompt?api-version=2024-09-01` |
| **Azure OpenAI Content Filters (RAI Policy)** | Request/response filter policies configurable via the RAI Policy Management API. Can optionally enable `jailbreak` and `indirect_attack` filters | Azure Management API `/raiPolicies/{name}` |
| **PyRIT `AzureContentFilterScorer`** | Offensive: uses Content Safety to score whether an assistant *response* crossed a harm threshold | Calls `/text:analyze` |
| **PyRIT `PromptShieldScorer`** | Offensive: uses Prompt Shields to determine whether a *crafted* prompt is classified as an attack (useful for adversarial-testing the shield itself) | Calls `/text:shieldPrompt` |

Key Microsoft sources: [Prompt Shields in Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection); [Azure AI Content Safety documentation](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/); [Azure AI announces Prompt Shields (Microsoft blog, Mar 2024)](https://thewindowsupdate.com/2024/03/28/azure-ai-announces-prompt-shields-for-jailbreak-and-indirect-prompt-injection-attacks/); [PyRIT Azure and API-Based Scorers](https://deepwiki.com/Azure/PyRIT/7.3-azure-and-api-based-scorers); [Prompt Shield and Attack Prevention (azure-openai-samples DeepWiki)](https://deepwiki.com/Azure/azure-openai-samples/4.2-prompt-shield-and-attack-prevention).

A production red-team workflow typically uses PyRIT to:
1. **Probe the filters** (Prompt Shields, content filters) — did the adversarial prompt get blocked?
2. **Probe the model behind the filters** — if the filter lets it through, did the model comply?
3. Use `AzureContentFilterScorer` to programmatically score how harmful the *response* was even if the filter did not block it.

**Confidence:** HIGH.

---

## 6. Recommended End-to-End Workflow

The Microsoft-recommended, PyRIT-native workflow:

```
┌────────────────────────────────────────────────────────────────────┐
│ 1. AUTHENTICATE                                                    │
│    az login  →  DefaultAzureCredential / get_azure_token_provider  │
│    OR         API key in ~/.pyrit/.env                             │
│                                                                    │
│ 2. INITIALIZE PyRIT                                                │
│    initialize_pyrit_async(memory_db_type=IN_MEMORY)                │
│                                                                    │
│ 3. CONFIGURE TARGET (Azure OpenAI)                                 │
│    OpenAIChatTarget(endpoint=..., api_key=...)                     │
│                                                                    │
│ 4. LOAD DATASET                                                    │
│    SeedDatasetProvider.fetch_datasets_async(["harmbench"])         │
│                                                                    │
│ 5. RUN ORCHESTRATOR                                                │
│    PromptSendingOrchestrator (single-turn baseline)                │
│    RedTeamingOrchestrator     (multi-turn adversarial)             │
│    CrescendoOrchestrator      (gradual escalation)                 │
│    TreeOfAttacksOrchestrator  (TAP search)                         │
│                                                                    │
│ 6. SCORE                                                           │
│    AzureContentFilterScorer  (harm categories)                     │
│    PromptShieldScorer        (jailbreak / injection detection)     │
│    SelfAskTrueFalseScorer    (LLM-as-judge)                        │
│                                                                    │
│ 7. REPORT                                                          │
│    orchestrator.print_conversations_async() or export from memory  │
│    (Attack Success Rate per risk category / complexity)            │
└────────────────────────────────────────────────────────────────────┘
```

Source: [PyRIT documentation home](https://microsoft.github.io/PyRIT/); [PyRIT Zero to Red Team tutorial](https://www.aisecurityinpractice.com/attack-and-red-team/pyrit-zero-to-red-team/); [BreakPoint Labs walkthrough](https://breakpoint-labs.com/ai-red-teaming-playground-labs-setup-and-challenge-1-walkthrough-with-pyrit/).

**Confidence:** HIGH.

---

## 7. Minimal Runnable Example

The following code is **synthesized from and cross-checked against** the [verified Stack Overflow answer](https://stackoverflow.com/questions/79548081/how-to-call-azureopenai-api-with-pyrit), [PyRIT Azure Content Safety scorer example](https://github.com/microsoft/PyRIT/blob/main/doc/code/scoring/1_azure_content_safety_scorers.py), and [PyRIT documentation](https://microsoft.github.io/PyRIT/). Treat it as a starting template — exact class/method names may evolve per PyRIT release (pin your version; see version-compatibility note in the docs).

```python
"""
PyRIT red-team scan against your own Azure OpenAI deployment
using a HarmBench-style seed dataset + Azure Content Safety scoring.

Prereqs:
  pip install pyrit
  ~/.pyrit/.env with:
      AZURE_OPENAI_CHAT_ENDPOINT=https://<name>.openai.azure.com/openai/deployments/<deploy>/chat/completions?api-version=2024-10-21
      AZURE_OPENAI_CHAT_KEY=<key>                # OR use Entra ID (az login)
      AZURE_CONTENT_SAFETY_API_ENDPOINT=https://<cs-name>.cognitiveservices.azure.com/
      AZURE_CONTENT_SAFETY_API_KEY=<cs-key>      # OR use Entra ID
"""
import asyncio
import os

from pyrit.setup import IN_MEMORY, initialize_pyrit_async
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.datasets import SeedDatasetProvider
from pyrit.score.float_scale.azure_content_filter_scorer import AzureContentFilterScorer
from pyrit.auth import get_azure_token_provider


async def main() -> None:
    # 1. Initialize PyRIT in-memory (SQLite or Azure SQL also supported)
    await initialize_pyrit_async(memory_db_type=IN_MEMORY)

    # 2. Target: Azure OpenAI deployment. Two auth options shown.
    # Option A: API key (simpler for local dev)
    target = OpenAIChatTarget(
        endpoint=os.environ["AZURE_OPENAI_CHAT_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_CHAT_KEY"],
    )
    # Option B: Entra ID (recommended for production)
    # target = OpenAIChatTarget(
    #     endpoint=os.environ["AZURE_OPENAI_CHAT_ENDPOINT"],
    #     api_key=get_azure_token_provider(
    #         "https://cognitiveservices.azure.com/.default"
    #     ),
    # )

    # 3. Load HarmBench seed dataset (504 harmful behaviors, 7 categories)
    datasets = await SeedDatasetProvider.fetch_datasets_async(
        dataset_names=["harmbench"]
    )
    # For a smoke test, take the first 10 prompts
    prompt_list = [seed.value for ds in datasets for seed in ds.seeds[:10]]

    # 4. Azure Content Safety scorer: assigns a 0..1 float per harm category
    #    (hate, sexual, self_harm, violence)
    content_safety_scorer = AzureContentFilterScorer(
        api_key=os.environ["AZURE_CONTENT_SAFETY_API_KEY"],
        endpoint=os.environ["AZURE_CONTENT_SAFETY_API_ENDPOINT"],
    )

    # 5. Run the attack: single-turn PromptSendingOrchestrator
    orchestrator = PromptSendingOrchestrator(
        objective_target=target,
        auxiliary_scorers=[content_safety_scorer],
    )
    await orchestrator.send_prompts_async(prompt_list=prompt_list)

    # 6. Report: print each conversation + scores
    await orchestrator.print_conversations_async()


if __name__ == "__main__":
    asyncio.run(main())
```

**Using the azure-ai-evaluation RedTeam wrapper** (Microsoft's higher-level, Foundry-integrated path — simpler, less customizable):

```python
# pip install "azure-ai-evaluation[redteam]"
import os
from azure.identity import DefaultAzureCredential
from azure.ai.evaluation.red_team import RedTeam, RiskCategory, AttackStrategy

azure_ai_project = os.environ["AZURE_AI_PROJECT"]  # https://<acct>.services.ai.azure.com/api/projects/<project>

red_team = RedTeam(
    azure_ai_project=azure_ai_project,
    credential=DefaultAzureCredential(),
    risk_categories=[
        RiskCategory.Violence,
        RiskCategory.HateUnfairness,
        RiskCategory.Sexual,
        RiskCategory.SelfHarm,
    ],
    num_objectives=5,
)

azure_openai_config = {
    "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
    "api_key": os.environ["AZURE_OPENAI_KEY"],      # omit for Entra ID
    "azure_deployment": os.environ["AZURE_OPENAI_DEPLOYMENT"],
}

result = await red_team.scan(
    target=azure_openai_config,
    scan_name="AOAI-baseline-scan",
    attack_strategies=[
        AttackStrategy.EASY,      # Base64, Flip, Morse
        AttackStrategy.MODERATE,  # Tense
        AttackStrategy.DIFFICULT, # Composition of Tense + Base64
    ],
    output_path="redteam-scan.json",
)
```

Verbatim code pattern from [Run AI Red Teaming Agent locally - Microsoft Learn](https://learn.microsoft.com/en-us/azure/foundry/how-to/develop/run-scans-ai-red-teaming-agent).

**Confidence:** HIGH for the RedTeam example (copied from Microsoft's official docs); MODERATE for the raw PyRIT example (synthesized from multiple verified sources; class signatures change between PyRIT 0.x releases — pin your version).

---

## 8. Appropriateness Caveats: Licensing, ToS, Data Residency, Abuse Monitoring, Responsible Disclosure

**Licensing.** PyRIT is MIT-licensed ([microsoft/PyRIT LICENSE](https://github.com/microsoft/PyRIT)). Many built-in datasets (AdvBench MIT, HarmBench MIT) carry permissive licenses; check each dataset's own license before redistribution (HarmBench and AdvBench both require citation).

**Terms of service / scope.** Microsoft is explicit that you may red-team **your own** Azure OpenAI deployment:
- > "You can begin by testing the base model to understand the risk surface … (Testing is usually done through an API endpoint.)" — [Microsoft Foundry red-teaming guidance](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/red-teaming)
- > "PyRIT is designed for testing your own applications and safety layers … It is not for attempting to jailbreak third-party APIs directly. Firing automated adversarial prompts at a public API violates most providers' usage policies … This applies to OpenAI, Azure OpenAI, Google, Anthropic, and others. … Azure OpenAI customers can request content filter removal specifically for authorised testing." — [AI Security in Practice tutorial (independent)](https://www.aisecurityinpractice.com/attack-and-red-team/pyrit-zero-to-red-team/)

**Abuse monitoring.** Azure OpenAI runs abuse monitoring on customer traffic by default. Red-team runs producing large volumes of harmful-looking prompts *can* trigger monitoring. Large organisations typically (a) use a dedicated red-team subscription/tenant, (b) apply for **Modified Abuse Monitoring** and **Modified Content Filtering** through Microsoft's Limited Access application (the same form used to reduce logging), and (c) time-box scans. (Note: this report did not directly retrieve the current Limited Access URL; verify on Microsoft Learn under "Limited access to Azure OpenAI" before submitting.)

**Content filters during testing.** To meaningfully test the *model*, you often need content filters minimized. The [RaffaChiaverini implementation README](https://github.com/RaffaChiaverini/Azure-AI-Red-Teaming-Agent---Implementation) warns: *"To clearly observe the agent's full functionality … it is recommended to minimize the content filtering settings on your Azure OpenAI deployment. During testing, Microsoft may block or sanitize certain requests using Prompt Shield."* Follow Microsoft's sanctioned path (RAI Policy API) rather than trying to bypass filters silently.

**Data residency.** PyRIT's memory (SQLite or Azure SQL) holds adversarial prompts and model responses, which may contain content that violates organisational policy if exfiltrated. Store memory in the same region as your production deployment; treat the memory DB as sensitive. For `AI Red Teaming Agent` (Foundry preview), region support is limited to **East US 2, France Central, Sweden Central, Switzerland West, North Central US** ([Microsoft Learn](https://learn.microsoft.com/en-us/azure/foundry/how-to/develop/run-scans-ai-red-teaming-agent)).

**Responsible disclosure.** Microsoft maintains a bug-bounty program that *includes* AI-related vulnerabilities in Microsoft products (Copilot, Azure OpenAI Service, etc.). If PyRIT surfaces a *platform*-level vulnerability (not an issue with your own deployment), report via [msrc.microsoft.com](https://msrc.microsoft.com/). PyRIT's own security-contact process is in the repo's `SECURITY.md` ([microsoft/PyRIT](https://github.com/microsoft/PyRIT)).

**Confidence:** MODERATE-HIGH. The high-level caveats are directly supported by Microsoft docs; specific URLs for the Limited Access form / abuse-monitoring modification were not re-verified in this pass and should be double-checked before applying.

---

## 9. Alternatives and Complements

| Offering | Relationship to PyRIT | Use when |
|---|---|---|
| **Azure AI Foundry "AI Red Teaming Agent" (preview)** | **PyRIT is the engine**; exposed via `azure-ai-evaluation[redteam]` and the Foundry portal. | You want a quick, managed scan with Microsoft-curated attack seeds + standardised scorecard; limited to single-turn text and four+ risk categories ([Microsoft Learn](https://learn.microsoft.com/en-us/azure/foundry/how-to/develop/run-scans-ai-red-teaming-agent)). |
| **`azure-ai-evaluation` SDK (`RedTeam` class)** | Higher-level Python wrapper around PyRIT. Adds `RiskCategory`, `AttackStrategy.EASY/MODERATE/DIFFICULT`, `SupportedLanguages`, `.scan(...)`, JSON scorecard output. | You want Foundry-portal integration, managed attack-objective generation, and an ASR scorecard without assembling orchestrators yourself ([RedTeam class docs](https://learn.microsoft.com/en-us/python/api/azure-ai-evaluation/azure.ai.evaluation.red_team.redteam?view=azure-python); [azure-ai-evaluation PyPI](https://pypi.org/project/azure-ai-evaluation/)). |
| **`azure-ai-evaluation` safety evaluators** (`ViolenceEvaluator`, `SexualEvaluator`, `SelfHarmEvaluator`, `HateUnfairnessEvaluator`, `IndirectAttackEvaluator`, `ProtectedMaterialEvaluator`) | Complementary — these evaluate a *dataset of prompt-response pairs* (possibly generated by PyRIT). | Batch evaluation / regression-style safety measurement in CI/CD ([azure-ai-evaluation PyPI](https://pypi.org/project/azure-ai-evaluation/)). |
| **Prompt Shield evaluation** (direct `/text:shieldPrompt` calls) | PyRIT's `PromptShieldScorer` wraps this. | You want to measure the Shield's own detection rate against your adversarial corpus ([Prompt Shields docs](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection)). |
| **CoPyRIT GUI** | Official Microsoft GUI front-end for PyRIT (human-led red teaming). | Interactive exploration, non-engineer stakeholders ([PyRIT home](https://microsoft.github.io/PyRIT/)). |
| **AI Red-Teaming Playground Labs** ([github.com/microsoft/AI-Red-Teaming-Playground-Labs](https://github.com/microsoft/AI-Red-Teaming-Playground-Labs)) | Training target system; not a PyRIT alternative. | Skills practice before testing production systems ([BreakPoint Labs walkthrough](https://breakpoint-labs.com/ai-red-teaming-playground-labs-setup-and-challenge-1-walkthrough-with-pyrit/)). |
| **garak, Giskard, promptfoo, Counterfit** | Independent red-team / eval frameworks; PyRIT even ingests garak jailbreak templates. | Cross-tool coverage; defense-in-depth; non-Azure model stacks. |

**Confidence:** HIGH for the Microsoft-ecosystem items; MODERATE for third-party comparisons (included for completeness, not deeply verified here).

---

## 10. Key URLs

**PyRIT itself**
- GitHub (canonical): https://github.com/microsoft/PyRIT
- Documentation: https://microsoft.github.io/PyRIT/
- PyPI: https://pypi.org/project/pyrit/
- Authors' paper: https://arxiv.org/abs/2410.02828
- Announcement blog: https://www.microsoft.com/en-us/security/blog/2024/02/22/announcing-microsofts-open-automation-framework-to-red-team-generative-ai-systems/
- Datasets doc: https://microsoft.github.io/PyRIT/code/datasets/loading-datasets/
- Scorers API doc: https://microsoft.github.io/PyRIT/api/pyrit-score/

**Azure AI Foundry / azure-ai-evaluation**
- AI Red Teaming Agent (local, Evaluation SDK): https://learn.microsoft.com/en-us/azure/foundry/how-to/develop/run-scans-ai-red-teaming-agent
- `RedTeam` class API reference: https://learn.microsoft.com/en-us/python/api/azure-ai-evaluation/azure.ai.evaluation.red_team.redteam
- azure-ai-evaluation PyPI: https://pypi.org/project/azure-ai-evaluation/
- Planning red teaming (official guidance): https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/red-teaming

**Azure AI Content Safety**
- Content Safety docs: https://learn.microsoft.com/en-us/azure/ai-services/content-safety/
- Prompt Shields concept: https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection
- Content Safety Studio: https://contentsafety.cognitive.azure.com/

**Practice environments**
- AI Red Teaming Playground Labs: https://github.com/microsoft/AI-Red-Teaming-Playground-Labs

---

## Conclusion

**Why PyRIT is the right tool for Azure AI Foundry security testing:** Microsoft built it, uses it internally on its own flagship models, has open-sourced it under MIT, and has woven it into the Azure AI Foundry product surface via the `RedTeam` class in `azure-ai-evaluation[redteam]` and the preview "AI Red Teaming Agent". It is no longer an external framework bolted onto Azure — it is the reference implementation Microsoft ships.

**How the pieces fit:** PyRIT provides the scaffolding (targets → converters → datasets → orchestrators → scorers → memory); the datasets make adversarial coverage standardised (HarmBench, AdvBench, XSTest, JailbreakBench, ~30 others); and Azure AI Content Safety (Prompt Shields + harm-category analysis) plays both as a *defense* to red-team *against* and as a *scoring oracle* via `AzureContentFilterScorer` and `PromptShieldScorer`. The Foundry `RedTeam.scan()` wrapper collapses this into a single async call with an Attack Success Rate JSON scorecard — at the cost of less control than raw PyRIT.

**What to watch for:** (a) Class names evolve — pin your PyRIT version and match the notebooks to that version; (b) abuse monitoring and default content filters *will* interfere with honest red-team signal unless you go through Microsoft's Limited Access / RAI Policy process; (c) PyRIT memory (prompts + responses) is sensitive data — treat its storage as such; (d) the managed Foundry AI Red Teaming Agent is preview, single-turn text only, and available in a limited list of Azure regions.

**Residual risks / pre-mortem:** This report could be wrong if (1) PyRIT's class layout changes materially in a future minor release (likely — pin versions), (2) the `AI Red Teaming Agent` GA release changes the `RedTeam` API surface, or (3) Microsoft revises its Limited Access / abuse-monitoring policy (check Microsoft Learn at time of running a real engagement). The core story — PyRIT is Microsoft's recommended tool, `OpenAIChatTarget` is the Azure OpenAI target class, and the listed datasets ship with PyRIT — is unlikely to change materially.

---

## Sources

1. [PyRIT Documentation home (microsoft.github.io/PyRIT)](https://microsoft.github.io/PyRIT/)
2. [microsoft/PyRIT GitHub repo (MIT license, README)](https://github.com/microsoft/PyRIT)
3. [Announcing Microsoft's open automation framework to red team generative AI Systems — Ram Shankar Siva Kumar, Microsoft Security Blog, 22 Feb 2024](https://www.microsoft.com/en-us/security/blog/2024/02/22/announcing-microsofts-open-automation-framework-to-red-team-generative-ai-systems/)
4. [Lopez Munoz et al., "PyRIT: A Framework for Security Risk Identification and Red Teaming in Generative AI Systems," arXiv:2410.02828 (2024)](https://arxiv.org/html/2410.02828v1)
5. [PyRIT Loading Built-in Datasets documentation](https://microsoft.github.io/PyRIT/code/datasets/loading-datasets/)
6. [HarmBench dataset loader source (pyrit/datasets/seed_datasets/remote/harmbench_dataset.py)](https://github.com/microsoft/PyRIT/blob/main/pyrit/datasets/seed_datasets/remote/harmbench_dataset.py)
7. [OpenAI Chat and Completion Targets documentation (DeepWiki, sourced from PyRIT)](https://deepwiki.com/Azure/PyRIT/5.2-openai-chat-and-completion-targets)
8. [Built-in Datasets (DeepWiki)](https://deepwiki.com/Azure/PyRIT/3.1-built-in-datasets)
9. [Orchestrators documentation (DeepWiki)](https://deepwiki.com/Azure/PyRIT/9-development-guide)
10. [PyRIT Azure Content Safety scorer example (doc/code/scoring/1_azure_content_safety_scorers.py)](https://github.com/microsoft/PyRIT/blob/main/doc/code/scoring/1_azure_content_safety_scorers.py)
11. [Azure and API-Based Scorers (DeepWiki)](https://deepwiki.com/Azure/PyRIT/7.3-azure-and-api-based-scorers)
12. [pyrit.score API reference](https://microsoft.github.io/PyRIT/api/pyrit-score/)
13. [Run AI Red Teaming Agent locally (Azure AI Evaluation SDK) — Microsoft Learn](https://learn.microsoft.com/en-us/azure/foundry/how-to/develop/run-scans-ai-red-teaming-agent)
14. [azure.ai.evaluation.red_team.RedTeam class — Microsoft Learn](https://learn.microsoft.com/en-us/python/api/azure-ai-evaluation/azure.ai.evaluation.red_team.redteam?view=azure-python)
15. [azure-ai-evaluation on PyPI (v1.16.5, April 2026)](https://pypi.org/project/azure-ai-evaluation/)
16. [Prompt Shields in Azure AI Content Safety — Microsoft Learn](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection)
17. [Planning red teaming for LLMs and their applications — Microsoft Foundry](https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/red-teaming)
18. [Azure AI announces Prompt Shields for Jailbreak and Indirect prompt injection attacks (Microsoft Community Hub, March 2024)](https://thewindowsupdate.com/2024/03/28/azure-ai-announces-prompt-shields-for-jailbreak-and-indirect-prompt-injection-attacks/)
19. [Azure AI Content Safety documentation](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/)
20. [PyRIT from Zero to Red Team: Complete Setup and Attack Guide (AI Security in Practice)](https://www.aisecurityinpractice.com/attack-and-red-team/pyrit-zero-to-red-team/)
21. [AI Red-Teaming Playground Labs walkthrough with PyRIT (BreakPoint Labs)](https://breakpoint-labs.com/ai-red-teaming-playground-labs-setup-and-challenge-1-walkthrough-with-pyrit/)
22. [How to call AzureOpenAI API with PyRIT (Stack Overflow, verified answer)](https://stackoverflow.com/questions/79548081/how-to-call-azureopenai-api-with-pyrit)
23. [RaffaChiaverini Azure-AI-Red-Teaming-Agent implementation (GitHub)](https://github.com/RaffaChiaverini/Azure-AI-Red-Teaming-Agent---Implementation)
24. [HarmBench dataset homepage (Mazeika et al.)](https://www.harmbench.org/explore)
25. [AdvBench dataset card (Hugging Face walledai/AdvBench)](https://huggingface.co/datasets/walledai/AdvBench)
26. [Prompt Shield and Attack Prevention (Azure/azure-openai-samples DeepWiki)](https://deepwiki.com/Azure/azure-openai-samples/4.2-prompt-shield-and-attack-prevention)
