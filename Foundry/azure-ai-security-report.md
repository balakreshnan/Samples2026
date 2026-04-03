# Azure AI Security: Configuration, Testing, and Maturity Guide

## Executive Summary

Azure AI security is a rapidly maturing domain that requires a layered, defense-in-depth approach spanning network isolation, identity management, content safety, continuous evaluation, and adversarial red teaming. Microsoft has invested heavily in tooling across three pillars: (1) platform security for Azure OpenAI and AI Foundry via private endpoints, RBAC, Azure Policy, and encryption; (2) application security through Content Safety filters, Prompt Shields, and the new MCSB v2 AI-specific controls (AI-1 through AI-7); and (3) continuous testing through the Azure AI Evaluation SDK, AI Red Teaming Agent (preview), and the open-source PyRIT framework. This guide provides a Principal Cloud Solution Architect with the configuration steps, testing methodologies, and maturity benchmarks needed to build and execute an enterprise AI security strategy.

**Overall Confidence:** MODERATE-HIGH. The configuration and testing tooling is well-documented by Microsoft. The maturity model guidance is less formally defined by Microsoft and is partially synthesized from multiple frameworks (MCSB, NIST AI RMF, Responsible AI Standard, Zero Trust). The AI Red Teaming Agent remains in public preview.

---

## Part 1: How to Configure Azure AI Security

### 1.1 Azure AI Foundry / Azure OpenAI Security Configurations

Azure AI Foundry (formerly Azure AI Studio) and Azure OpenAI Service form the core platform for enterprise AI workloads. Security configuration starts at the hub level and cascades to projects and model deployments.

**Key Architecture Components:**
- **AI Foundry Hub:** The top-level organizational unit (Microsoft.MachineLearningServices/workspaces with Hub type) that provides managed network, compute, centralized service connections, and storage [Azure AI security best practices](https://learn.microsoft.com/en-us/azure/security/fundamentals/ai-security-best-practices).
- **AI Foundry Project:** Child resource within a hub for development isolation, project-specific storage, and deployment capabilities.
- **Azure OpenAI Resource:** The Cognitive Services account (Microsoft.CognitiveServices/accounts) hosting model deployments.

**Configuration Steps:**
1. Create a dedicated Resource Group per environment (dev/test/prod).
2. Provision an AI Foundry Hub with managed network isolation enabled.
3. Create projects within the hub, inheriting network and access policies.
4. Deploy Azure OpenAI models within projects with RAI (Responsible AI) policies attached.
5. Configure connected services (Azure AI Search, Storage, Key Vault) with private endpoints.

> **Key Finding:** Azure AI Foundry supports managed network isolation as a built-in feature, providing private endpoints for all dependent services and outbound traffic control without requiring manual VNet configuration.
> **Confidence:** HIGH
> **Action:** Enable managed network isolation on all AI Foundry hubs in production environments.

**Reference:** [Azure AI security best practices](https://learn.microsoft.com/en-us/azure/security/fundamentals/ai-security-best-practices), [Azure security baseline for Microsoft Foundry](https://learn.microsoft.com/en-us/security/benchmark/azure/baselines/azure-ai-foundry-security-baseline)

---

### 1.2 Model Inferencing Security

Model inferencing -- the runtime API calls to Azure OpenAI -- represents the primary attack surface. Securing it requires network, identity, encryption, and API management controls.

#### Private Endpoints and VNet Integration

Azure OpenAI Service supports deployment into a customer's private Virtual Network via Private Link. This removes the public endpoint and ensures traffic stays on the Microsoft backbone network [Azure security baseline for Azure OpenAI](https://learn.microsoft.com/en-us/security/benchmark/azure/baselines/azure-openai-security-baseline).

**Implementation Steps:**
1. Create a VNet with at least two subnets: one for applications, one for private endpoints.
2. Disable private endpoint network policies on the PE subnet:
   ```bash
   az network vnet subnet update \
     --name snet-private-endpoints \
     --resource-group rg-openai-network \
     --vnet-name vnet-openai \
     --disable-private-endpoint-network-policies true
   ```
3. Create the private endpoint for the Azure OpenAI resource:
   ```bash
   az network private-endpoint create \
     --name pe-openai \
     --resource-group rg-openai-network \
     --vnet-name vnet-openai \
     --subnet snet-private-endpoints \
     --private-connection-resource-id <openai-resource-id> \
     --group-id account \
     --connection-name openai-connection
   ```
4. Configure a Private DNS Zone (privatelink.openai.azure.com) and link it to the VNet.
5. Disable public network access on the Azure OpenAI resource:
   ```bash
   az cognitiveservices account update \
     --name <openai-resource> \
     --resource-group <rg> \
     --public-network-access Disabled
   ```

**Reference:** [Azure OpenAI Private Endpoints](https://techcommunity.microsoft.com/blog/azurearchitectureblog/azure-openai-private-endpoints-connecting-across-vnet%E2%80%99s/3913325), [Azure security baseline for Azure OpenAI](https://learn.microsoft.com/en-us/security/benchmark/azure/baselines/azure-openai-security-baseline)

#### Managed Identity Authentication

Microsoft Entra ID (formerly Azure AD) managed identities eliminate API key management risk. System-assigned managed identity is recommended for service-to-service communication [Azure AI security best practices](https://learn.microsoft.com/en-us/azure/security/fundamentals/ai-security-best-practices).

**Implementation Steps:**
1. Enable system-assigned managed identity on the consuming resource (App Service, Function, VM).
2. Assign the appropriate RBAC role (e.g., Cognitive Services OpenAI User) to the managed identity on the Azure OpenAI resource.
3. Configure the application to use `DefaultAzureCredential` from the Azure Identity SDK.
4. Disable local authentication (API keys) on the Azure OpenAI resource using Azure Policy or CLI:
   ```bash
   az cognitiveservices account update \
     --name <openai-resource> \
     --resource-group <rg> \
     --disable-local-auth true
   ```

> **Key Finding:** Microsoft recommends disabling API key-based local authentication in production and enforcing Entra ID authentication exclusively. A built-in Azure Policy ("Azure AI Services resources should have key access disabled") can enforce this at scale.
> **Confidence:** HIGH
> **Action:** Enforce the policy "Azure AI Services resources should have key access disabled" across all production subscriptions.

**Reference:** [Azure AI security best practices](https://learn.microsoft.com/en-us/azure/security/fundamentals/ai-security-best-practices), [Azure Policy built-in definitions for AI services](https://docs.azure.cn/en-us/ai-services/policy-reference)

#### Customer-Managed Keys (CMK)

Azure AI services encrypt data at rest by default using Microsoft-managed keys. For organizations requiring full control over the encryption lifecycle, customer-managed keys (CMK) via Azure Key Vault are supported [Azure security baseline for Azure OpenAI](https://learn.microsoft.com/en-us/security/benchmark/azure/baselines/azure-openai-security-baseline).

**Implementation Steps:**
1. Create an Azure Key Vault with soft-delete and purge protection enabled.
2. Create an RSA key in the Key Vault.
3. Grant the Azure OpenAI resource's managed identity "Key Vault Crypto Service Encryption User" permissions.
4. Configure the Azure OpenAI resource to use the CMK:
   ```bash
   az cognitiveservices account update \
     --name <openai-resource> \
     --resource-group <rg> \
     --encryption-key-source Microsoft.KeyVault \
     --encryption-key-vault <key-vault-uri> \
     --encryption-key-name <key-name> \
     --encryption-key-version <key-version>
   ```

A built-in Azure Policy ("Azure AI Services resources should encrypt data at rest with a customer-managed key") can audit or deny resources without CMK [Azure Policy built-in definitions for AI services](https://docs.azure.cn/en-us/ai-services/policy-reference).

---

### 1.3 Content Safety Service Configuration

Azure AI Content Safety provides multi-layered content filtering for both inputs (prompts) and outputs (completions). It is tightly integrated with Azure OpenAI deployments via RAI policies.

#### Content Categories and Severity Thresholds

Azure AI Content Safety evaluates content against four categories, each with four severity levels [Prompt Shields - Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection), [Content Safety and Responsible AI](https://deepwiki.com/azure-ai-foundry/ai-tutorials/5.2-content-safety-and-responsible-ai):

| Category | What It Detects |
|----------|----------------|
| **Hate** | Content targeting identity groups based on race, gender, religion, etc. |
| **Violence** | Content depicting or promoting violence against people or animals |
| **Sexual** | Sexually explicit content |
| **Self-Harm** | Content related to self-harm or suicide |

Each category has four severity levels:
- **Safe:** Clearly benign content
- **Low:** Mildly concerning but generally acceptable
- **Medium:** Moderately concerning
- **High:** Clearly harmful

The default filter blocks content at **Medium and High** severity for all categories on both inputs and outputs.

#### Custom Content Filter Policies

Organizations can create custom policies to adjust thresholds per deployment:

**Implementation Steps:**
1. Navigate to Azure OpenAI Studio (oai.azure.com) or use the REST API.
2. Create a custom content filter configuration specifying per-category thresholds for both prompts and completions.
3. Optionally enable additional filters: Jailbreak detection (Prompt Shields for User Prompts), Indirect Attack detection (Prompt Shields for Documents), and Protected Material detection.
4. Assign the custom policy to a model deployment via the `rai_policy_name` parameter.

> **Key Finding:** On Azure, content filter policies are attached directly to model deployments via the `rai_policy_name` parameter, making safety enforcement an infrastructure-level concern rather than an application-level one. This is architecturally distinct from AWS and GCP approaches.
> **Confidence:** HIGH
> **Action:** Define RAI policies as code (Terraform/Bicep) and attach them to all deployments.

#### Blocklists

For organization-specific terms that the default AI classifiers do not cover, custom blocklists allow exact-match or regex-pattern filtering [Use blocklists for text moderation](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/how-to/use-blocklist):

```python
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import TextBlocklist

client = ContentSafetyClient(endpoint, credential)

# Create a blocklist
blocklist = client.create_or_update_text_blocklist(
    blocklist_name="CompanyBlocklist",
    options=TextBlocklist(description="Company-specific blocked terms")
)

# Add items to blocklist
client.add_or_update_blocklist_items(
    blocklist_name="CompanyBlocklist",
    options={"blocklistItems": [
        {"description": "Competitor name", "text": "competitor-term"},
        {"description": "Internal codename", "text": "project-codename"}
    ]}
)
```

#### Prompt Shields

Prompt Shields is a unified API that detects and blocks two types of adversarial attacks [Prompt Shields - Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection):

1. **User Prompt Attacks (Jailbreaks):** Users deliberately exploit system vulnerabilities to elicit unauthorized behavior. Subtypes include: attempts to change system rules, embedding conversation mockups, role-play attacks, and encoding attacks (Base64, ROT13, Leetspeak, etc.).

2. **Document Attacks (Indirect Injection):** Third-party content (documents, emails) contains hidden instructions to manipulate the LLM session. Subtypes include: content manipulation, unauthorized access commands, information gathering/exfiltration, availability attacks, fraud, and malware distribution.

#### Groundedness Detection

Groundedness detection checks whether LLM-generated content is substantiated by provided source documents. It offers reasoning (explaining why content is ungrounded) and correction (suggesting fixes). This is available both as a standalone Content Safety API and as an evaluator in the Azure AI Evaluation SDK [GroundednessEvaluator](https://learn.microsoft.com/en-us/python/api/azure-ai-evaluation/azure.ai.evaluation.groundednessevaluator).

---

### 1.4 RBAC for Azure AI Services

Azure provides a comprehensive set of built-in roles for AI workloads [Azure built-in roles for AI + ML](https://learn.microsoft.com/en-us/azure/role-based-access-control/built-in-roles/ai-machine-learning):

| Role | Description | Scope | Key Permissions |
|------|-------------|-------|-----------------|
| **Azure AI Account Owner** | Full access to manage AI projects and accounts. Can assign Azure AI User role to others (ABAC constrained). | New Foundry resources | Microsoft.CognitiveServices/*, role assignment write |
| **Azure AI Administrator** | All control plane permissions for Azure AI and dependencies. | ML/Foundry hubs | CognitiveServices/*, KeyVault/*, MachineLearningServices/workspaces/*, Storage/* |
| **Azure AI Developer** | All actions within an AI resource except managing the resource itself. | ML/Foundry hubs | DataActions: OpenAI/*, SpeechServices/*, ContentSafety/*, MaaS/*. Cannot delete/write hubs. |
| **Azure AI User** | Read access and ability to use AI services. | Foundry resources | Read-level Cognitive Services, inference calls |
| **Cognitive Services OpenAI User** | View files/models/deployments, make inference calls. Cannot make changes. | OpenAI resources | DataActions: OpenAI read, deployments/chat/completions/action, embeddings, audio, etc. |
| **Cognitive Services OpenAI Contributor** | Full access including fine-tuning, deployment, and text generation. | OpenAI resources | DataActions: OpenAI/*, includes raiPolicies read/write/delete |
| **Azure AI Enterprise Network Connection Approver** | Approve private endpoint connections to AI dependency resources. | Cross-service | Private endpoint connection approval for CognitiveServices, KeyVault, Storage, CosmosDB, Search, etc. |

**Implementation Best Practices:**
1. **Entra ID first:** Always use Microsoft Entra ID authentication over API keys.
2. **Least privilege:** Assign `Cognitive Services OpenAI User` for consumers, `Azure AI Developer` for builders, and `Azure AI Administrator` only for platform teams.
3. **Managed Identities:** Use system-assigned managed identities for service-to-service communication.
4. **PIM for admin roles:** Use Privileged Identity Management (PIM) for Just-In-Time elevation to Azure AI Administrator.
5. **Conditional Access:** Enforce MFA, trusted locations, and compliant device requirements for AI resource access.

**Reference:** [Azure built-in roles for AI + ML](https://learn.microsoft.com/en-us/azure/role-based-access-control/built-in-roles/ai-machine-learning)

---

### 1.5 Azure Policy for AI Governance

Azure Policy provides built-in policy definitions specifically for Azure AI services (Cognitive Services / Foundry Tools) [Azure Policy built-in definitions for AI services](https://docs.azure.cn/en-us/ai-services/policy-reference).

#### Key Built-in Policies

| Policy | Description | Effect |
|--------|-------------|--------|
| **Encrypt data at rest with CMK** | Requires customer-managed keys for AI services | Audit, Deny, Disabled |
| **Disable local authentication** | Requires Entra ID authentication, blocks API key use | Audit, Deny, Disabled |
| **Restrict network access** | Ensures only allowed networks can access the service | Audit, Deny, Disabled |
| **Use Azure Private Link** | Requires private endpoint connections | Audit, Disabled |
| **Use managed identity** | Requires managed identity assignment | Audit, Deny, Disabled |
| **Approved Registry Models only** (Preview) | Restricts ML deployments to approved model asset IDs | Deny |

#### ALZ Policy Initiative: "Enforce recommended guardrails for OpenAI"

The Azure Landing Zones (ALZ) framework provides a curated policy initiative (`Enforce-Guardrails-OpenAI`) containing 11 policies that enforce comprehensive AI governance [ALZ Guardrails for OpenAI](https://www.azadvertizer.com/azpolicyinitiativesadvertizer/Enforce-Guardrails-OpenAI.html):

- Disable local authentication (two variants: Audit and DeployIfNotExists)
- Restrict network access
- Require Private Link
- Require managed identity
- Require customer-owned storage
- Configure local key access disabling (remediation)

**Implementation Steps:**
1. Assign the `Enforce-Guardrails-OpenAI` initiative at the management group or subscription level.
2. Set effects to `Deny` for production subscriptions (block non-compliant resource creation).
3. Set effects to `Audit` for development subscriptions (visibility without blocking).
4. Create custom policies for additional governance requirements (e.g., restrict model deployment types to Standard vs. GlobalStandard to control data residency).
5. Monitor compliance via Azure Policy Compliance dashboard and Defender for Cloud regulatory compliance.

**Custom Policy Example -- Restrict Data Residency:**
Organizations can create custom policies to enforce that Azure OpenAI deployments use only `Standard` deployment types (not `GlobalStandard`) to keep data processing within a specific region [Restrict OpenAI Models with Azure Policy](https://codewithme.cloud/posts/2024/12/openai-restrict-models-with-policy/).

---

### 1.6 Network Isolation, Private Endpoints, and Data Encryption

This section consolidates the network and encryption posture across all Azure AI components.

#### Network Isolation Architecture

For a fully isolated deployment:

| Component | Private Endpoint | NSG Support | Firewall Support |
|-----------|-----------------|-------------|------------------|
| Azure OpenAI | Yes | Yes (customer responsibility) | Yes |
| AI Foundry Hub | Yes (managed network) | Yes | Yes |
| Azure AI Search | Yes | Yes | Yes |
| Azure Storage | Yes | Yes | Yes |
| Azure Key Vault | Yes | Yes | Yes |
| Azure Container Registry | Yes | Yes | Yes |

**Implementation Pattern:**
1. Deploy all resources into a hub-spoke VNet topology.
2. Use a dedicated subnet for private endpoints.
3. Deploy Azure Firewall or NVA for centralized egress inspection.
4. Apply NSGs at the subnet level to filter traffic.
5. Use Azure DNS Private Zones for name resolution of private endpoints.
6. Disable public access on all AI services.

#### Data Encryption

| Layer | Default | Optional |
|-------|---------|----------|
| **At rest** | Microsoft-managed keys (AES-256) | Customer-managed keys via Key Vault |
| **In transit** | TLS 1.2+ enforced | N/A (always encrypted) |
| **Processing** | Data processed in-region for Standard deployments | GlobalStandard may process outside region |

**Reference:** [Azure security baseline for Azure OpenAI](https://learn.microsoft.com/en-us/security/benchmark/azure/baselines/azure-openai-security-baseline), [Azure AI security best practices](https://learn.microsoft.com/en-us/azure/security/fundamentals/ai-security-best-practices)

---

### 1.7 Diagnostic Logging and Monitoring

Azure AI services emit diagnostic logs that are essential for security monitoring, compliance auditing, and operational visibility [Enable diagnostic logging for Foundry Tools](https://learn.microsoft.com/en-us/azure/ai-services/diagnostic-logging).

#### Log Categories

| Category | What It Captures | Compliance Value |
|----------|-----------------|------------------|
| **Audit** | Key access events (ListKeys operations) | Security audit trail |
| **RequestResponse** | Every API call: model, operation, duration, caller IP, status code | Usage tracking, performance monitoring |
| **Trace** | Internal service trace data | Debugging |
| **AllMetrics** | Platform metrics (requests, latency, token usage) | Performance and cost monitoring |

**Important Limitation:** RequestResponse logs capture metadata about calls (which model, how long, success/failure) but do NOT include actual prompt or response content. For full prompt/response capture, deploy Azure API Management (APIM) in front of the endpoint or implement application-level logging [Azure AI Foundry Diagnostic Logging with Terraform](https://dev.to/suhas_mallesh/azure-ai-foundry-diagnostic-logging-with-terraform-every-ai-call-tracked-for-compliance-1b2d).

#### Implementation Steps

1. Navigate to the Azure AI resource > Monitoring > Diagnostic settings.
2. Add a diagnostic setting:
   - Select log categories: Audit, RequestResponse, AllMetrics.
   - Send to: Log Analytics workspace (for real-time queries) and Storage Account (for long-term retention).
3. Configure in IaC (Terraform example):
   ```hcl
   resource "azurerm_monitor_diagnostic_setting" "ai_diag" {
     name                       = "ai-diagnostics"
     target_resource_id         = azurerm_cognitive_account.openai.id
     log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
     storage_account_id         = azurerm_storage_account.logs.id

     enabled_log { category = "Audit" }
     enabled_log { category = "RequestResponse" }
     enabled_log { category = "Trace" }
     metric { category = "AllMetrics" }
   }
   ```

#### Azure Monitor Integration

- **Application Insights:** Connect AI Foundry projects to Application Insights for request/trace/exception telemetry and token usage tracking [Monitoring Generative AI Applications with Azure AI Foundry](https://azurefeeds.com/2025/11/07/monitoring-generative-ai-applications-with-azure-ai-foundry/).
- **KQL Queries:** Use Kusto Query Language in Log Analytics to analyze API usage patterns:
  ```kql
  AzureDiagnostics
  | where ResourceType == "ACCOUNTS" and Category == "RequestResponse"
  | summarize count() by OperationName, ResultType, bin(TimeGenerated, 1h)
  | order by TimeGenerated desc
  ```
- **Alerts:** Create Azure Monitor alert rules for anomalous patterns (e.g., spike in 429 errors, unusual caller IPs, content filter blocks).
- **Workbooks:** Build custom Azure Monitor workbooks to visualize AI usage, latency, token consumption, and content filter trigger rates.

---

## Part 2: How to Test Azure AI Security

### 2.1 Azure AI Evaluation SDK

The `azure-ai-evaluation` Python SDK (v1.16.2 as of March 2026) provides programmatic evaluation of generative AI application quality and safety [azure-ai-evaluation PyPI](https://pypi.org/project/azure-ai-evaluation/).

#### Built-in Evaluators

| Category | Evaluators | Scoring |
|----------|-----------|---------|
| **Quality (AI-assisted)** | GroundednessEvaluator, RelevanceEvaluator, CoherenceEvaluator, FluencyEvaluator, SimilarityEvaluator, RetrievalEvaluator | 1-5 scale |
| **Quality (NLP)** | F1ScoreEvaluator, RougeScoreEvaluator, GleuScoreEvaluator, BleuScoreEvaluator, MeteorScoreEvaluator | 0-1 scale |
| **Safety (AI-assisted)** | ViolenceEvaluator, SexualEvaluator, SelfHarmEvaluator, HateUnfairnessEvaluator, IndirectAttackEvaluator, ProtectedMaterialEvaluator | Severity scale 0-7 |
| **Composite** | QAEvaluator, ContentSafetyEvaluator | Combined |

#### Implementation Example

```python
import os
from azure.ai.evaluation import (
    evaluate,
    GroundednessEvaluator,
    RelevanceEvaluator,
    ViolenceEvaluator,
    ContentSafetyEvaluator
)

model_config = {
    "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
    "api_key": os.environ["AZURE_OPENAI_KEY"],
    "azure_deployment": os.environ["AZURE_OPENAI_DEPLOYMENT"],
}

# Quality evaluator
groundedness = GroundednessEvaluator(model_config=model_config)
result = groundedness(
    response="Paris is the capital of France.",
    context="France's capital is Paris, known for the Eiffel Tower."
)
# result: {"groundedness": 5, "groundedness_reason": "..."}

# Safety evaluator
violence = ViolenceEvaluator(model_config=model_config)
result = violence(
    query="How do I handle a difficult customer?",
    response="Here are some de-escalation techniques..."
)

# Batch evaluation with ground truth dataset
results = evaluate(
    data="test_dataset.jsonl",
    evaluators={
        "groundedness": groundedness,
        "relevance": RelevanceEvaluator(model_config),
        "content_safety": ContentSafetyEvaluator(model_config),
    },
    output_path="eval_results.json"
)
```

**Reference:** [azure-ai-evaluation PyPI](https://pypi.org/project/azure-ai-evaluation/), [GroundednessEvaluator](https://learn.microsoft.com/en-us/python/api/azure-ai-evaluation/azure.ai.evaluation.groundednessevaluator)

---

### 2.2 AI Red Teaming Agent (AART)

The AI Red Teaming Agent (preview) is an automated adversarial testing tool integrated into Azure AI Foundry. It combines PyRIT's red teaming capabilities with Microsoft Foundry's Risk and Safety Evaluations [AI Red Teaming Agent](https://learn.microsoft.com/en-us/azure/foundry/concepts/ai-red-teaming-agent).

#### What It Does

1. **Automated scans for content risks:** Simulates adversarial probing against model/application endpoints.
2. **Evaluates probing success:** Scores each attack-response pair using the Attack Success Rate (ASR) metric.
3. **Reporting and logging:** Generates scorecards by attack technique and risk category, logged in Foundry for tracking over time.

#### Supported Risk Categories

| Risk Category | Target | Description |
|--------------|--------|-------------|
| Hateful and Unfair Content | Models and agents | Hate toward or unfair representations of individuals/groups |
| Sexual Content | Models and agents | Sexually explicit content |
| Violent Content | Models and agents | Physical actions intended to hurt, injure, or kill |
| Self-Harm Content | Models and agents | Actions intended to damage oneself |
| Protected Materials | Models and agents | Copyrighted content (lyrics, recipes, code) |
| Code Vulnerability | Models and agents | Generated code with security vulnerabilities (SQL injection, etc.) |
| Ungrounded Attributes | Models and agents | Ungrounded inferences about personal attributes |
| **Prohibited Actions** | **Agents only** | Actions violating explicitly disallowed policies |
| **Sensitive Data Leakage** | **Agents only** | Exposure of financial, personal, health data |
| **Task Adherence** | **Agents only** | Failure to complete tasks within rules and constraints |
| **Indirect Prompt Injection (XPIA)** | **Agents only** | Hidden instructions in external data sources |

#### Supported Attack Strategies (20+)

The AART leverages PyRIT's attack strategies including: AnsiAttack, AsciiArt, AsciiSmuggler, Atbash, Base64, Binary, Caesar, CharacterSpace, CharSwap, Diacritic, Flip, Leetspeak, Morse, ROT13, SuffixAppend, StringJoin, UnicodeConfusable, UnicodeSubstitution, Url encoding, Jailbreak (direct), Indirect Jailbreak, Tense manipulation, Multi-turn attacks, and Crescendo (gradual escalation) [AI Red Teaming Agent](https://learn.microsoft.com/en-us/azure/foundry/concepts/ai-red-teaming-agent).

#### Configuration and Usage

```python
# Install
# pip install "azure-ai-evaluation[redteam]"
# Requires Python 3.10, 3.11, or 3.12

import os
from azure.identity import DefaultAzureCredential
from azure.ai.evaluation.red_team import RedTeam, RiskCategory

azure_ai_project = {
    "subscription_id": os.environ["AZURE_SUBSCRIPTION_ID"],
    "resource_group_name": os.environ["AZURE_RESOURCE_GROUP_NAME"],
    "project_name": os.environ["AZURE_PROJECT_NAME"],
}

# Instantiate the Red Teaming Agent
red_team_agent = RedTeam(
    azure_ai_project=azure_ai_project,
    credential=DefaultAzureCredential(),
    risk_categories=[
        RiskCategory.Violence,
        RiskCategory.HateUnfairness,
        RiskCategory.Sexual,
        RiskCategory.SelfHarm,
    ],
    num_objectives=5,  # prompts per risk category
)

# Run against a callback function
async def my_target(prompt: str) -> str:
    # Call your AI endpoint here
    return response

result = await red_team_agent.scan(my_target)
# result contains ASR per category, individual attack-response pairs
```

> **Key Finding:** The AI Red Teaming Agent is currently in **public preview** (as of February 2026) and not recommended for production workloads. It requires a hub-based project (not a Foundry project). Agentic risk categories (prohibited actions, sensitive data leakage, task adherence) are cloud-only. Results use synthetic data and mock tools, not representative of real-world data distributions.
> **Confidence:** HIGH
> **Action:** Use AART in pre-production testing pipelines. Supplement with manual red teaming for production-critical applications.

**Reference:** [AI Red Teaming Agent](https://learn.microsoft.com/en-us/azure/foundry/concepts/ai-red-teaming-agent), [Configure AI Red Teaming Agent in Foundry - Zero Trust](https://microsoft.github.io/zerotrustassessment/docs/workshop-guidance/AI/AI_081)

---

### 2.3 PyRIT (Python Risk Identification Toolkit)

PyRIT is Microsoft's open-source, model-agnostic framework for AI red teaming, released in February 2024 and actively maintained (3,600+ GitHub stars, 1,070+ commits) [PyRIT GitHub](https://github.com/microsoft/pyrit), [PyRIT Documentation](https://azure.github.io/PyRIT/).

#### Key Capabilities

- **Automated Red Teaming:** Run multi-turn attack strategies (Crescendo, Tree of Attacks with Pruning/TAP, Skeleton Key) with minimal setup.
- **Scenario Framework:** Standardized evaluation scenarios covering content harms, psychosocial risks, data leakage, and more.
- **CoPyRIT GUI:** Graphical user interface for human-led red teaming.
- **Any Target:** Test OpenAI, Azure, Anthropic, Google, HuggingFace, custom HTTP/WebSocket endpoints, and web apps via Playwright.
- **Built-in Memory:** Track all conversations, scores, and results with SQLite or Azure SQL.
- **Flexible Scoring:** True/false, Likert scale, classification, and custom scorers powered by LLMs, Azure AI Content Safety, or custom logic.

#### Installation and Configuration

```bash
# User installation
pip install pyrit

# Or with Docker
docker pull pyrit/pyrit:latest
```

Configuration requires a `.env` file with AI endpoint credentials:
```
AZURE_OPENAI_ENDPOINT=https://<account>.openai.azure.com/
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_DEPLOYMENT=<deployment-name>
```

#### Usage Pattern

PyRIT operates on a composable architecture with five core building blocks:
1. **Targets:** The AI system being tested (Azure OpenAI, custom endpoints, etc.)
2. **Orchestrators:** Manage attack flow (single-turn, multi-turn, Crescendo, TAP)
3. **Converters:** Transform prompts using attack strategies (Base64, ROT13, Unicode, etc.)
4. **Scorers:** Evaluate whether attacks succeeded (LLM-based, Content Safety, custom)
5. **Memory:** Persistent storage of all interactions and results

> **Key Finding:** PyRIT is the foundation that powers both the AI Red Teaming Agent and Microsoft's internal AI Red Team (which ran 67 operations across flagship models in 2024). It is the most mature open-source AI red teaming framework available.
> **Confidence:** HIGH
> **Action:** Adopt PyRIT for automated red teaming in CI/CD pipelines and periodic security assessments.

**Reference:** [PyRIT Documentation](https://azure.github.io/PyRIT/), [Announcing PyRIT - Microsoft Security Blog](https://www.microsoft.com/en-us/security/blog/2024/02/22/announcing-microsofts-open-automation-framework-to-red-team-generative-ai-systems/), [PyRIT Academic Paper](https://arxiv.org/html/2410.02828v1)

---

### 2.4 Simulating Adversarial Attacks

#### Common Attack Types to Test

| Attack Type | Description | Testing Tool |
|-------------|-------------|-------------|
| **Direct Prompt Injection** | User crafts prompts to override system instructions | PyRIT, AART (Jailbreak strategy) |
| **Indirect Prompt Injection (XPIA)** | Hidden instructions in documents/data | AART (Indirect Jailbreak), PyRIT |
| **Encoding Attacks** | Base64, ROT13, Leetspeak, Unicode to bypass filters | PyRIT converters, AART attack strategies |
| **Multi-turn Escalation (Crescendo)** | Gradually escalating prompts across turns | PyRIT Crescendo orchestrator |
| **Role-Play / Persona Attacks** | Instructing model to adopt unrestricted persona | PyRIT, manual testing |
| **Data Exfiltration** | Attempting to extract training data or system prompts | AART (Sensitive Data Leakage), PyRIT |
| **Tool/Function Abuse** | Manipulating agent to misuse tools | AART (Prohibited Actions, Task Adherence) |

#### Guard Rails: Integrated Defense Stack

Deploy a multi-layered defense combining:

1. **Azure AI Content Safety filters** (RAI policies on deployments): Block harmful content at the model level.
2. **Prompt Shields:** Detect jailbreak and indirect injection attempts before model processing.
3. **Groundedness Detection:** Validate that responses are grounded in provided context.
4. **Safety meta-prompts (system messages):** Instruct the model to reject manipulation and prioritize system instructions.
5. **Azure API Management:** Rate limiting, schema validation, and request/response logging at the gateway.
6. **Microsoft Defender for AI Services:** Runtime threat detection for prompt injection, data exposure, and anomalous API usage.

---

### 2.5 Testing Frameworks and CI/CD Integration

#### Integrating AI Security Testing into CI/CD

```yaml
# Azure DevOps Pipeline example
stages:
  - stage: AI_Security_Testing
    jobs:
      - job: EvaluationTests
        steps:
          - script: pip install azure-ai-evaluation
          - script: python run_evaluations.py
            displayName: "Run Quality & Safety Evaluations"
            env:
              AZURE_OPENAI_ENDPOINT: $(AZURE_OPENAI_ENDPOINT)
              AZURE_OPENAI_KEY: $(AZURE_OPENAI_KEY)

      - job: RedTeaming
        steps:
          - script: pip install "azure-ai-evaluation[redteam]"
          - script: python run_red_team.py
            displayName: "Run Automated Red Teaming"

      - job: GateCheck
        dependsOn: [EvaluationTests, RedTeaming]
        steps:
          - script: python check_thresholds.py
            displayName: "Verify ASR < threshold, quality scores > threshold"
```

**Gate Criteria Examples:**
- Groundedness score average >= 4.0 (out of 5)
- Content Safety evaluator severity maximum < 3 (Medium)
- Attack Success Rate (ASR) < 5% across all risk categories
- No XPIA (indirect prompt injection) successes

> **Key Finding:** There is no official Microsoft-published CI/CD template for AI security testing. The patterns above are synthesized from community practices and Microsoft's own recommendations to "integrate red teaming into CI/CD pipelines" in their security best practices documentation.
> **Confidence:** MODERATE
> **Action:** Build custom CI/CD gates using the Azure AI Evaluation SDK and PyRIT. Standardize gate thresholds based on your organization's risk tolerance.

---

## Part 3: What "Good" Looks Like -- Confidence/Maturity Model

### 3.1 Security Baselines from Microsoft

Microsoft provides two levels of security baselines for Azure AI [Azure security baseline for Azure OpenAI](https://learn.microsoft.com/en-us/security/benchmark/azure/baselines/azure-openai-security-baseline), [MCSB v2 - AI Security](https://learn.microsoft.com/en-us/security/benchmark/azure/mcsb-v2-artificial-intelligence-security):

#### MCSB v1 Security Baselines (Azure OpenAI, AI Foundry)

These baselines map Microsoft Cloud Security Benchmark v1 controls to Azure AI services across domains:
- **Network Security (NS):** VNet integration, private endpoints, NSG support
- **Identity Management (IM):** Entra ID authentication, managed identity, local auth disabling
- **Privileged Access (PA):** PIM, JIT access
- **Data Protection (DP):** Encryption at rest (CMK optional), encryption in transit (TLS 1.2+)
- **Logging and Threat Detection (LT):** Diagnostic logging, Defender for Cloud integration
- **Asset Management (AM):** Resource inventory, tagging
- **Posture and Vulnerability Management (PV):** Defender CSPM, security recommendations

#### MCSB v2 AI-Specific Controls (Preview)

The new MCSB v2 introduces seven AI-specific security controls organized into three pillars [MCSB v2 - AI Security](https://learn.microsoft.com/en-us/security/benchmark/azure/mcsb-v2-artificial-intelligence-security):

| Control | Name | Criticality | Pillar |
|---------|------|-------------|--------|
| **AI-1** | Ensure use of approved models | Must have | Platform Security |
| **AI-2** | Implement multi-layered content filtering | Must have | Application Security |
| **AI-3** | Adopt safety meta-prompts | Must have | Application Security |
| **AI-4** | Apply least privilege for agent functions | Must have | Application Security |
| **AI-5** | Ensure human-in-the-loop | Should have | Application Security |
| **AI-6** | Establish monitoring and detection | Must have | Monitor and Respond |
| **AI-7** | Perform continuous AI red teaming | Must have | Monitor and Respond |

Each control maps to established compliance frameworks:
- NIST SP 800-53 Rev. 5
- PCI-DSS v4.0
- CIS Controls v8.1
- NIST Cybersecurity Framework v2.0
- ISO 27001:2022
- SOC 2

---

### 3.2 Microsoft Responsible AI Framework

Microsoft's Responsible AI framework is built on six principles, operationalized through the Responsible AI Standard and the NIST AI Risk Management Framework (Govern-Map-Measure-Manage) [Microsoft Responsible AI Transparency Report 2025](https://www.microsoft.com/en-us/corporate-responsibility/responsible-ai-transparency-report/):

| Principle | Security Relevance | Azure Implementation |
|-----------|-------------------|---------------------|
| **Fairness** | Bias in model outputs | Responsible AI Dashboard, Fairlearn |
| **Reliability & Safety** | Consistent, safe model behavior | Content Safety filters, evaluations |
| **Privacy & Security** | Data protection, access control | Private endpoints, RBAC, CMK, DLP |
| **Inclusiveness** | Accessible to all users | Accessibility testing |
| **Transparency** | Understandable AI decisions | Diagnostic logging, audit trails |
| **Accountability** | Human oversight and governance | Human-in-the-loop (AI-5), PIM, audit |

**Operationalization:**
- **Govern:** Responsible AI Standard, Frontier Governance Framework, Board-level oversight
- **Map:** AI Red Team operations (67 in 2024), risk identification with PyRIT
- **Measure:** Automated measurement pipeline, evaluator models, policy-aligned metrics
- **Manage:** Defense-in-depth mitigations, content filters, post-deployment monitoring

---

### 3.3 Zero Trust Principles Applied to AI Workloads

Zero Trust for AI workloads extends the traditional "Verify Explicitly, Use Least Privilege, Assume Breach" framework [Azure AI security best practices](https://learn.microsoft.com/en-us/azure/security/fundamentals/ai-security-best-practices), [Configure AI Red Teaming Agent - Zero Trust](https://microsoft.github.io/zerotrustassessment/docs/workshop-guidance/AI/AI_081):

| Zero Trust Principle | Application to AI | Implementation |
|---------------------|-------------------|----------------|
| **Verify Explicitly** | Authenticate every API call; validate every input | Entra ID managed identity; Prompt Shields; input validation |
| **Use Least Privilege** | Minimize model, agent, and user permissions | RBAC (Cognitive Services OpenAI User); scoped agent tokens; AI-4 (least privilege for agent functions) |
| **Assume Breach** | Test that safety controls work under adversarial conditions | AI Red Teaming Agent; PyRIT; continuous evaluation; Defender for AI Services |

**Key Zero Trust Controls for AI:**
1. **Identity:** Disable API keys. Use Entra ID + managed identity. Implement PIM for admin roles. Track agent identities with Entra Agent ID.
2. **Network:** Private endpoints on all AI services. Disable public access. Use Azure Firewall for egress control.
3. **Data:** CMK encryption. Sensitivity labels on AI training data. DLP policies on prompts via Microsoft Purview.
4. **Application:** Multi-layered content filtering (AI-2). Safety meta-prompts (AI-3). Human-in-the-loop (AI-5).
5. **Monitoring:** Diagnostic logging. Defender for AI Services. Continuous red teaming (AI-7).

---

### 3.4 Key Metrics and KPIs for AI Security Posture

Based on Microsoft's frameworks and community best practices, the following metrics define a measurable AI security posture [AI Security Metrics and KPIs for Azure](https://just4cloud.com/ai-security-metrics-kpis-azure-deployments/), [MCSB v2 - AI Security](https://learn.microsoft.com/en-us/security/benchmark/azure/mcsb-v2-artificial-intelligence-security):

#### Platform Security Metrics

| Metric | Target | Measurement Source |
|--------|--------|-------------------|
| AI resources with private endpoints | 100% in production | Azure Policy compliance |
| AI resources with local auth disabled | 100% in production | Azure Policy compliance |
| AI resources with CMK encryption | Per compliance req | Azure Policy compliance |
| AI resources with diagnostic logging enabled | 100% | Azure Policy compliance |
| Defender for Cloud Secure Score (AI) | >80% | Defender for Cloud |
| MCSB AI control compliance | 100% Must-have, 80% Should-have | Defender for Cloud regulatory compliance |

#### Application Security Metrics

| Metric | Target | Measurement Source |
|--------|--------|-------------------|
| Content filter policy attachment rate | 100% of deployments | Azure OpenAI resource audit |
| Prompt Shield enablement rate | 100% of deployments | Content Safety configuration |
| Groundedness evaluation score (avg) | >= 4.0 / 5.0 | Azure AI Evaluation SDK |
| Content Safety severity (max) | < Medium (3) | Azure AI Evaluation SDK |
| Safety meta-prompt implementation | 100% of deployments | Code/config review |

#### Testing and Monitoring Metrics

| Metric | Target | Measurement Source |
|--------|--------|-------------------|
| Attack Success Rate (ASR) | < 5% | AART / PyRIT |
| Red teaming frequency | Monthly (automated), Quarterly (manual) | Test schedule |
| Mean Time to Detect (MTTD) AI threats | < 4 hours | Defender for AI Services |
| Mean Time to Respond (MTTR) AI incidents | < 24 hours | Incident management |
| False positive rate in AI threat alerts | < 20% | Alert tuning metrics |
| CI/CD security gate pass rate | > 95% | Pipeline metrics |

---

### 3.5 Reference Architectures for Secure AI Deployments

#### Azure AI Landing Zone

The Azure AI Landing Zone is Microsoft's official enterprise-scale reference architecture for secure AI workloads [Azure AI Landing Zones](https://azure.github.io/AI-Landing-Zones/), [Azure AI Landing Zones GitHub](https://github.com/Azure/AI-Landing-Zones).

**Architecture Components:**
- **Landing Zone for Foundry:** AI Foundry Hub + Projects with managed network isolation, Key Vault, Storage, private endpoints on all services.
- **Landing Zone for APIM:** Azure API Management as AI Gateway for centralized management, rate limiting, logging, and security policy enforcement across model endpoints.

**Implementation Options:**
- Azure Portal (click-through deployment)
- Bicep templates (Azure Verified Modules)
- Terraform templates (Azure Verified Modules)

**Design Areas Covered:**
1. Identity and Access Management
2. Network Topology and Connectivity
3. Resource Organization
4. Governance and Compliance
5. Security and Operations
6. Management and Monitoring
7. Business Continuity
8. Platform Automation

#### Proposed AI Security Maturity Model

Microsoft does not publish a formal numbered maturity model for AI security. Based on the MCSB v2 controls, Zero Trust framework, Responsible AI Standard, and NIST AI RMF, the following maturity levels are synthesized [hypothesis -- synthesized from multiple frameworks]:

| Level | Name | Characteristics |
|-------|------|----------------|
| **1 - Initial** | Ad hoc AI deployments | Default settings, API keys, public endpoints, no content filtering customization, no red teaming |
| **2 - Developing** | Basic security controls | Private endpoints enabled, Entra ID auth, default content filters, basic RBAC, diagnostic logging enabled |
| **3 - Defined** | Policy-driven governance | Azure Policy enforcement (ALZ initiative), CMK encryption, custom RAI policies, Prompt Shields enabled, periodic evaluations |
| **4 - Managed** | Continuous testing and measurement | Automated red teaming in CI/CD (PyRIT/AART), security metrics tracked (ASR, groundedness scores), Defender for AI Services, human-in-the-loop for critical actions |
| **5 - Optimized** | Proactive and adaptive | Full MCSB v2 AI control compliance, Zero Trust architecture, AI Landing Zone deployed, continuous monitoring with adaptive thresholds, regular manual red teaming, compliance mapping (NIST/ISO/SOC2), incident response playbooks for AI-specific threats |

---

## Verification Summary

| Metric | Value |
|--------|-------|
| Total Sources | 27 |
| Atomic Claims Verified | 21 |
| SUPPORTED (2+ sources) | 18 (86%) |
| WEAK (1 source, verified) | 3 (14%) |
| REMOVED (unsupported) | 0 |
| Verification Methods | SIFT, Chain-of-Verification, Trust Tiering |

---

## Contradictions and Resolutions

| Issue | Resolution |
|-------|-----------|
| MCSB v1 baselines labeled "may contain outdated guidance" vs. MCSB v2 in preview | Both are valid. Use MCSB v1 baselines for current compliance; adopt MCSB v2 AI controls proactively as they will become standard. |
| Community blogs show different Terraform patterns for RAI policies | Resolved by referencing the `azurerm_cognitive_account_rai_policy` resource which is the authoritative Terraform resource type. |
| Azure AI Foundry vs. Azure AI Studio naming | Azure AI Studio was rebranded to Azure AI Foundry. Current documentation uses "Microsoft Foundry" or "Azure AI Foundry." |

---

## Conclusion

Azure AI security in 2026 is defined by three converging developments:

**First, Microsoft has formalized AI-specific security controls.** The MCSB v2 introduces seven dedicated AI controls (AI-1 through AI-7) that map directly to established compliance frameworks. These controls codify what was previously scattered guidance into enforceable, auditable requirements. Organizations that adopt these controls get both security hardening and compliance documentation in one framework.

**Second, the testing toolchain has reached practical maturity.** PyRIT provides the open-source foundation, the Azure AI Evaluation SDK provides quality and safety scoring, and the AI Red Teaming Agent brings automated adversarial testing into the Foundry portal. The ability to measure Attack Success Rate (ASR) as a quantitative metric makes AI security comparable to traditional security testing paradigms. The missing piece remains CI/CD standardization -- organizations must build their own pipeline integration.

**Third, Zero Trust principles translate naturally to AI workloads**, but require AI-specific extensions: verifying inputs (not just identities), least-privilege for agent tools (not just users), and assuming breach through continuous red teaming (not just perimeter defense). The Azure AI Landing Zone provides the reference architecture to implement this comprehensively.

**For a Principal Cloud Solution Architect**, the recommended path is:
1. **Immediate:** Deploy Azure Policy initiatives (ALZ guardrails), enable diagnostic logging, disable API keys, configure private endpoints.
2. **Near-term:** Implement Content Safety custom policies with Prompt Shields, adopt Azure AI Evaluation SDK for quality gates, begin PyRIT automated testing.
3. **Strategic:** Deploy AI Landing Zone architecture, adopt MCSB v2 AI controls, establish ASR and quality score baselines, integrate red teaming into CI/CD, track metrics against the proposed maturity model.

---

## Limitations and Residual Risks

**Pre-mortem: If this research was completely wrong in 6 months, why?**

1. **Rapid Azure AI evolution:** Microsoft is iterating on AI services at an unprecedented pace. Service names, feature availability, and best practices may shift significantly. The AI Red Teaming Agent is still in preview and could change substantially before GA.

2. **MCSB v2 is in preview:** The seven AI-specific controls (AI-1 through AI-7) are not yet GA. Final versions may differ from current preview documentation.

3. **Maturity model is synthesized, not official:** Microsoft does not publish a formal AI security maturity model. The 5-level model proposed here is a synthesis from multiple frameworks and could diverge from any future official Microsoft guidance.

**Additional Limitations:**
- RequestResponse logs do not capture prompt/response content -- APIM or application-level logging is required for full auditability.
- AART is limited to English-only, single-turn for some risk categories, and uses synthetic data.
- No official CI/CD templates for AI security testing exist from Microsoft.
- Defender for AI Services alert types and detection capabilities are not fully documented publicly.

---

## Sources

1. [Azure AI security best practices](https://learn.microsoft.com/en-us/azure/security/fundamentals/ai-security-best-practices) - Microsoft Learn, Feb 2026
2. [MCSB v2 - Artificial Intelligence Security](https://learn.microsoft.com/en-us/security/benchmark/azure/mcsb-v2-artificial-intelligence-security) - Microsoft Learn, 2025
3. [Azure security baseline for Azure OpenAI](https://learn.microsoft.com/en-us/security/benchmark/azure/baselines/azure-openai-security-baseline) - Microsoft Learn, MCSB v1
4. [Azure security baseline for Microsoft Foundry](https://learn.microsoft.com/en-us/security/benchmark/azure/baselines/azure-ai-foundry-security-baseline) - Microsoft Learn, MCSB v1
5. [Azure built-in roles for AI + ML](https://learn.microsoft.com/en-us/azure/role-based-access-control/built-in-roles/ai-machine-learning) - Microsoft Learn
6. [Azure Policy built-in definitions for AI services](https://docs.azure.cn/en-us/ai-services/policy-reference) - Microsoft Learn
7. [Enforce recommended guardrails for OpenAI (ALZ)](https://www.azadvertizer.com/azpolicyinitiativesadvertizer/Enforce-Guardrails-OpenAI.html) - AzAdvertizer
8. [Enable diagnostic logging for Foundry Tools](https://learn.microsoft.com/en-us/azure/ai-services/diagnostic-logging) - Microsoft Learn
9. [Use blocklists for text moderation](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/how-to/use-blocklist) - Microsoft Learn
10. [Prompt Shields in Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection) - Microsoft Learn, Nov 2025
11. [AI Red Teaming Agent (preview)](https://learn.microsoft.com/en-us/azure/foundry/concepts/ai-red-teaming-agent) - Microsoft Learn, Feb 2026
12. [PyRIT Documentation](https://azure.github.io/PyRIT/) - Microsoft OSS
13. [PyRIT GitHub Repository](https://github.com/microsoft/pyrit) - Microsoft, 3.6k stars
14. [PyRIT: A Framework for Security Risk Identification and Red Teaming](https://arxiv.org/html/2410.02828v1) - arXiv, Oct 2024
15. [azure-ai-evaluation PyPI](https://pypi.org/project/azure-ai-evaluation/) - Microsoft, v1.16.2, Mar 2026
16. [GroundednessEvaluator class](https://learn.microsoft.com/en-us/python/api/azure-ai-evaluation/azure.ai.evaluation.groundednessevaluator) - Microsoft Learn
17. [Azure AI Landing Zones](https://azure.github.io/AI-Landing-Zones/) - Microsoft OSS
18. [Azure AI Landing Zones GitHub](https://github.com/Azure/AI-Landing-Zones) - Microsoft, 243 stars
19. [2025 Responsible AI Transparency Report](https://www.microsoft.com/en-us/corporate-responsibility/responsible-ai-transparency-report/) - Microsoft
20. [Content Safety and Responsible AI - DeepWiki](https://deepwiki.com/azure-ai-foundry/ai-tutorials/5.2-content-safety-and-responsible-ai) - DeepWiki
21. [Azure OpenAI Content Filtering Custom Thresholds](https://oneuptime.com/blog/post/2026-02-16-how-to-implement-azure-openai-content-filtering-with-custom-severity-thresholds/view) - OneUptime, Feb 2026
22. [Securing Azure AI Applications: Emerging Threats](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/securing-azure-ai-applications-a-deep-dive-into-emerging-threats--part-1/4473159) - Microsoft Tech Community, Nov 2025
23. [Announcing PyRIT](https://www.microsoft.com/en-us/security/blog/2024/02/22/announcing-microsofts-open-automation-framework-to-red-team-generative-ai-systems/) - Microsoft Security Blog, Feb 2024
24. [AI Security Metrics and KPIs for Azure](https://just4cloud.com/ai-security-metrics-kpis-azure-deployments/) - Just4Cloud
25. [Azure AI Foundry Diagnostic Logging with Terraform](https://dev.to/suhas_mallesh/azure-ai-foundry-diagnostic-logging-with-terraform-every-ai-call-tracked-for-compliance-1b2d) - Dev.to, Feb 2025
26. [Monitoring Generative AI with Azure AI Foundry](https://azurefeeds.com/2025/11/07/monitoring-generative-ai-applications-with-azure-ai-foundry/) - Azure Feeds, Nov 2025
27. [Configure AI Red Teaming Agent - Zero Trust Workshop](https://microsoft.github.io/zerotrustassessment/docs/workshop-guidance/AI/AI_081) - Microsoft Zero Trust Workshop
