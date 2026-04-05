# Microsoft Foundry Reference Architecture for MAF

## Introduction

- Create a Reference Architecture for Microsoft Foundry for MAF (Microsoft Agent Framework).
- This architecture will provide a blueprint for building agentic AI applications using Microsoft Foundry and MAF.
- The architecture will cover various components and services that can be used to build, deploy, and manage agentic AI applications in an enterprise environment.
- The architecture will also include best practices for security, governance, and compliance when building agentic AI applications using Microsoft Foundry and MAF.
- Also Shows a end to end architecture for building agentic AI applications using Microsoft Foundry and MAF, including data sources, knowledge base creation, agent development, deployment, and management.
- This architecture can be used as a reference for organizations looking to build agentic AI applications using Microsoft Foundry and MAF, and can be customized based on specific requirements and use cases.

## Architecture Diagram

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/foundryarchv2.jpg 'fine tuning model')

## Architecture Overview

The reference architecture is divided into six major layers that work together to enable enterprise-scale agentic AI applications:

1. **Data Sources** — Ingestion and connectivity layer for enterprise data
2. **Microsoft Foundry – Agent Lifecycle** — Core platform for building, managing, and governing agents
3. **Agent Orchestration (Agent Orch)** — Runtime orchestration and execution layer using MAF
4. **Agent 365** — Central agent gateway bridging data sources and orchestration to user experiences
5. **UX (User Experience)** — Multi-channel delivery layer for end users and customers
6. **Operations & Security** — Cross-cutting concerns for governance, monitoring, compliance, and DevOps

---

## 1. Data Sources

The Data Sources layer is the foundation for all enterprise knowledge and information that agents consume. Building agentic AI at scale requires connecting to diverse, heterogeneous data sources securely and efficiently.

### Components

| Component | Purpose |
|-----------|---------|
| **SharePoint** | Enterprise document management and collaboration platform. Agents can index and retrieve organizational knowledge, policies, SOPs, and unstructured documents stored in SharePoint Online. |
| **API** | Internal APIs that expose structured business data from LOB (Line of Business) applications such as ERP, CRM, HRIS, and custom microservices. |
| **Data** | Structured data stores including Azure SQL, Azure Data Lake, Fabric OneLake, Synapse, or any relational/NoSQL data source that agents need for grounding responses. |
| **External API** | Third-party and partner APIs outside the organizational boundary—weather services, market data feeds, regulatory databases, SaaS connectors, etc. |
| **MCP (Model Context Protocol)** | An open protocol that standardizes how agents discover and invoke tools, data sources, and prompts. MCP servers expose capabilities that agents can call at runtime without hard-coded integrations. |

### Design Considerations for Scale

- **Data Indexing & Vectorization**: Use Azure AI Search or Foundry IQ to create vector indexes over SharePoint and Data stores so agents can perform semantic retrieval (RAG) at low latency.
- **Data Freshness**: Implement incremental indexing pipelines (e.g., change feed from Cosmos DB, delta crawl for SharePoint) to keep agent knowledge current without full re-indexing.
- **Access Control**: Propagate user identity through the stack so agents only return data the calling user is authorized to see. Use SharePoint RBAC and API-level OAuth scopes.
- **Rate Limiting & Circuit Breakers**: Protect external APIs with retry policies, circuit breakers, and caching via Azure API Management (APIM) or AI Gateway.
- **MCP Server Hosting**: Deploy MCP servers as containerized services in Azure Container Instances or Azure Kubernetes Service. Register them in the Foundry tool catalog for dynamic agent discovery.

---

## 2. Microsoft Foundry – Agent Lifecycle

This is the core platform layer where agents are designed, built, tested, evaluated, and governed. Microsoft Foundry provides a unified control plane for the entire agent lifecycle.

### Components

| Component | Purpose |
|-----------|---------|
| **MCP** | Model Context Protocol integration within Foundry allows agents to dynamically discover and bind to tools and data sources at runtime, enabling plug-and-play extensibility. |
| **Agents (A2A)** | Agent-to-Agent communication protocol enabling multi-agent collaboration. Agents can delegate sub-tasks to specialized peer agents, enabling complex workflows that span domains. |
| **Foundry IQ** | The knowledge and retrieval engine within Foundry. Foundry IQ manages vector indexes, knowledge bases, and grounding data that agents use for RAG-based responses. |
| **Azure OpenAI** | The LLM backbone providing access to GPT-4o, GPT-4.1, o3, o4-mini, and other frontier models. Used for reasoning, planning, code generation, and natural language understanding. |
| **AI Gateway** | A centralized gateway that manages model routing, load balancing, rate limiting, and failover across multiple Azure OpenAI deployments and regions. Critical for scale and resilience. |
| **Redis** | In-memory caching and session state management. Redis stores agent conversation history, intermediate reasoning state, and frequently accessed data to reduce latency and LLM token consumption. |
| **Tracing** | Distributed tracing infrastructure for observing agent behavior end-to-end. Captures each step of agent reasoning—tool calls, LLM invocations, retrieval queries—for debugging and optimization. |
| **Tokens** | Token usage tracking and quota management. Monitors consumption across agents, teams, and projects to enforce budgets and prevent runaway costs. |
| **Security** | Built-in security features including content filtering, prompt shields, groundedness detection, and jailbreak protection to ensure agents operate within safe boundaries. |
| **RedTeam / Eval** | Automated red-teaming and evaluation pipelines. Systematically test agents for harmful outputs, hallucinations, bias, and adversarial robustness before production deployment. |
| **MarketPlace Model Catalog** | Access to 1,900+ open-source and proprietary models from the Foundry model catalog. Enables model selection, comparison, and fine-tuning for domain-specific agent capabilities. |

### Design Considerations for Scale

- **Multi-Agent Patterns**: Use A2A for fan-out/fan-in patterns where a supervisor agent delegates to specialist agents (e.g., a finance agent, HR agent, IT agent) and aggregates results. Define clear agent boundaries and protocols.
- **AI Gateway for Multi-Region**: Deploy AI Gateway with priority-based routing across multiple Azure OpenAI deployments (e.g., East US primary, West Europe secondary) to handle regional capacity constraints and provide disaster recovery.
- **Caching Strategy**: Use Redis to cache tool results, embeddings, and agent session state. Implement TTL-based invalidation aligned with data freshness requirements.
- **Tracing at Scale**: Integrate OpenTelemetry-based tracing with Azure Monitor / Application Insights. Use sampling strategies (e.g., 10% in production, 100% in staging) to balance observability with cost.
- **Continuous Evaluation**: Run evaluation pipelines in CI/CD. Define quality metrics (groundedness, relevance, coherence, fluency) and set pass/fail thresholds. Block deployments that regress below baseline.
- **Red-Teaming Cadence**: Schedule automated red-teaming runs weekly and after every model or prompt change. Use Azure AI Content Safety and custom adversarial datasets.
- **Fine-Tuning**: Use the Model Catalog to fine-tune smaller models (e.g., GPT-4.1-nano, Phi-4) for high-volume, latency-sensitive agent tasks. Deploy fine-tuned models behind AI Gateway for unified routing.

---

## 3. Agent Orchestration (Agent Orch)

The orchestration layer is the runtime execution environment where agents are hosted, scaled, and managed. This layer uses MAF (Microsoft Agent Framework) as the core orchestration SDK.

### Components

| Component | Purpose |
|-----------|---------|
| **MAF (Microsoft Agent Framework)** | The SDK and runtime for building agentic applications. MAF provides agent abstractions, tool binding, memory management, planning strategies, and multi-turn conversation handling. |
| **Cosmos DB** | Globally distributed database for persisting agent state, conversation history, user preferences, and session data. Provides low-latency access with multi-region writes for global applications. |
| **Container Instances** | Azure Container Instances (ACI) or Azure Container Apps (ACA) for hosting agent runtime containers. Enables rapid scaling, isolation, and serverless execution of agent workloads. |

### Design Considerations for Scale

- **Horizontal Scaling**: Deploy agent containers in Azure Container Apps with auto-scaling rules based on HTTP request count, CPU, or custom metrics (e.g., active agent sessions). Set min/max replica counts per agent type.
- **State Management with Cosmos DB**: Use Cosmos DB with partition keys aligned to user/session IDs for optimal throughput distribution. Enable change feed for event-driven architectures where downstream systems react to agent state changes.
- **Multi-Region Deployment**: Deploy Cosmos DB with multi-region writes and agent containers in multiple Azure regions. Use Azure Front Door or Traffic Manager for global load balancing.
- **Agent Isolation**: Run each agent type in its own container with dedicated resource limits (CPU, memory). Use Dapr sidecars for service-to-service communication, state management, and pub/sub messaging between agents.
- **Long-Running Workflows**: For complex agent tasks that span minutes or hours (e.g., multi-step research, approval workflows), use Durable Functions or workflow engines to manage state, retries, and human-in-the-loop checkpoints.
- **Container Image Management**: Store agent container images in Azure Container Registry (ACR). Implement image scanning, signing, and promotion pipelines (dev → staging → production).

---

## 4. Agent 365

Agent 365 sits at the center of the architecture as the unified agent gateway. It bridges data sources, the Foundry lifecycle platform, and the orchestration layer with the user experience channels.

### Role & Responsibilities

- **Unified Agent Endpoint**: Provides a single entry point for all agent interactions regardless of the source channel (Teams, Copilot, custom apps).
- **Ops Users Access**: Operations teams interact with Agent 365 for monitoring, configuration, and manual intervention in agent workflows.
- **Routing & Dispatch**: Routes incoming requests to the appropriate agent or agent group based on intent classification, user context, and business rules.
- **Session Management**: Maintains cross-channel session continuity so a user can start a conversation in Teams and continue in a custom app.

### Design Considerations for Scale

- **Throttling & Fairness**: Implement per-tenant and per-user throttling to prevent noisy-neighbor problems in multi-tenant deployments.
- **Ops Dashboards**: Provide real-time dashboards for ops users showing agent health, request volumes, error rates, latency percentiles, and token consumption.
- **Canary Deployments**: Use Agent 365 routing rules to direct a percentage of traffic to new agent versions for gradual rollout and validation.

---

## 5. UX (User Experience)

The UX layer defines how end users, internal employees, and external customers interact with agentic AI applications. The architecture supports multiple channels to meet users where they are.

### Components

| Component | Purpose |
|-----------|---------|
| **Teams** | Microsoft Teams integration for conversational agent experiences within the enterprise collaboration platform. Agents appear as bots or message extensions. |
| **Copilot** | Microsoft 365 Copilot integration. Agents extend Copilot capabilities with domain-specific knowledge and actions, surfacing within Word, Excel, PowerPoint, Outlook, and other M365 apps. |
| **Copilot Studio** | Low-code/no-code platform for building and customizing agent experiences. Business users can create and modify agent behaviors without developer involvement. |
| **App Services** | Azure App Service for hosting custom web applications, APIs, and portals that embed agent experiences via SDKs or REST APIs. |
| **APIM (API Management)** | Azure API Management for exposing agent capabilities as managed, secured, and versioned APIs to internal and external consumers. |

### User Personas

| Persona | Channel | Description |
|---------|---------|-------------|
| **Users** | Teams, Copilot, Custom Apps | Internal employees who interact with agents for productivity, knowledge retrieval, and task automation. |
| **Apps** | App Services, APIM | Custom applications and third-party systems that consume agent capabilities programmatically. |
| **Customers** | Custom Apps, APIM | External customers who interact with customer-facing agents for support, self-service, and engagement. |

### Design Considerations for Scale

- **Channel-Specific Optimization**: Tailor agent responses for each channel. Teams messages have formatting constraints; Copilot integrations use Adaptive Cards; custom apps may render rich interactive UIs.
- **Adaptive Cards & Rich UX**: Use Adaptive Cards in Teams and Copilot for structured responses, forms, and interactive elements that guide users through multi-step workflows.
- **APIM Policies**: Configure APIM with rate limiting, JWT validation, request/response transformation, and caching policies. Use subscription keys and OAuth for external consumer authentication.
- **Progressive Disclosure**: Design agent UX to show concise answers first with options to drill deeper. Avoid overwhelming users with verbose responses.
- **Accessibility**: Ensure all agent interfaces meet WCAG 2.1 AA standards. Support screen readers, keyboard navigation, and high-contrast modes.

---

## 6. Agentic AI Business Application

The right side of the architecture represents the end-to-end agentic AI business application that serves three key personas:

### Target Audiences

- **Users (Internal Employees)**: Leverage agents for day-to-day tasks—summarizing documents, querying databases, automating workflows, generating reports, and getting contextualized answers from enterprise knowledge bases.
- **Apps (Application Integration)**: Backend services and custom applications that integrate agent capabilities via APIs. Enables scenarios like automated ticket routing, intelligent document processing, and real-time decision support.
- **Customers (External End Users)**: Customer-facing agents for support chatbots, self-service portals, personalized recommendations, and proactive engagement.

### Design Considerations for Scale

- **Multi-Tenant Architecture**: Design agents to serve multiple business units or customers with data isolation, per-tenant configuration, and independent scaling.
- **Personalization**: Use Cosmos DB to store user preferences, interaction history, and learned behaviors. Agents should adapt their tone, depth, and proactivity based on user profiles.
- **SLA Tiers**: Define different SLA tiers (e.g., P1 for customer-facing agents with 99.99% availability, P2 for internal productivity agents with 99.9%) and architect accordingly.

---

## 7. Operations & Security

The Operations & Security layer spans the entire architecture, providing cross-cutting capabilities for governance, compliance, monitoring, and secure operations.

### Components

| Component | Purpose |
|-----------|---------|
| **Defender AI** | AI-specific threat protection. Detects and mitigates prompt injection attacks, data exfiltration attempts, and adversarial inputs targeting AI models and agents. |
| **Sentinel** | Cloud-native SIEM (Security Information and Event Management). Aggregates security signals across the agentic AI stack, enables threat hunting, and triggers automated incident response via playbooks. |
| **Container Registries** | Azure Container Registry (ACR) for storing, scanning, and distributing agent container images. Enforces image signing and vulnerability scanning before deployment. |
| **Monitor** | Azure Monitor for infrastructure and application monitoring. Collects metrics, logs, and traces from all architecture layers. Powers custom dashboards and alerts. |
| **DevOps** | Azure DevOps or GitHub Actions for CI/CD pipelines. Automates agent build, test, evaluation, and deployment workflows with quality gates and approval stages. |
| **Key Vault** | Azure Key Vault for centralized secrets management. Stores API keys, connection strings, certificates, and model endpoint credentials with HSM-backed encryption. |
| **App Insights** | Application Insights for deep application performance monitoring (APM). Tracks agent response times, failure rates, dependency calls, and custom telemetry like token usage and groundedness scores. |
| **Entra ID** | Microsoft Entra ID (Azure AD) for identity and access management. Provides SSO, MFA, Conditional Access, and managed identity for secure, passwordless agent-to-service authentication. |
| **GitHub** | Source control, collaboration, and CI/CD platform. Hosts agent source code, prompt templates, evaluation datasets, and infrastructure-as-code definitions. Enables code review, branch protection, and automated workflows. |
| **Purview** | Microsoft Purview for data governance, cataloging, and compliance. Tracks data lineage through agent pipelines, enforces data classification policies, and supports regulatory compliance (GDPR, HIPAA, SOX). |

### Design Considerations for Scale

- **Zero Trust Architecture**: Implement Zero Trust principles throughout—verify explicitly (Entra ID + MFA), use least-privilege access (RBAC + managed identities), and assume breach (Defender AI + Sentinel).
- **Managed Identities Everywhere**: Eliminate secrets from code by using Azure Managed Identities for agent containers to access Cosmos DB, Key Vault, Azure OpenAI, and other services. Store any remaining secrets in Key Vault with automated rotation.
- **CI/CD Pipeline Design**:
  1. **Build**: Compile agent code, run unit tests, lint prompt templates
  2. **Evaluate**: Run automated evaluation suites (groundedness, relevance, safety) against test datasets
  3. **Red-Team**: Execute automated adversarial testing
  4. **Stage**: Deploy to staging environment with synthetic traffic
  5. **Approve**: Manual or automated approval gate based on quality metrics
  6. **Deploy**: Blue/green or canary deployment to production
  7. **Monitor**: Post-deployment validation with smoke tests and metric comparison
- **Observability Stack**: Combine Azure Monitor + App Insights + Tracing for a three-pillar observability approach (metrics, logs, traces). Create unified dashboards in Azure Workbooks or Grafana.
- **Security Monitoring**: Route all agent interaction logs to Sentinel. Create detection rules for anomalous patterns—unusual token consumption, repeated prompt injection attempts, data exfiltration indicators.
- **Compliance & Audit**: Use Purview to catalog all data sources agents access. Maintain audit logs of every agent action, tool invocation, and data retrieval for regulatory compliance.
- **Cost Management**: Use Azure Cost Management + token tracking to monitor per-agent, per-team, and per-tenant costs. Set budget alerts and implement automated throttling when approaching limits.

---

## Building at Scale — End-to-End Walkthrough

### Step 1: Data Foundation

1. Identify all enterprise data sources (SharePoint, databases, APIs, third-party services)
2. Create data connectors using MCP servers or native Foundry connectors
3. Build vector indexes in Foundry IQ for unstructured data (documents, emails, wikis)
4. Set up incremental data pipelines for real-time freshness
5. Implement access control propagation from source to agent response

### Step 2: Agent Development

1. Define agent personas and responsibilities (e.g., HR Agent, IT Support Agent, Finance Agent)
2. Build agents using MAF SDK with clear tool definitions, system prompts, and guardrails
3. Configure A2A protocols for multi-agent collaboration scenarios
4. Integrate Foundry IQ knowledge bases for RAG-grounded responses
5. Select appropriate models from Azure OpenAI or Model Catalog based on latency/quality trade-offs

### Step 3: Testing & Evaluation

1. Create evaluation datasets with ground-truth answers
2. Run automated quality evaluations (groundedness, relevance, coherence, fluency)
3. Execute red-teaming suites for safety and adversarial robustness
4. Conduct human-in-the-loop evaluation for subjective quality assessment
5. Establish baseline metrics and quality gates for CI/CD

### Step 4: Deployment

1. Containerize agents and push images to Azure Container Registry
2. Deploy to Azure Container Apps or Container Instances with auto-scaling
3. Configure Cosmos DB for state persistence with appropriate partition strategies
4. Set up AI Gateway for model routing and load balancing across regions
5. Deploy behind APIM for API governance and external access

### Step 5: Integration & UX

1. Deploy Teams bot manifests for enterprise chat integration
2. Build Copilot plugins for M365 integration
3. Configure Copilot Studio for citizen-developer customization
4. Deploy custom web apps on App Services with embedded agent UX
5. Publish managed APIs via APIM for application and partner consumption

### Step 6: Operations

1. Configure Azure Monitor alerts for latency, error rates, and availability
2. Set up App Insights for agent-specific telemetry (token usage, tool call success rates)
3. Enable Sentinel integration for security monitoring and incident response
4. Implement Purview data governance policies for compliance
5. Establish operational runbooks for common failure scenarios and escalation paths

---

## Scaling Patterns Summary

| Pattern | When to Use | Azure Services |
|---------|------------|----------------|
| **Horizontal Pod Scaling** | High request volume | Container Apps, AKS |
| **Multi-Region Active-Active** | Global availability, low latency | Cosmos DB, Front Door, AI Gateway |
| **AI Gateway Load Balancing** | Model capacity limits | AI Gateway with priority routing |
| **Caching & Session State** | Reduce LLM calls, improve latency | Redis Cache |
| **Event-Driven Fan-Out** | Complex multi-agent workflows | Event Grid, Service Bus, A2A |
| **Blue/Green Deployment** | Zero-downtime agent updates | Container Apps revisions, APIM |
| **Circuit Breaker** | External API resilience | APIM policies, Polly (.NET) |

---

## Security Best Practices Summary

| Practice | Implementation |
|----------|---------------|
| **Identity** | Entra ID + Managed Identities for all service-to-service auth |
| **Secrets** | Key Vault with automated rotation; no secrets in code or config |
| **Content Safety** | Azure AI Content Safety for input/output filtering |
| **Prompt Security** | Prompt Shields for injection detection |
| **Network** | Private endpoints, VNet integration, NSGs |
| **Data Protection** | Encryption at rest (CMK) and in transit (TLS 1.3) |
| **Audit** | Purview data lineage + Sentinel security logs |
| **Compliance** | Regulatory controls via Azure Policy and Purview |

---

## Conclusion

This reference architecture provides a comprehensive blueprint for building enterprise-scale agentic AI applications using Microsoft Foundry and MAF. The layered approach ensures separation of concerns while the cross-cutting Operations & Security layer provides the governance and observability needed for production workloads. Organizations should start with a focused use case, validate the architecture end-to-end, and then expand horizontally to additional agents and data sources as maturity grows.