# NVIDIA Model Training and Inference at Scale

## Training / Customization Lifecycle

| Phase | NVIDIA SDK / Component | Used For |
|---|---|---|
| Data Creation | NeMo Data Designer | Synthetic data generation for training datasets |
| Privacy-Safe Data | NeMo Safe Synthesizer | Privacy-safe synthetic data generation |
| Data Preparation | NeMo Curator | Data collection, cleaning, preprocessing, and curation |
| Model Fine-Tuning | NeMo Customizer | Fine-tuning and model customization |
| Model Alignment | NeMo RL | Reinforcement learning and model alignment |
| Model Evaluation | NeMo Evaluator | Benchmarking and evaluation of models and agents |
| Safety Validation | NeMo Guardrails | Safety, governance, and policy controls |
| Knowledge Integration | NeMo Retriever | RAG pipelines and enterprise knowledge retrieval |
| Agent Optimization | NeMo Agent Toolkit | Agent evaluation, profiling, observability, and optimization |

## Inferencing at Scale Lifecycle

| Phase | NVIDIA SDK / Component | Used For |
|---|---|---|
| Model Serving | NVIDIA NIM | Production inference microservices for optimized AI model serving |
| Reasoning Models | Nemotron Models | Reasoning foundation models for agent and model development |
| Workflow Execution | NVIDIA Dynamo | Multi-step workflow orchestration and execution |
| Retrieval-Augmented Inference | NeMo Retriever | Connecting inference workflows to enterprise knowledge sources |
| Safety and Policy Enforcement | NeMo Guardrails | Safe and governed inference responses |
| Agent Runtime | OpenShell | Secure runtime environment for executing AI agents |
| Multi-Agent Orchestration | NeMo Agent Toolkit | Connecting, evaluating, profiling, and optimizing agentic workflows |
| Enterprise Agent Blueprint | NVIDIA AI-Q | Connecting AI agents, data, tools, and reasoning workflows |
| Reference Deployments | NVIDIA AI Blueprints | Repeatable reference implementations for industry AI use cases |

## End-to-End Training to Inference Flow

| Stage | Primary Components | Outcome |
|---|---|---|
| 1. Generate Data | NeMo Data Designer, NeMo Safe Synthesizer | Create high-quality or privacy-safe synthetic training data |
| 2. Curate Data | NeMo Curator | Prepare clean, high-quality datasets |
| 3. Customize Model | NeMo Customizer, NeMo RL | Adapt models to domain-specific tasks and business needs |
| 4. Evaluate Model | NeMo Evaluator | Validate accuracy, quality, and readiness |
| 5. Add Safety | NeMo Guardrails | Apply policy, safety, and governance controls |
| 6. Add Enterprise Knowledge | NeMo Retriever | Ground models with enterprise data and RAG pipelines |
| 7. Deploy Inference | NVIDIA NIM | Serve models as optimized inference microservices |
| 8. Orchestrate Workflows | NVIDIA Dynamo, NeMo Agent Toolkit | Execute and optimize multi-step AI or agent workflows |
| 9. Run Securely | OpenShell | Provide a secure execution layer for production AI agents |
| 10. Scale with Blueprints | NVIDIA AI-Q, NVIDIA AI Blueprints | Reuse reference architectures for industry deployments |

## Quick Learning Priority

| Priority | Component | Why Learn First |
|---|---|---|
| 1 | NeMo Curator | Foundation for high-quality training and customization data |
| 2 | NeMo Customizer | Core service for fine-tuning and adapting models |
| 3 | NeMo Evaluator | Required to benchmark and validate model quality |
| 4 | NeMo Guardrails | Needed for safety, governance, and enterprise readiness |
| 5 | NVIDIA NIM | Core platform for scalable inference deployment |
| 6 | NeMo Retriever | Critical for RAG and enterprise knowledge grounding |
| 7 | NeMo Agent Toolkit | Important for agent evaluation, profiling, and optimization |
| 8 | NVIDIA Dynamo | Useful for orchestrating advanced multi-step workflows |
