# HuiHui Integration: Dependency & Integration Map (EOTS v2.5)

## High-Level Overview
The `huihui_integration` package is the advanced AI expert system for EOTS v2.5, providing Market Regime, Options Flow, and Sentiment experts (MOEs). It is designed for modular, Pydantic-first integration with the EOTS core analytics engine, supporting learning, monitoring, and robust configuration.

---

## Directory/Submodule Roles

- **core/**: Base expert classes, model interface, Pydantic schemas, and core logic for expert communication and LLM integration.
- **experts/**: Implements the three main experts (Market Regime, Options Flow, Sentiment), each as a specialized module. Each expert uses core base classes and Pydantic models.
- **orchestrator_bridge/**: Contains the Expert Coordinator, which manages expert consensus, routes requests, and bridges the experts to the EOTS orchestrator (core_analytics_engine).
- **monitoring/**: Tracks usage, safety, security, and Supabase integration. Provides system health and logging for all expert operations.
- **learning/**: Handles feedback loops, performance tracking, and knowledge sharing. Feeds learning and performance data back to the experts for continuous improvement.
- **config/**: Centralized configuration management, including symbol-specific overrides. All modules load and validate config from here.
- **databases/**: Manages expert-specific and shared databases for storing learning, performance, and knowledge data.

---

## Integration Flow

1. **EOTS Core Analytics Engine** invokes `HuiHuiAIIntegrationV2_5` (in `core_analytics_engine/huihui_ai_integration_v2_5.py`), which uses the Orchestrator Bridge to coordinate all experts.
2. **Expert Coordinator** (in `orchestrator_bridge/`) manages requests to the three experts, aggregates their results, and returns unified intelligence to the EOTS system.
3. **Experts** (in `experts/`) perform specialized analysis (regime, flow, sentiment) using core logic and Pydantic models.
4. **Monitoring** (in `monitoring/`) tracks all expert/system health, usage, and logs, reporting issues to the coordinator.
5. **Learning** (in `learning/`) provides feedback and performance data to experts, enabling adaptive learning and improvement.
6. **Config** (in `config/`) is loaded and validated by all modules, ensuring consistent, override-able settings.
7. **Databases** (in `databases/`) store all persistent data for learning, performance, and shared knowledge.

---

## How This Assists HuiHui AI (MOEs) & EOTS Integration
- **Modular Expert Design:** Each expert is independently developed, tested, and improved, but coordinated for consensus and unified output.
- **Pydantic-First Validation:** All data, configs, and results are validated at every boundary, ensuring robustness and schema compliance.
- **Centralized Orchestration:** The Expert Coordinator ensures all experts work together, resolving conflicts and aggregating intelligence.
- **Continuous Learning:** Feedback and performance tracking enable the system to adapt and improve over time.
- **Robust Monitoring:** System health, usage, and security are tracked and logged for reliability and auditability.
- **Configurable & Extensible:** Symbol-specific overrides and modular design allow for rapid adaptation to new instruments or strategies.

---

For a visual dependency graph, see the included Mermaid diagram or request a rendered version. 