# HuiHui AI System – Full End-to-End Integration Plan  
_EOTS v2.5 · July 2025_

---

## 1. Executive Summary
This technical blueprint details the steps required to:
* Eliminate **all legacy/obsolete AI-Dashboard files**.
* Achieve **seamless, real-time integration** between the **4 HuiHui MoEs** and the EOTS UI/analytics stack.
* Establish a **dual-tier data architecture** — Supabase (long-term) + EnhancedCacheManager (intraday).
* Build robust data pipelines for **live collection → training → expert enhancement**.
* Provide a **timeline & priority matrix** to deliver an elite, self-optimising expert system.

---

## 2. Legacy Removal & Codebase Hygiene
| Category | Files / Folders | Action | Rationale |
|----------|-----------------|--------|-----------|
| Prototype layouts | `layouts.py`, `layouts_panels.py`, `layouts_metrics.py`, `layouts_health.py`, `layouts_regime.py` | Delete / archive tag `ai_dashboard_legacy_v2.5` | Fully superseded by modular Pydantic panels |
| Old demos | `ai_dashboard_display_v2_5.py`, `ai_hub_layout.py` | Delete | Non-MoE, dict-based |
| Static HTML mocks | `clean_elegant_dashboard (1).html`, `~elite_moe_command_center~.html` | Move to `/deprecated_static/` | UI reference only |
| Stand-alone visual | `elite_moe_system.py` | Archive | Supplanted by Market Compass |
| Any file with hard-coded data / dict access | (scan output) | Remove or refactor | Zero-dict policy |

**Process**  
1. Create archive branch & tag.  
2. Move files to `dashboard_application/modes/ai_dashboard/deprecated_legacy/`.  
3. Add `__init__.py` guard that raises `ImportError` if imported.  
4. Update CI to block new imports from deprecated folder.

---

## 3. Target Architecture Overview
```
┌───────────────┐   live JSON   ┌────────────────┐
│  Data Feeds   │ ───────────▶ │  Intraday L1   │
│ (Tradier etc) │              │  Cache (ECM)   │
└───────────────┘              │  <60 min TTL   │
         │                     └────────────────┘
         ▼                              │
 raw market data                 cached enriched data
         │                              ▼
┌────────────────────┐   asyncio   ┌──────────────────┐
│  ITS Orchestrator  │───────────▶│  HuiHui 4-MoE     │
└────────────────────┘            │  (regime, flow,   │
         │                        │   sentiment, meta)│
 MOE unified response             └──────────────────┘
         ▼                              │
┌────────────────────┐     write & read  │
│  AI Hub Dashboard  │◀──────────────────┘
└────────────────────┘
         │
 long-term snapshots
         ▼
┌────────────────────────────┐
│  Supabase (Postgres)       │
│  • historical metrics      │
│  • training corpora        │
│  • educational materials   │
└────────────────────────────┘
```

---

## 4. Database Architecture

### 4.1 Supabase (Long-Term)
| Schema | Tables | Purpose |
|--------|--------|---------|
| `moe_analysis` | `expert_responses` (PK id, expert_id, symbol, jsonb details, confidence, created_at) | Persist every MoE result for back-testing |
| `market_data` | `underlying_ohlc`, `options_chain` | Raw daily snapshots |
| `training` | `labeled_examples`, `feature_vectors` | Supervised data for fine-tuning |
| `education` | `documents`, `embeddings` | RAG corpus (PDF, MD, notebooks) |
| `usage_stats` | `huihui_usage`, `orchestrator_stats` | Token & latency tracking |
| `system_health` | `alerts`, `uptime_logs` | Monitoring |

*All JSON columns are **validated** against their Pydantic schemas before insert.*

### 4.2 Enhanced Cache Manager (Intraday)
* Path: `cache/enhanced_v2_5/`
* TTL tiers:  
  * **5 min** – raw provider payload  
  * **1 min** – enriched metrics bundles  
  * **30 sec** – MoE expert responses  
* LRU in-memory + disk spill, size caps (100 MB RAM / 1 GB disk)  
* Shared between **EOTS metrics** and **HuiHui experts** via singleton factory.

---

## 5. Live Data Collection for 4 MoEs
| Expert | Collector Function | Source | Frequency | Cache Layer |
|--------|-------------------|--------|-----------|-------------|
| Market Regime | `MarketRegimeExpert.fetch_features()` | ConvexValue + Tradier | every N sec (control panel) | ECM 1 min |
| Options Flow | `OptionsFlowExpert.fetch_flow_vectors()` | Options chain (Level 2) | same | ECM 30 sec |
| Sentiment | `MarketIntelligenceExpert.fetch_news_sentiment()` | News APIs, social streams | same | ECM 5 min |
| Meta-Orch | consumes 3 experts | — | same | ECM 30 sec |

---

## 6. Training & Educational Material Integration

### 6.1 Data Ingestion
1. **ETL workers** parse research papers, markdown docs → chunk → embed (OpenAI/BGE) → store in `education.documents / embeddings`.
2. Daily batch loads historical market & options data into `market_data.*`.

### 6.2 Training Loops
| Stage | Tooling | Output |
|-------|---------|--------|
| Feature extraction | `core_analytics_engine.eots_metrics` | feature_vectors |
| Label generation | Orchestrator / manual annotations | labeled_examples |
| Model fine-tuning | `huihui_integration.learning.feedback_loops` | updated expert weights / checkpoints |
| Evaluation | `tests/test_huihui_performance.py` | performance_metrics |

### 6.3 Continuous Learning
* AdaptiveLearningManager monitors prediction-vs-outcome, writes back to Supabase
* Periodic **vector-based retrieval** of educational corpus to augment expert prompts (RAG)

---

## 7. Data-Flow Pipelines

```
[Provider APIs] → (Fetch) → ECM → (Pre-process) → Metrics Calculator
                 → (Bundle) → ITS Orchestrator → HuiHui Experts
                 → (Response) → ECM (30 sec) → AI Hub
                 ↘ Supabase (hourly) ↗
                        ↑
             (Nightly back-fill)  (Training jobs)
```

Each arrow = Pydantic-validated JSON / MsgPack for zero-dict safety.

---

## 8. Integration Points Matrix

| Component | Reads From | Writes To | Notes |
|-----------|-----------|-----------|-------|
| **Data Fetchers** | Provider APIs | ECM (raw) | Fail-fast on missing fields |
| **Metrics Calculator** | ECM (raw) | ECM (enriched) | Adds advanced metrics |
| **ITS Orchestrator** | ECM (enriched) | HuiHui experts RPC | Handles retries, rate-limit |
| **HuiHui Expert** | ECM (features) | ECM (responses), Supabase (archive) | Async ⚡ |
| **AI Hub** | ECM (responses) | — | Renders via Pydantic models |
| **Adaptive Learner** | Supabase (labeled data) | Supabase (weights) | Periodic tuning |
| **Compliance Tracker** | Callbacks | Supabase (usage_stats) | Alerts on anomalies |

---

## 9. Performance Optimisation Strategy
* **Cache-first Reads** – experts check ECM before external calls.
* **Vectorised Numpy ops** in compass & gauge generation.
* **Batch inserts** to Supabase (COPY protocol or `asyncpg.copy_records_to_table`).
* **Web-socket streaming** for high-freq data to ECM; fall back to REST if latent.
* **Connection pooling** – shared `asyncpg` pool per expert to Supabase (max 20).
* **Supabase RLS** – fine-grained access without latency penalty.

---

## 10. Implementation Timeline & Priority Matrix

| Phase | Duration | Priority | Deliverables |
|-------|----------|----------|--------------|
| **P-0 Legacy Cleanup** | 1 day | P0 | Archive & block old modules |
| **P-1 DB Bootstrapping** | 2 days | P0 | Supabase schemas, ECM singleton refactor |
| **P-2 Live Data Agents** | 3 days | P0 | Fetchers write to ECM; MoEs read from ECM |
| **P-3 Panel Wiring** | 2 days | P1 | Flow & Volatility panels live; Row-3 health panel |
| **P-4 Training Pipeline** | 4 days | P1 | ETL → labeled_examples → fine-tune loop |
| **P-5 Educational Corpus** | 2 days | P2 | RAG ingestion of docs/PDFs |
| **P-6 Compliance & Perf** | 2 days | P2 | Supabase usage logs, compass optimisation |
| **P-7 QA & Deployment** | 1 day | P0 | End-to-end tests, CI green, prod release |

### RACI (high-level)
| Role | Legacy Cleanup | DB/Cache | Fetchers/MoE | Training | Deployment |
|------|----------------|----------|-------------|----------|------------|
| Core Dev | R | A | C | I | C |
| DB Engineer | C | A | C | C | I |
| AI Researcher | I | C | C | A | C |
| DevOps | C | C | I | I | A |
(R=Responsible, A=Accountable, C=Consulted, I=Informed)

---

## 11. Success Criteria
* **100 % Pydantic-validated data flow**
* < 500 ms MoE round-trip for single-symbol analysis
* ≥ 95 % expert-response persistence rate to Supabase
* Dash UI refresh aligned with control-panel interval, zero drift
* Automated nightly re-training job completes in < 60 min
* Unit + integration tests > 90 % coverage

---

## 12. Risk & Mitigation
| Risk | Impact | Mitigation |
|------|--------|-----------|
| Supabase latency spikes | Delays UI updates | Local ECM fall-back + async retries |
| Provider API rate-limits | Data gaps | Dual-provider fetcher, cached back-off |
| Schema drift | Validation failures | Schema versioning + CI contract tests |
| Large embeddings cost | ↑ Spend | Use open-source BGE + local vector DB for cold storage |

---

### 🚀 _With this plan, the HuiHui expert suite evolves into a continuously learning, data-driven, elite AI system—fully integrated, highly performant, and future-proof._  
