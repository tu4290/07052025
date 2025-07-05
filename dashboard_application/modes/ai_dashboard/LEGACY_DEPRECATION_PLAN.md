# AI Hub — Legacy Deprecation Plan  
_EOTS v2.5 • July 2025_

This document formalises the retirement of **static UI modules** inside  
`dashboard_application/modes/ai_dashboard` that do **not** integrate with the
HuiHui Mixture-of-Experts (MoE) system.  
It also defines the roadmap for migrating essential functionality to the new
Pydantic-first, MoE-aware component set.

---

## 1  Files to Deprecate 📁✖️  

| File | Reason for Deprecation | Action |
|------|-----------------------|--------|
| `ai_dashboard_display_v2_5.py` | Hard-wired demo layout, no MoE hooks | Archive & remove from import paths |
| `ai_hub_layout.py` | Early prototype; uses dict-based props | Archive |
| `layouts.py` | 1 400-line monolith, pre-Pydantic | Replace with modular components |
| `layouts_panels.py` | Static panel builders; direct dict usage | Split & refactor or archive |
| `layouts_metrics.py` | No MoE linkage; heavy dicts | Migrate logic to **Flow/Volatility Panel** |
| `layouts_health.py` | Legacy health cards, not wired to HuiHui monitor | Replace with new Health Panel |
| `layouts_regime.py` | Duplicates regime logic now supplied by HuiHui | Remove — regime handled in Market Compass |
| `callbacks.py` (legacy) | Dict-based callbacks superseded by new Pydantic callbacks | Delete after migration |
| `clean_elegant_dashboard (1).html` | Static HTML mock-up | Move to `/deprecated_static/` |
| `~elite_moe_command_center~.html` | Obsolete concept demo | Move to `/deprecated_static/` |
| `elite_moe_system.py` | Stand-alone demo visual; no live data | Archive |
| Any `constants.py`, `*.md` UI mock docs | Non-functional assets | Archive only |

---

## 2  Files to Refactor for HuiHui Integration 🔄  

| File | Required Refactor | New Target Model |
|------|------------------|------------------|
| `layouts_metrics.py` | Consume `GaugeConfigModel`, `MetricsPanelModel`, stream MoE flow/vol metrics | Flow & Volatility Panels |
| `layouts_health.py` | Replace static values with live `HuiHuiExpertsMonitorModel` + data-pipeline status | Health Panel |
| `layouts_panels.py` (partial) | Extract any still-useful UI helpers into `components.py` | — |
| Remaining Dash callbacks | Must load/return `AIHubStateModel` not dicts | — |

---

## 3  New Modular Replacements ✅  

| Replacement Module | MoE Connectivity | Status |
|--------------------|------------------|--------|
| `market_compass_component.py` | Regime, Flow, Sentiment fusion from 3 experts | Implemented |
| `ai_recommendations_component.py` | Synthesises all 4 MoEs via `MOEUnifiedResponseV2_5` | Implemented |
| `enhanced_ai_hub_layout.py` | Renders from `AIHubStateModel` | Implemented |
| `visualizations.py` (refactored) | Uses `MarketCompassModel`, `GaugeConfigModel` | Implemented |
| `components.py` (refactored) | Exposes `CardStyle`, `TypographyStyle`, etc. | Implemented |
| Upcoming **Flow Metrics Panel** | Options-flow expert → `MetricsPanelModel` | *Planned* |
| Upcoming **Volatility/Gamma Panel** | Market-regime expert → `MetricsPanelModel` | *Planned* |
| Upcoming **System Health Panel** | HuiHui monitor → `HuiHuiExpertsMonitorModel` | *Planned* |

---

## 4  Migration Timeline 🗓️  

| Phase | Scope | Duration | Owner |
|-------|-------|----------|-------|
| **P-0 Audit** | Tag & branch legacy files, create backups | **Day 0** | Core Dev |
| **P-1 Migration Sprint** | Build Flow / Volatility / Health panels, port metrics | **Days 1–4** | Dashboard Team |
| **P-2 Validation Sprint** | Unit tests (`tests/test_ai_dashboard_models.py`), UI QA, HuiHui integration tests | **Days 5–7** | QA & AI Team |
| **P-3 Cut-over** | Remove legacy imports, switch Dash routes to new layout | **Day 8** | DevOps |
| **P-4 Cleanup** | Delete legacy modules in main branch, keep archive tag | **Day 9** | Core Dev |

_Total timeline: **≈ 2 working weeks**._

---

## 5  Backup / Archival Strategy 📦  

1. **Git Tag** — create tag `ai_dashboard_legacy_v2.5` before removal.  
2. **Archive Folder** — move deprecated files to  
   `dashboard_application/modes/ai_dashboard/deprecated_legacy/`  
   (excluded from runtime via `__init__.py` guard).  
3. **Retention** — keep archives **90 days**; auto-delete via CI workflow.  
4. **Read-Only** — set file permissions to read-only; CI fails if imported.  
5. **Documentation** — link this plan and the tag in `docs/refactoring_completion_report.md`.

---

### ✅ Outcome

After executing this plan, **all runtime AI Hub code paths will rely solely on
Pydantic models and live HuiHui MoE data**, guaranteeing consistency with the
system-wide “Zero-Tolerance Fake Data” mandate and eliminating obsolete static
UI modules.
