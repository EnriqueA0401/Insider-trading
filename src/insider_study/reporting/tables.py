"""Render results.json into markdown tables and figures (results/<mode>/tables.md)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from insider_study.reporting import figures


def _pct(x: Any, nd: int = 1) -> str:
    return "—" if x is None or pd.isna(x) else f"{100 * float(x):.{nd}f}%"


def _num(x: Any, nd: int = 2) -> str:
    return "—" if x is None or pd.isna(x) else f"{float(x):.{nd}f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def portfolio_table(res: dict[str, Any], keys: list[str], split: str) -> str:
    rows = []
    for k in keys:
        r = res["portfolios"].get(k, res.get("robustness", {}).get(k, {})).get(split, {})
        if "alpha_net_annual" not in r:
            continue
        rows.append(
            [
                k,
                _num(r["months"], 0),
                _num(r["avg_holdings"], 0),
                _pct(r["cagr_net"]),
                _num(r["sharpe_net"]),
                _pct(r["max_drawdown"]),
                _pct(r["alpha_gross_annual"]),
                _pct(r["alpha_net_annual"]),
                _num(r["alpha_net_t"]),
                _pct(r["cost_annual"]),
                _pct(r["turnover_monthly"]),
                _num(r["betas"]["smb"]),
                _num(r["betas"]["hml"]),
            ]
        )
    return _table(
        [
            "Cartera",
            "Meses",
            "Nº medio",
            "CAGR neto",
            "Sharpe neto",
            "MaxDD",
            "α bruto",
            "α neto",
            "t(α neto)",
            "Coste anual",
            "Turnover/mes",
            "β SMB",
            "β HML",
        ],
        rows,
    )


def tests_table(res: dict[str, Any], split: str) -> str:
    rows = []
    for name, t in res["tests"][split].items():
        rows.append(
            [
                name,
                t.get("spread", t.get("portfolio", "")),
                _num(t["months"], 0),
                _pct(t["alpha_annual"]),
                _num(t["alpha_t"]),
                _num(t["alpha_p"], 3),
                _num(t.get("holm_p"), 3) if "holm_p" in t else "secundario",
            ]
        )
    return _table(["Test", "Contraste", "Meses", "α neto anual", "t", "p (1 cola)", "p Holm"], rows)


def render(mode: str, results_dir: Path = Path("results")) -> Path:
    base = results_dir / mode
    res = json.loads((base / "results.json").read_text())
    net = pd.read_parquet(base / "monthly_net_returns.parquet")
    figs = base / "figures"
    figs.mkdir(exist_ok=True)
    splits = ["train", "validation"] if mode == "dev" else ["holdout"]
    out: list[str] = [f"# Tablas — modo `{mode}`\n", f"Config hash: `{res['config_hash']}`\n"]
    p = res["prepare"]
    out.append("## Datos\n")
    out.append(
        f"- Eventos elegibles: {p['n_events']:,} (más {p['n_plan_events']:,} compras 10b5-1 segregadas)"
    )
    out.append(f"- Estado de precio: {p['price_status']}")
    out.append(f"- Proxy de tamaño: {p['size_proxy']}\n")
    main = [
        "ALL",
        "CFO",
        "NONCFO",
        "CEO",
        "DIR",
        "OPP",
        "ROUT",
        "CFO_OPP",
        "CFO_ROUT",
        "CONV_HI",
        "CONV_LO",
        "CLUSTER",
        "ISOLATED",
        "CFO_MIDLARGE",
        "ALL_MIDLARGE",
        "CFO_MICRO",
        "CFO_SMALL",
        "CFO_MID",
        "CFO_LARGE",
        "ALL_MICRO",
        "ALL_SMALL",
        "ALL_MID",
        "ALL_LARGE",
    ]
    for s in splits:
        out.append(f"## Carteras — {s}\n")
        out.append(portfolio_table(res, main, s) + "\n")
        out.append(f"## Contrastes de hipótesis — {s}\n")
        out.append(tests_table(res, s) + "\n")
    rob = res["robustness"]
    rkeys = [k for k in rob if k not in ("spreads", "subperiods")]
    for s in splits:
        out.append(f"## Robustez — {s}\n")
        rows = []
        for k in rkeys:
            r = rob[k].get(s, {})
            if "alpha_net_annual" in r:
                rows.append(
                    [
                        k,
                        _num(r["months"], 0),
                        _pct(r["alpha_gross_annual"]),
                        _pct(r["alpha_net_annual"]),
                        _num(r["alpha_net_t"]),
                        _num(r["sharpe_net"]),
                    ]
                )
        out.append(
            _table(["Variante", "Meses", "α bruto", "α neto", "t", "Sharpe neto"], rows) + "\n"
        )
        rows = [
            [n, t["spread"], _pct(t["alpha_annual"]), _num(t["alpha_t"])]
            for n, t in rob["spreads"][s].items()
        ]
        out.append(_table(["Spread", "Contraste", "α anual", "t"], rows) + "\n")
    out.append("## Sub-periodos (α neto FF5+UMD)\n")
    rows = [
        [k, lab, _pct(v["alpha_net_annual"]), _num(v["alpha_t"]), _pct(v["cagr_net"])]
        for k, d in rob["subperiods"].items()
        for lab, v in d.items()
    ]
    out.append(_table(["Cartera", "Periodo", "α neto", "t", "CAGR neto"], rows) + "\n")
    out.append("## Capacidad\n")
    for k, d in res["capacity"].items():
        rows = [
            [
                f"{int(a):,}",
                _pct(v["alpha_net_annual"]),
                _num(v["alpha_net_t"]),
                _pct(v["avg_cash"]),
                _pct(v["cost_annual"]),
            ]
            for a, v in d["curve"].items()
        ]
        out.append(f"**{k}** — AUM máx. con α neto > 0: {d['max_aum_positive_alpha']:,}\n")
        out.append(_table(["AUM USD", "α neto", "t", "Caja media", "Coste anual"], rows) + "\n")
    out.append("## Acotamiento de supervivencia (posiciones sin precio = IWM + δ)\n")
    rows = []
    for k, d in res["unpriced_bounding"].items():
        rows.append(
            [
                k,
                _pct(d["avg_unpriced_share"]),
                _pct(d["delta_0.0"]["alpha_net_annual"]),
                _pct(d["delta_-0.01"]["alpha_net_annual"]),
                _pct(d["delta_-0.03"]["alpha_net_annual"]),
                _pct(d["breakeven_delta_monthly"], 2),
            ]
        )
    out.append(
        _table(
            ["Cartera", "% sin precio", "α (δ=0)", "α (δ=−1%/m)", "α (δ=−3%/m)", "δ breakeven/mes"],
            rows,
        )
        + "\n"
    )
    es = res["event_study"]
    for s in splits:
        out.append(f"## Event study — {s} (media de medias mensuales, t Newey-West)\n")
        rows = []
        for g, d in es[s].items():
            row = [g, _num(d.get("fbhar_6", {}).get("n_events"), 0)]
            for col in (
                "bhr_6",
                "mar_1",
                "mar_3",
                "mar_6",
                "mar_12",
                "fbhar_6",
                "fbhar_12",
                "car_6",
            ):
                c = d.get(col, {})
                row.append(f"{_pct(c.get('mean'), 2)} ({_num(c.get('t'), 1)})")
            rows.append(row)
        out.append(
            _table(
                [
                    "Grupo",
                    "N",
                    "BHR 6m",
                    "MAR 1m",
                    "MAR 3m",
                    "MAR 6m",
                    "MAR 12m",
                    "BHAR-FF 6m",
                    "BHAR-FF 12m",
                    "CAR arit. 6m (sesgado)",
                ],
                rows,
            )
            + "\n"
        )
        rows = []
        for g, d in es[s].items():
            rows.append(
                [g]
                + [
                    f"{_pct(d.get(f'lag_{n}', {}).get('mean'), 2)} "
                    f"({_num(d.get(f'lag_{n}', {}).get('t'), 1)})"
                    for n in ("open_t1", "close_t1", "open_t2", "open_t5")
                ]
            )
        out.append("Sensibilidad al rezago de entrada (MAR 6m):\n")
        out.append(
            _table(["Grupo", "t+1 apertura", "t+1 cierre", "t+2 apertura", "t+5 apertura"], rows)
            + "\n"
        )
    out.append("## Regresión de panel (BHAR-FF 6m, EF de mes, cluster emisor×mes)\n")
    for name, coefs in es["panel_regression"].items():
        rows = [
            [k, _num(v["coef"] * 100, 2) + " pp", _num(v["t"]), _num(v["p_two_sided"], 3)]
            for k, v in coefs.items()
            if k != "const"
        ]
        n = next(iter(coefs.values()))["n_obs"]
        out.append(f"**{name}** (N = {n:,})\n")
        out.append(_table(["Variable", "Coef.", "t", "p"], rows) + "\n")
    out.append("## Decadencia temporal\n")
    for g, d in es["decay"].items():
        out.append(
            f"- {g}: tendencia {_pct(d['trend_per_year'], 2)} por año (t = {_num(d['trend_t'])})"
        )
    if "ml_check" in res:
        ml = res["ml_check"]
        out.append("\n## Chequeo ML (CPCV purgado)\n")
        out.append(
            f"- IC Spearman lineal: {_num(ml['ic_linear_mean'], 4)}; GBM: {_num(ml['ic_gbm_mean'], 4)}; "
            f"diferencia t = {_num(ml['ic_gbm_minus_linear_t'])}; folds GBM mejor: "
            f"{_pct(ml['share_folds_gbm_better'], 0)} de {ml['n_folds']}"
        )
        out.append(
            f"- Corr(predicción, log mcap) = {_num(ml['corr_pred_log_mcap'])}; "
            f"corr(pred, log Amihud) = {_num(ml['corr_pred_log_amihud'])}"
        )
        imp = sorted(ml["permutation_importance"].items(), key=lambda kv: -kv[1])
        out.append("- Importancia por permutación: " + ", ".join(f"{k} {v:.4f}" for k, v in imp))
    if "multiple_testing" in res:
        mt = res["multiple_testing"]
        out.append("\n## Multiple testing\n")
        out.append(
            f"- Trials registrados: {mt['n_trials']}; var(SR mensual) entre trials: {mt['var_sr_monthly']:.5f}"
        )
        for k in ("CFO", "ALL", "CFO_MIDLARGE"):
            d = mt[k]
            out.append(
                f"- {k}: DSR (neutral a factores) = {_num(d['dsr_factor_neutral'], 3)}, "
                f"DSR (exceso bruto) = {_num(d['dsr_raw_excess'], 3)}, SR0 = {_num(d['sr0_monthly'], 3)}"
            )
        out.append(
            f"- PBO (CSCV, {mt['pbo_n_configs']} configs, {mt['pbo_months']} meses): {_num(mt['pbo'], 3)}"
        )

    # Figures
    if {"CFO", "ALL"} <= set(net.columns):
        panel = pd.read_parquet(
            Path("data/processed") / mode / "panel.parquet", columns=["rebal", "ticker", "ret_nodl"]
        )
        iwm = panel.loc[panel["ticker"].eq("IWM")].set_index("rebal")["ret_nodl"]
        iwm.index = pd.DatetimeIndex(iwm.index).to_period("M").to_timestamp("M")
        net = net.assign(IWM=iwm.reindex(net.index))
        cols = ["CFO", "ALL", "CFO_MIDLARGE"]
        bench = "IWM"
        lo, hi = ("2006-07-31", "2019-12-31") if mode == "dev" else ("2020-01-31", "2025-12-31")
        figures.cumulative(
            net.loc[lo:hi],
            cols,
            bench,
            "Carteras de compras de insiders (neto)",
            figs / "cumulative.png",
        )
    figures.capacity(res["capacity"], "Alpha neto vs. capital gestionado", figs / "capacity.png")
    figures.decay(
        {g: d["yearly_car6"] for g, d in es["decay"].items()},
        "BHAR-FF 6m medio por año de filing",
        figs / "decay.png",
    )
    path = base / "tables.md"
    path.write_text("\n".join(out) + "\n")
    return path
