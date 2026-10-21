# CFO Insider Trading Study

Estudio cuantitativo para evaluar si imitar las **compras discrecionales en mercado
abierto** de insiders corporativos (con foco en **CFOs**), reportadas a la SEC en
Forms 3/4/5, genera alpha **neto de costes** y **residual a factores** (FF5 + momentum).

> Un edge falso es peor que no encontrar edge. Una conclusión negativa honesta es
> un entregable válido.

## Documentos clave

| Documento | Contenido |
|---|---|
| [`docs/HYPOTHESES.md`](docs/HYPOTHESES.md) | Pre-registro congelado: hipótesis H1–H7, definiciones, particiones, regla de decisión |
| [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md) | Bitácora cronológica: decisiones, resultados y veredictos (incluidos los fallidos) |
| [`configs/base.yaml`](configs/base.yaml) | Configuración base que refleja el pre-registro |
| `trials/registry.jsonl` | Registro append-only de cada configuración evaluada (base del Deflated Sharpe/PBO) |

## Estado por fases

| Fase | Descripción | Estado |
|---|---|---|
| F0 | Pre-registro de hipótesis | ✅ |
| F1 | Esqueleto de ingeniería (config, interfaces, guards, CI) | ✅ |
| F2 | Ingesta y limpieza SEC (Forms 3/4/5) | ⏳ |
| F3 | Precios, factores, market cap point-in-time | ⏳ |
| F4 | Tests de fuga temporal sobre datos reales | ⏳ |
| F5 | Event study (CAR, alpha factorial, efecto CFO marginal) | ⏳ |
| F6 | Cartera, costes y capacidad | ⏳ |
| F7 | Validación (walk-forward, CPCV, DSR, PBO, robustez) | ⏳ |
| F8 | Holdout único y reporte final | ⏳ |

## Instalación reproducible

Requiere [uv](https://docs.astral.sh/uv/). Python y dependencias quedan fijados por
`.python-version` y `uv.lock`.

```bash
uv sync                      # crea .venv con versiones exactas del lockfile
uv run pre-commit install    # hooks: ruff, mypy, tests de fuga
uv run pytest                # suite de tests
uv run insider-study validate-config configs/base.yaml   # imprime el config hash
uv run insider-study status configs/base.yaml            # trials usados / holdout
```

Antes de descargar datos de la SEC, define tu identificación (política de acceso de
la SEC) en un overlay de configuración: `data.sec_user_agent: "Nombre email"`.

## Arquitectura

```
EDGARIngestor → Form4Parser → InsiderNormalizer → TransactionFilter → SignalBuilder
   → PortfolioConstructor → RiskManager → Backtester → Evaluator → Reporter
```

Las fuentes externas (precios, factores, acciones en circulación) están detrás de
adaptadores (`PriceProvider`, `FactorProvider`, `SharesOutstandingProvider`), de modo
que yfinance puede sustituirse por CRSP/Norgate sin tocar el resto.

| Módulo | Responsabilidad |
|---|---|
| `config.py` | Config pydantic inmutable, `extra="forbid"`, hash canónico SHA-256 |
| `domain.py` | Enums (códigos SEC, roles, buckets) y contratos de columnas entre etapas |
| `interfaces.py` | Protocolos de todos los componentes y adaptadores |
| `temporal.py` | Primitivas point-in-time: entrada estrictamente posterior al filing |
| `holdout.py` | `HoldoutGuard`: bloquea datos ≥ 2020 hasta el desbloqueo único |
| `trials.py` | Registro append-only de trials con presupuesto de multiple testing |
| `reproducibility.py` | Seeds, fingerprint del entorno, manifiesto de hashes de datos |
| `tracking.py` | MLflow (config hash + fingerprint en cada run) |

## Salvaguardas de integridad

1. **Filing date, nunca transaction date.** La entrada es la apertura de la primera
   sesión NYSE estrictamente posterior al filing. `entry_lag_sessions=0` es inválido
   en la config; `assert_no_lookahead` falla si una decisión es ≤ filing date; un
   test estático prohíbe `transaction_date` en los módulos de decisión.
2. **Holdout sellado por código.** Solo se desbloquea para un config hash congelado
   y una única vez; el registro se commitea.
3. **Multiple testing honesto.** Cada evaluación se registra; el DSR usa el conteo real.
4. **Datos versionados.** Los ficheros crudos no se commitean; su hash queda en un
   manifiesto que se verifica antes de cada run.

## Limitaciones conocidas (declaradas de antemano)

- **yfinance tiene sesgo de supervivencia** (cobertura pobre de empresas deslistadas).
  Se cuantifica y se trata con supuestos de retorno de deslistado (−30%, 0%, −100%).
- Market cap point-in-time 2004–2008 sin fuente estructurada gratuita (riesgo abierto).
