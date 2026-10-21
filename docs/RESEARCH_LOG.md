# Research Log

Registro cronológico de decisiones y resultados. Cada sub-hipótesis termina con un
veredicto explícito (CONFIRMADA / RECHAZADA / NO CONCLUYENTE), incluidas las fallidas.
Nunca se borran entradas.

---

## 2026-10-21 — F0: Configuración y pre-registro

**Decisiones de Fase 0 (aprobadas por el usuario):**
- Universo: acciones US de todos los tamaños, 2004–2025. El usuario pidió
  explícitamente no limitarse a small caps → se añade H6 (análisis por bucket de
  tamaño NYSE y cartera mid+large separada).
- Precios: yfinance, aceptando y documentando el sesgo de supervivencia.
- Horizontes 1/3/6/12m (6m primario), rebalanceo mensual, long-only.
- AUM base USD 1M; benchmarks IWM + cartera "todos los insiders".

**Comprobaciones de acceso (sin mirar contenido):** los datasets estructurados de la
SEC (`insider-transactions-data-sets`), los índices de EDGAR, Yahoo Finance y la
librería de Ken French responden desde el entorno.

**Pre-registro:** `docs/HYPOTHESES.md` v1.0 congelado antes de descargar datos.
Familia confirmatoria de 8 contrastes (Holm), presupuesto de 200 trials.

**Riesgos abiertos:**
1. Supervivencia en yfinance → umbral de 15% de eventos sin precio para declarar
   resultado no concluyente.
2. Market cap point-in-time 2004–2008 → decidir fuente/método en F3 antes de
   calcular retornos.
3. 10b5-1 antes de 2023 solo inferible por notas al pie → posible contaminación de
   la señal "discrecional" en train; sensibilidad prevista.

## 2026-10-21 — F1: Esqueleto de ingeniería

- Paquete `insider_study` tipado (mypy strict), config pydantic inmutable con hash.
- Protocolos para los 10 componentes del pipeline + adaptadores de datos.
- `temporal.py`: entrada = primera sesión NYSE estrictamente posterior al filing;
  tests con festivos, cierres extraordinarios (Sandy 2012) y property-based testing.
- Test explícito que demuestra que un builder ingenuo con transaction date es
  detectado como look-ahead.
- `HoldoutGuard`, `TrialRegistry`, manifiesto de datos, MLflow, logging JSON.
- CI (GitHub Actions) + pre-commit (ruff, mypy, pytest).

**Siguiente:** F2 — ingesta de los datasets trimestrales Form 3/4/5 de la SEC.

## 2026-10-21 — F2: Datos SEC

- **Hallazgo:** los datasets estructurados de la SEC empiezan en 2006Q1 → Enmienda 1
  (train 2006–2014; CMP desde 2009). Registrada antes de calcular retornos.
- Descargados 80 trimestres (2006Q1–2025Q4), hash en `data/manifest_sec.json`.
- Embudo de filtrado (líneas de transacción P/S, tras resolver amendments):

| Paso | Líneas |
|---|---|
| Líneas P/S de entrada | 3.441.089 |
| 4/A con original localizado (descartados; señal en fecha original) | 52.843 |
| 4/A huérfanos (se mantienen con su propia fecha) | 16.021 |
| Duplicados multi-filer (misma compra, entidades relacionadas) | 177.145 |
| Form 5 / no elegible | 17.528 |
| Precio o acciones ≤ 0 | 14.121 |
| Valores no comunes (preferentes, warrants…) | 23.461 |
| Ticker ausente | 10.739 |
| Filing > 30 días tras la operación | 40.904 |
| Planes 10b5-1 (segregados, no borrados) | 18.346 |
| **Líneas de compra elegibles** | **615.264** |
| **Eventos (insider × emisor × filing)** | **309.579** |
| Líneas de venta (dataset aparte, diagnóstico) | 2.461.912 |

- **Auditoría de roles** (300 títulos aleatorios, seed fija,
  `data/processed/role_audit_sample.csv`, revisión manual por el investigador):
  - CFO: 100% de precisión y de recall en la muestra (objetivo ≥ 97%).
  - Un solo error, en CEO ("Chief Exec Officer"), ya corregido y con test.
- Eventos por rol principal:

| Rol | Eventos |
|---|---|
| Director | 139.966 |
| Otro officer | 53.986 |
| CEO | 50.961 |
| Propietario >10% | 40.134 |
| **CFO** | **18.438** |

## 2026-10-21 — F3: Datos de mercado

- Factores FF5 + UMD (diarios y mensuales) y breakpoints ME de NYSE de la librería
  de Ken French, hasta 2026-08.
- Acciones en circulación y book equity desde los XBRL frames de la SEC (2009+).
- **Incidencia con yfinance:** en lotes, Yahoo devolvía tablas vacías *sin error*
  cuando limitaba (rate limit). Se habrían marcado como "sin datos" miles de tickers
  vivos, inflando artificialmente el sesgo de supervivencia. Se descartó toda esa
  caché y se reescribió la descarga contra el endpoint público `v8/chart`, ticker a
  ticker, distinguiendo 404 (no existe) de 429 (reintentar). La reconstrucción del
  precio sin ajustar está verificada (split 7:1 de AAPL en 2014).

## 2026-10-21 — F5–F7: Resultados de desarrollo (train + validation)

- Cobertura de precios: solo el 43% de los eventos 2006–2019 tiene serie en Yahoo, y
  **todas** las series llegan a 2026. La muestra con precio es 100% supervivientes
  (Enmienda 3b).
- Corrección post-hoc del estadístico del event study: CAR aritmético → BHAR
  factorial (Enmienda 3a/c).
- Veredictos en validation (α FF5+UMD neto anual, p de Holm):

| Hipótesis | α neto | p Holm | Veredicto |
|---|---|---|---|
| H1 | −0,9% | 1,00 | RECHAZADA |
| H2a | −1,0% | 1,00 | RECHAZADA |
| H2b (vs CEO) | +1,3% | 1,00 | RECHAZADA |
| H2b (vs director) | −2,1% | 1,00 | RECHAZADA |
| H3 | +0,2% | 1,00 | RECHAZADA |
| H4 | +0,2% | 1,00 | RECHAZADA |
| H5 | +0,8% | 1,00 | RECHAZADA |
| H6 | −4,1% | 1,00 | RECHAZADA |

- DSR (CFO) = 0,056 con N = 187; PBO = 0,43. ML: IC fuera de muestra negativo en
  los dos modelos → descartado.

## 2026-10-21 — F8: Holdout

- **Mirada 1** (config `25963eaa`): H1 = −3,7% (t −1,01). Reveló un fallo de datos
  (ticks erróneos de Yahoo en OTC de céntimos; −108% en un mes en "todos los
  insiders"). Resultados conservados en `results/final_v1_data_bug/`.
- **Enmienda 4** (post-holdout): filtro de invertibilidad y límites al coste,
  aplicados a todas las particiones. Dev se re-ejecutó y los trials se acumulan
  (N: 95 → 187).
- **Mirada 2** (config `dfb89f0b`): H1 = −3,4% (t −1,01). Ningún contraste
  confirmatorio sobrevive a Holm (el mínimo, CFO − CEO, tiene p Holm = 0,30).
- **Veredicto final: no hay edge explotable** (ver `docs/REPORT.md`).
