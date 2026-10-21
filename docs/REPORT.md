# Reporte final — ¿Copiar las compras de los CFOs genera alpha?

**Fecha:** 2026-10-21 · **Config final:** `dfb89f0b…` (ver `configs/base.yaml`) ·
**Pre-registro:** [`HYPOTHESES.md`](HYPOTHESES.md) (v1.0 + enmiendas 1–4) ·
**Bitácora:** [`RESEARCH_LOG.md`](RESEARCH_LOG.md)

---

## 1. Veredicto

> **No hay edge explotable.** Copiar las compras en mercado abierto de los CFOs no
> genera alpha neto de costes y de factores (FF5 + momentum) que sea
> estadísticamente distinguible de cero, ni en validation (2015–2019) ni en el
> holdout (2020–2025). El rol CFO tampoco aporta señal significativa sobre la de
> un insider cualquiera. **No se propone cartera.**

| Criterio pre-registrado (§9) | Resultado | ¿Pasa? |
|---|---|---|
| 1. H1 significativa tras Holm (validation) | α neto −0,9%/año, t = −0,55, p Holm = 1,00 | ❌ |
| 2. Deflated Sharpe ≥ 0,95 | 0,056 (N = 187 trials) | ❌ |
| 3. PBO < 0,5 | 0,43 | ✅ (por poco) |
| 4. CFO bate a "todos los insiders" neto | −1,1%/año, t = −1,02 | ❌ |
| 5. Holdout: mismo signo y p < 0,10 | α neto −3,4%/año, t = −1,01 | ❌ |
| 6. α neto > 0 con AUM = USD 1M | −0,9% (validation), −3,4% (holdout) | ❌ |

Lo que sí existe, y explica por qué la idea es tentadora:
- Hay un **retorno anormal bruto** tras las compras de insiders (BHAR factorial a 6
  meses de +3,7% en 2006–2014; CFOs +3,8%).
- Ese retorno es **decreciente en el tiempo**, se concentra en **micro/small
  caps** ilíquidas y en las **primeras semanas** tras el filing.
- Los **costes de ejecución (~4–5,5% anual)** lo consumen entero.
- Lo que en bruto parece batir al Russell 2000 es, en gran parte, exposición al
  factor tamaño (β SMB ≈ 0,6–1,0).

## 2. Qué se probó

- **Universo:** acciones US de todos los tamaños.
- **Señal:** compras discrecionales (código P) de Forms 4 y 4/A de la SEC, filings
  2006–2025.
  - 309.579 eventos (insider × emisor × filing), de los que 18.438 son de CFOs.
  - Ventas, Form 5, grants, ejercicios, gifts y planes 10b5-1 van segregados.
- **Entrada:**
  - Event study: apertura de la primera sesión NYSE **estrictamente posterior a la
    fecha de filing**.
  - Carteras: primera sesión del mes siguiente al filing (rebalanceo mensual).
- **Cartera:** long-only, equal-weight, máximo 5% por nombre, horizonte primario de
  6 meses (también 1, 3 y 12).
  - Límite de liquidez: 10% del ADV de 20 días.
  - Filtro de invertibilidad: precio ≥ USD 1 y ADV ≥ USD 10k.
- **Costes:**
  - Medio spread Corwin-Schultz (entre 5 y 500 pb).
  - Impacto `σ·√(Q/ADV)`.
  - Comisión de USD 0,005 por acción (mínimo USD 1).
- **Particiones:**
  - Train 2006–2014.
  - Validation 2015–2019.
  - Holdout 2020–2025, sellado por código hasta el final.
- **Multiple testing:**
  - 187 trials registrados en `trials/registry.jsonl`.
  - Holm sobre los 8 contrastes confirmatorios.
  - DSR y PBO (CSCV, S = 10).

## 3. Resultados por hipótesis

α = alpha FF5 + UMD anual, **neto de costes**, errores Newey-West. Los spreads son
la diferencia entre dos carteras long-only. Entre paréntesis, el t-estadístico.

| Hipótesis | Train 2006–14 | Validation 2015–19 | Holdout 2020–25 | Veredicto |
|---|---|---|---|---|
| **H1** CFO α > 0 | +3,8% (1,90) | −0,9% (−0,55) | −3,4% (−1,01) | **RECHAZADA** |
| **H2a** CFO − no-CFO | +1,8% (1,21) | −1,0% (−0,97) | +1,4% (0,79) | **RECHAZADA** |
| **H2b** CFO − CEO | +3,4% (2,20) | +1,3% (0,82) | +3,2% (1,82) | **NO CONCLUYENTE** (1) |
| **H2b** CFO − director | +2,2% (1,30) | −2,1% (−1,87) | +0,6% (0,33) | **RECHAZADA** |
| **H3** oportunista − rutinario | −0,6% (−0,38) | +0,2% (0,11) | +3,0% (1,31) | **RECHAZADA** |
| **H4** convicción alta − baja | −0,5% (−0,37) | +0,2% (0,18) | −1,9% (−1,48) | **RECHAZADA** |
| **H5** cluster − aislada | −0,6% (−0,68) | +0,8% (1,09) | −0,1% (−0,04) | **RECHAZADA** |
| **H6** CFO mid + large α > 0 | −0,8% (−0,32) | −4,1% (−1,99) | −1,7% (−0,47) | **RECHAZADA** |
| **H7** decadencia | tendencia del BHAR 6m: −0,3 pp/año (t −1,5, todos); −0,3 pp/año (t −1,0, CFO) | | | **DESCRIPTIVA**: compatible con arbitraje |

(1) CFO − CEO es positivo en los tres periodos, pero no sobrevive a Holm (p Holm
= 0,12 / 1,00 / 0,30). Además lo explica que las compras de **CEOs** rinden mal,
no que las de CFOs rindan bien.

p de Holm en validation: 1,00 para las ocho hipótesis.

**Efecto CFO marginal (regresión de panel):**
- Variable dependiente: BHAR factorial a 6 meses por evento.
- Controles: tamaño, momentum, Amihud, valor de la compra y retraso de filing.
- Efectos fijos de mes y clusters emisor × mes.

| Periodo | Coef. de `is_cfo` | t |
|---|---|---|
| Train | +0,8 pp | 0,80 |
| Validation | +2,4 pp | 1,53 |
| Holdout | +2,1 pp | 1,38 |

Siempre positivo, nunca significativo. Añadir B/M (2009+) no cambia nada.

## 4. Intentos de refutación

**¿Es size/value disfrazado?** En gran parte, sí.
- La cartera CFO crece más que IWM en bruto ([figura](../results/dev/figures/cumulative.png)),
  pero carga β SMB ≈ 0,6–1,0.
- Al controlar por FF5 + UMD, el alpha bruto cae a +8,4% / +3,4% / +2,0% en los
  tres periodos, y tras costes queda ≤ 0 fuera de train.
- En mid + large caps no hay alpha ni siquiera bruto.

**¿Sobrevive a costes en small caps?** No.

| Variante (validation, CFO) | α neto | t |
|---|---|---|
| Base | −0,9% | −0,55 |
| Costes × 0,5 | +1,2% | 0,77 |
| Costes × 2 | −4,9% | −3,03 |

- No hay ningún AUM entre USD 100k y 50M con α neto significativamente positivo
  ([curva](../results/dev/figures/capacity.png)).
- Con AUM bajo pesa la comisión mínima; con AUM alto, el límite de liquidez deja
  hasta un 45% en caja.

**¿Importa el rezago de filing?**
- La señal bruta está adelantada: un tercio o más del MAR a 6 meses llega el
  primer mes.
- Esperar 1–4 sesiones extra quita ~0,3–0,9 pp en todas las particiones.
- En carteras mensuales, usar la señal un mes más tarde no cambia el veredicto.

| MAR 6m CFO (entrada en…) | t+1 apertura | t+1 cierre | t+2 apertura | t+5 apertura |
|---|---|---|---|---|
| Train | 2,84% (1,9) | 2,29% (1,6) | 2,36% (1,6) | 2,08% (1,5) |
| Validation | 1,88% (1,0) | 1,56% (0,9) | 1,17% (0,6) | 1,37% (0,8) |
| Holdout | 2,13% (1,1) | 1,50% (0,8) | 1,26% (0,7) | 1,29% (0,7) |

**¿Aguanta en sub-periodos y crisis?**

| Cartera CFO | α neto | t |
|---|---|---|
| Crisis 2008–09 | +6,1% | 1,78 |
| 2010–14 | +2,1% | 1,24 |
| 2015–19 | −0,9% | −0,55 |
| COVID 2020 | −16,0% | −1,48 |
| 2021–25 | −3,0% | −0,78 |

El patrón es de decadencia: el edge bruto que documenta la literatura
pre-2010 ya no paga los costes.

**¿Y las definiciones alternativas?**
- CFO amplio (con CAO), excluir roles duales, CMP a 2 años o solo con compras,
  convicción por USD, cluster ≥ 2 o ventana de 60 días: ninguna cambia ningún
  veredicto (tablas en `results/*/tables.md`).
- Las compras 10b5-1 (diagnóstico) tienen alpha ≈ 0, como cabía esperar.

**El único candidato "prometedor", refutado.**
- CFO oportunista (CMP), contraste secundario no confirmatorio: α neto
  +2,0% (t 1,0) en train, +5,5% (t 2,25) en validation, +6,3% (t 1,11) en holdout.
- Pero tiene un DSR de 0,81 (< 0,95) con 187 trials.
- Solo tiene ~7–19 posiciones, con mucha caja por el límite del 5%.
- No supera el filtro de multiple testing. Queda como **hipótesis para un estudio
  nuevo con datos sin sesgo de supervivencia**, no como estrategia.

**¿Añade algo el ML?** No.
- Con CPCV purgado (45 folds, embargo de 6 meses), el IC de Spearman fuera de
  muestra es **negativo** tanto en el modelo lineal (−0,04) como en el GBM (−0,02).
- Las features más importantes son momentum, tamaño e iliquidez: el modelo
  reaprende factores conocidos, no información del insider (`is_cfo` tiene
  importancia ≈ 0).

## 5. Tratamiento de los datos SEC

| Paso | Detalle |
|---|---|
| Fuente | SEC *Insider Transactions Data Sets*, 80 trimestres 2006Q1–2025Q4, hashes en `data/manifest_sec.json` |
| Fechas | Señal fechada por **filing date**. Tests de fuga sintéticos, sobre datos reales (`tests/test_data_integrity.py`) y estáticos (prohibido usar `transaction_date` en módulos de decisión) |
| Amendments | 4/A con original localizado → se descarta (52.843 líneas); 4/A sin original → se usa con su propia fecha (16.021) |
| Duplicados | La misma compra reportada por entidades relacionadas (fondo, GP, gestor): 177.145 líneas eliminadas; se conserva la del officer/director |
| 10b5-1 | Casilla desde 2023 + regex en notas al pie y remarks; 18.346 líneas segregadas |
| Roles | Regex sobre el título en texto libre. Auditoría de 300 títulos aleatorios: CFO con 100% de precisión y recall |
| Rutinario / oportunista | CMP (2012), con clasificación point-in-time el 1 de enero de cada año usando solo filings anteriores; disponible desde 2009 |
| Tamaño | Breakpoints NYSE de French del mes previo × acciones en circulación XBRL (2009+). Antes de 2009, proxy por ADV con un 83% de acierto de bucket fuera de la muestra de calibración |

## 6. Riesgos y limitaciones (importantes)

1. **Sesgo de supervivencia severo y estructural.**
   - Yahoo no tiene **ninguna** empresa deslistada: las 5.739 series con precio
     llegan a 2026.
   - Solo tiene precio el 30% de los eventos de 2006, el 50% de 2015 y el 92% de 2025.
   - Supera con creces el umbral pre-registrado del 15%.
   - Este sesgo infla los retornos medidos: solo se ven las empresas que
     sobrevivieron.
   - Como el resultado ya es nulo o negativo *con* ese sesgo a favor, la
     conclusión negativa es conservadora.
   - El análisis de acotamiento lo confirma: para anular el alpha, las posiciones
     sin precio tendrían que **batir** a IWM en +0,17% al mes (CFO, validation) y
     +1,35% al mes (holdout).
   - Un resultado *positivo* con estos datos no habría sido creíble. Uno negativo sí.
2. **Dos miradas al holdout.**
   - La primera reveló un fallo de datos: ticks erróneos de Yahoo en OTC de céntimos
     (+47.000% en un mes) que hundían una cartera un −108% en un mes.
   - Se corrigió con un filtro de invertibilidad y límites al coste (Enmienda 4,
     aplicada a todas las particiones).
   - Ambas miradas están registradas en `HOLDOUT_UNSEALED`, y los resultados de la
     primera se conservan en `results/final_v1_data_bug/`.
   - H1 en el holdout fue −3,7% (v1) y −3,4% (v2): el veredicto no dependía de la
     corrección.
3. **Correcciones post-hoc documentadas** (Enmienda 3):
   - CAR aritmético → BHAR factorial (el CAR estaba sesgado al alza en microcaps).
   - Se descartan betas degeneradas y se winsoriza al 1/99%.
   - Ninguna cambia las carteras ni las hipótesis.
4. **Otras limitaciones:**
   - Sin datos SEC estructurados antes de 2006 (Enmienda 1).
   - B/M solo desde 2009.
   - El modelo de costes es estimado, no son fills reales.
   - Con rebalanceo mensual, la entrada llega de media ~2 semanas tarde (se
     cuantifica con las variantes de rezago).
   - Alineación aproximada entre retornos open-to-open y factores de mes calendario.
5. **Potencia limitada en validation** (60 meses). Un alpha neto real de ~1–2% al
   año no se distinguiría de cero; el holdout (72 meses) tampoco lo muestra.

## 7. Recomendación

- **No asignar capital** a una estrategia de copiar compras de CFOs (ni de insiders
  en general) con ejecución mensual long-only en este universo.
- Si se quiere seguir investigando, lo único que justifica más esfuerzo es:
  1. Repetir el estudio con un panel **sin sesgo de supervivencia** (CRSP o
     Norgate), que se conecta cambiando el adaptador `PriceProvider`.
  2. Probar una ejecución **diaria** que capture el primer mes tras el filing, con
     costes reales de micro caps.
  3. Contrastar la sub-hipótesis "CFO oportunista" como hipótesis nueva y
     pre-registrada, con su propio holdout.

## 8. Reproducibilidad

- Entorno: `uv sync` (lockfile).
- Pasos: `insider-study build-data` → descarga de precios → `run-study --mode dev`
  → `unseal-holdout` → `run-study --mode final`. Detalle en el README.
- Seeds y config hash fijos. Fingerprint del entorno en cada run de MLflow.
- Tests: `uv run pytest` (≈107, incluidos los de fuga temporal).
- Tablas completas: `results/dev/tables.md` y `results/final/tables.md`.
- Series mensuales: `results/*/monthly_net_returns.parquet`.
