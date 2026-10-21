# Pre-registro de hipótesis — Estudio CFO Insider Trading

**Estado:** CONGELADO (v1.0)
**Fecha de registro:** 2026-10-21
**Regla:** este documento se escribe *antes* de descargar o mirar ningún dato.
Cualquier cambio posterior se añade en la sección *Enmiendas* con fecha, motivo
y commit, nunca editando el texto original. Las enmiendas posteriores a ver
resultados de validación se marcan como `POST-HOC` y sus hipótesis pasan a ser
exploratorias (no confirmatorias).

---

## 1. Pregunta de investigación

¿Imitar las compras discrecionales en mercado abierto de insiders corporativos —con
foco en CFOs— genera retorno ajustado por riesgo **neto de costes** y **residual a
factores estándar** (FF5 + momentum), en horizontes de 1 a 12 meses, invirtiendo
solo con información disponible en la **fecha de filing**?

Una respuesta negativa honesta es un resultado válido.

## 2. Configuración fijada en Fase 0 (aprobada por el usuario)

| Parámetro | Valor |
|---|---|
| Universo | Acciones US (NYSE, NASDAQ, AMEX), **todos los tamaños** |
| Periodo | 2004-01-01 a 2025-12-31 |
| Fuente insiders | SEC EDGAR, *Insider Transactions Data Sets* (Forms 3/4/5) |
| Fuente precios | yfinance (con sesgo de supervivencia documentado, ver §7) |
| Factores | Ken French Data Library: FF5 + UMD (diarios y mensuales), ME breakpoints NYSE |
| Horizontes | 1, 3, 6, 12 meses (**6 meses = horizonte primario**) |
| Rebalanceo cartera | Mensual |
| Dirección | Long-only (ventas solo como diagnóstico, no se operan) |
| AUM base | USD 1.000.000 (curva de capacidad USD 100k – 50M) |
| Benchmarks | IWM (mercado) + cartera "todos los insiders" (baseline a batir) |

## 3. Particiones temporales (fijas)

| Partición | Ventana de *filing date* | Uso |
|---|---|---|
| Train | 2004-01-01 → 2014-12-31 | Diseño, calibración, exploración |
| Validation | 2015-01-01 → 2019-12-31 | Selección de configuración |
| **Holdout (OOS)** | 2020-01-01 → 2025-12-31 | **Una sola evaluación final** |

- Los eventos se asignan a una partición por su **filing date**.
- **Purga:** un evento de train/validation cuya ventana de holding cruce el inicio
  de la partición siguiente se trunca en esa frontera para cualquier ajuste de
  parámetros (embargo = horizonte). Los precios ≥ 2020-01-01 están bloqueados por
  código (`HoldoutGuard`) hasta el desbloqueo final.
- **Desbloqueo del holdout:** solo para un único *config hash* congelado,
  registrado en `HOLDOUT_UNSEALED` y commiteado. Cambiar la configuración después
  invalida el desbloqueo (el guard lo detecta).
- Un evento solo cuenta en un horizonte *h* si su ventana completa es observable
  con los datos de precios disponibles (o termina por deslistado, ver §7).

## 4. Definición del evento (señal)

Un **evento elegible** es una transacción que cumple TODO lo siguiente:

1. Form 4 o 4/A (Form 5 se excluye de la señal: es anual y tardío por diseño).
2. Tabla *non-derivative*, `TRANS_CODE = "P"` (compra en mercado abierto o privada),
   `TRANS_ACQUIRED_DISP_CD = "A"`, precio > 0, acciones > 0.
3. No marcada como plan 10b5-1 (casilla oficial desde abril 2023; antes, inferida
   de notas al pie por regex documentada). Las 10b5-1 se segregan, no se borran.
4. Retraso `filing_date − transaction_date` ≤ 30 días naturales en el análisis
   primario (los filings muy tardíos son información rancia y suelen ser
   correcciones). Sensibilidad: sin límite.
5. Emisor con *ticker* mapeable a una serie de precios válida en la fecha del evento.

**Excluidos de la señal** (se cuentan y reportan por motivo): grants (A), ejercicios
(M, X, C), retenciones fiscales (F), gifts (G), herencias (W), ventas (S → dataset
aparte de diagnóstico), transacciones derivadas.

**Amendments (4/A):** la señal se fecha en el **filing original**. Si un 4/A corrige
cantidades/precio, se usan los valores corregidos solo a partir de la fecha del 4/A
(versión point-in-time). Si un 4/A aparece sin original localizable, se trata como
filing nuevo con su propia fecha.

**Momento de entrada (integridad temporal):**
- La señal es conocible al cierre del día de filing (EDGAR acepta filings hasta
  las 22:00 ET con fecha del día).
- **Entrada primaria:** precio de **apertura** de la primera sesión NYSE
  **estrictamente posterior** a la filing date.
- Prohibido usar `transaction_date` para cualquier decisión de cartera; solo se
  permite como variable descriptiva (retraso de filing) y para la clasificación
  rutinario/oportunista, que usa únicamente transacciones con filing anterior a la
  fecha de decisión.
- Sensibilidades de rezago: cierre de t+1, apertura de t+2, t+5.

**Unidad de agregación:**
- Event study: evento = (insider, emisor, filing). Varias líneas del mismo filing se
  agregan (suma de acciones y valor; precio medio ponderado).
- Cartera: emisor-mes (un emisor entra una vez aunque tenga varios eventos).

## 5. Clasificación de rol

El título en texto libre (`RPTOWNER_TITLE`) y las casillas de relación se mapean con
reglas versionadas y testeadas:

- **CFO:** "chief financial officer", "CFO", "principal financial officer" y
  variantes. *Treasurer*, *VP Finance* y *controller* **no** cuentan como CFO
  (van a `OTHER_OFFICER`). Sensibilidad: incluir *principal accounting officer*.
- **CEO:** "chief executive officer", "CEO", "president and CEO", etc.
- **DIRECTOR:** casilla `IS_DIRECTOR` sin cargo ejecutivo.
- **OTHER_OFFICER**, **TEN_PCT_OWNER** (solo propietario >10%, sin cargo), **OTHER**.
- Roles duales (p. ej. CEO+CFO) se marcan; análisis primario los incluye en
  ambos grupos; sensibilidad los excluye.
- La precisión del mapeo se audita sobre una muestra aleatoria de 300 títulos
  (seed fija) etiquetada manualmente; objetivo ≥ 97% en la clase CFO.

## 6. Hipótesis confirmatorias

Para todas: retornos **netos de costes**, alpha estimado con regresión de series
temporales sobre **FF5 + UMD**, errores **Newey-West** (lags = horizonte en meses + 1)
para carteras; en regresiones de panel a nivel evento, errores agrupados por emisor
y por mes de filing.

### H1 — Edge CFO (hipótesis principal)
- **H1:** la cartera equal-weight de compras elegibles de CFOs, horizonte 6m,
  rebalanceo mensual, tiene alpha FF5+UMD neto > 0.
- **H0:** alpha neto ≤ 0. Test unilateral.

### H2 — Efecto marginal del rol CFO
- **H2a:** el retorno anormal de compras CFO supera al de compras de insiders no-CFO.
  Test: (i) cartera *spread* CFO − no-CFO; (ii) regresión de panel del CAR a 6m con
  dummies de rol, controlando por log(market cap), book-to-market, momentum 12-1,
  iliquidez de Amihud, convicción y efectos fijos de mes.
- **H2b:** CFO > CEO y CFO > DIRECTOR (dos contrastes).

### H3 — Oportunista vs rutinario (Cohen-Malloy-Pomorski 2012)
- Clasificación point-in-time a nivel insider: requiere ≥1 transacción de mercado
  abierto (P o S) en cada uno de los 3 años naturales anteriores. **Rutinario** si
  operó en el mismo mes natural en cada uno de esos 3 años; **oportunista** en caso
  contrario; **sin clasificar** si no hay 3 años de historia.
- **H3:** alpha(oportunistas) − alpha(rutinarios) > 0, en todos los insiders y en el
  subgrupo CFO.
- Sensibilidad: solo compras para el historial; ventana de 2 años.

### H4 — Convicción
- Medida primaria: `ΔH = acciones compradas / acciones poseídas antes de la compra`.
  Secundaria: valor en USD de la compra. (Compra vs. salario requiere proxies
  DEF 14A: **exploratorio**, fuera del bloque confirmatorio.)
- Terciles con *breakpoints* calculados en ventana expansiva solo con eventos
  pasados (sin look-ahead).
- **H4:** alpha(tercil alto) − alpha(tercil bajo) > 0.

### H5 — Clustering
- **Cluster:** ≥ 3 insiders distintos con compras elegibles en el mismo emisor en
  una ventana de 30 días naturales de *filing date*; la señal se fecha en el filing
  que completa el cluster.
- **H5:** alpha(compras en cluster) − alpha(compras aisladas) > 0.
- Sensibilidad: ≥ 2 insiders; ventana de 60 días.

### H6 — Tamaño (añadido por el usuario en Fase 0)
- Buckets point-in-time según *breakpoints* NYSE de Ken French:
  micro (< p20), small (p20–p50), mid (p50–p80), large (> p80).
- **H6:** el alpha neto de H1 es > 0 en el segmento **mid+large** (no solo en
  small/micro). Se reporta además cada bucket por separado.
- Se evalúan dos carteras: universo completo y solo mid+large.

### H7 — Decadencia de la señal
- **H7:** el alpha de H1 tiene tendencia negativa en el tiempo (regresión del
  alpha anual sobre año; y comparación 2004–2009 vs 2010–2014 vs 2015–2019).
  Es descriptiva: no condiciona el veredicto, pero se reporta siempre.

## 7. Supuestos de datos y sesgos conocidos (pre-declarados)

- **Supervivencia (yfinance):** yfinance no cubre bien empresas deslistadas.
  - Se reporta el % de eventos sin precio, por año y por bucket de tamaño.
  - Serie que termina antes del fin del horizonte → **retorno de deslistado −30%**
    (convención de Shumway 1997) en el análisis primario; sensibilidades 0% y −100%.
  - Eventos sin ninguna serie de precios: se excluyen y se cuentan. Si superan el
    15% de eventos en cualquier partición, el veredicto se marca como
    **no concluyente por calidad de datos** salvo que las sensibilidades extremas
    coincidan en signo.
- **Reutilización de tickers:** mapeo por CIK con validación cruzada del precio
  reportado en el Form 4 vs precio de mercado del día (desviación > 25% → descarte
  y conteo).
- **Market cap point-in-time:** acciones en circulación de XBRL (SEC, 2009+).
  **Riesgo abierto 2004–2008:** sin fuente estructurada gratuita; la solución se
  decide y documenta en Fase 3 antes de mirar retornos.
- **Splits/dividendos:** precios ajustados para retornos; precios sin ajustar para
  validación contra el Form 4 y para límites de liquidez.

## 8. Modelo de costes (pre-registrado)

- Comisión: USD 0,005/acción, mínimo USD 1 por orden.
- Medio spread: estimador de Corwin-Schultz (high-low) point-in-time, con suelo de
  5 pb y techo de 500 pb.
- Impacto: modelo raíz cuadrada `k · σ_diaria · sqrt(Q / ADV_20d)`, k = 1.
- Participación máxima: 10% del ADV de 20 días por día; el exceso no se ejecuta.
- Sensibilidad: costes × 2 y × 0,5.
- Coste de cada bucket de tamaño se reporta por separado.

## 9. Regla de decisión (veredicto)

Se declara **edge CFO existente** solo si se cumplen TODAS:

1. **H1** significativa al 5% tras corrección **Holm-Bonferroni** sobre la familia
   confirmatoria {H1, H2a, H2b-CEO, H2b-DIR, H3, H4, H5, H6} (8 contrastes), en
   validation.
2. **Deflated Sharpe Ratio ≥ 0,95**, con N = nº total de configuraciones registradas
   en `trials/registry.jsonl` (no un número elegido a mano).
3. **PBO < 0,5** (CSCV sobre las configuraciones probadas).
4. La cartera CFO bate a la cartera "todos los insiders" neta de costes
   (H2a en sentido de cartera).
5. **Holdout:** alpha neto con el mismo signo y p unilateral < 0,10, con magnitud
   ≥ 50% de la estimada en validation.
6. Alpha neto > 0 con AUM = USD 1M en la curva de capacidad.

Si se cumplen 1–3 pero falla 4 → el edge es de "insider genérico", no de CFO.
Si el alpha bruto es significativo pero el neto no → "edge no explotable".
Si el alpha desaparece al añadir factores → "es size/value disfrazado".

## 10. Presupuesto de multiple testing

- **Máximo pre-registrado:** 200 configuraciones evaluadas en train+validation.
- Toda evaluación (incluidas sensibilidades y pruebas fallidas) se registra
  automáticamente en `trials/registry.jsonl` (append-only, versionado en git).
- Superar el presupuesto exige una enmienda justificada; el DSR siempre usa el
  conteo real.
- ML (si se usa): solo si bate al baseline lineal en CPCV purgado; cada
  configuración de hiperparámetros cuenta como un trial.

---

## Enmiendas

### Enmienda 1 — 2026-10-21 — PRE-RESULTADOS (disponibilidad de datos)
Los *Insider Transactions Data Sets* de la SEC empiezan en **2006Q1** (2003–2005
devuelven 404). Raspar los XML de 2004–2005 (~600k filings) no es viable.
- Train pasa a **2006-01-01 → 2014-12-31**. Validation y holdout no cambian.
- La clasificación CMP necesita 3 años de historial: solo es posible desde **2009**.
  H3 en train usa 2009–2014.

### Enmienda 2 — 2026-10-21 — PRE-RESULTADOS (aclaraciones operativas)
Escrita tras ver la cobertura de los datos y antes de calcular ningún retorno.
a) **Limpieza adicional:** solo acciones comunes (se excluyen preferentes, warrants,
   units, notas…); se eliminan duplicados multi-filer (misma compra —emisor, fecha,
   acciones, precio— reportada por varias entidades relacionadas; se conserva la
   línea del officer/director).
b) **Entrada en cartera:** con rebalanceo mensual, la entrada efectiva es la
   apertura de la primera sesión del mes siguiente al filing (siempre posterior al
   filing). La entrada "t+1 open" se usa en el event study, y el coste de esperar
   se cuantifica con las variantes de rezago (t+1 cierre, t+2, t+5) y con la
   variante "saltar un mes".
c) **Market cap 2006–2009 y faltantes:** proxy por OLS
   `log(mcap) ~ log(ADV20 USD) + log(precio)` calibrada con eventos de 2009-07 a
   2010-12 (train) y validada en 2011–2014 (se reporta el acierto por bucket).
d) **Book-to-market** solo existe con XBRL (2009+). La regresión de panel principal
   de H2a controla por tamaño, momentum 12-1, iliquidez de Amihud, log del valor de
   la compra y retraso de filing, con efectos fijos de mes. B/M entra como robustez
   en la submuestra 2009+.
e) **Eventos sin precio (supervivencia):** se mantiene la regla del 15%. Además,
   análisis de acotamiento: las posiciones sin precio rinden IWM + δ, con
   δ ∈ {0, −1%, −3%} mensual, y se calcula el δ de breakeven que anula el alpha.
f) **DSR:** se calcula sobre la serie mensual neutral a factores del candidato
   (alpha + residuo de FF5+UMD), en validation, con N y la varianza de Sharpe
   tomados del registro de trials. Se reporta también el DSR sobre el exceso de
   retorno bruto. **PBO:** CSCV (S = 10) sobre los excesos de retorno netos
   mensuales de todas las configuraciones registradas, 2006-07 a 2019-12.
g) H3, H4 y H5 confirmatorias se contrastan sobre **todos los insiders**; el
   subgrupo CFO se reporta como secundario.
h) **Ventanas de evaluación** (meses calendario de retorno):
   - train: 2006-07 a 2014-12 (6 meses de calentamiento); las carteras CMP desde 2009-02;
   - validation: 2015-01 a 2019-12;
   - holdout: 2020-01 a 2025-12.
i) Cada configuración de cartera se registra **una vez** en el registro de trials
   (con su Sharpe de validation).

### Enmienda 3 — 2026-10-21 — POST-HOC (tras ver los resultados de train/validation)
Marcada como POST-HOC. No cambia ninguna hipótesis, cartera ni umbral; corrige un
estadístico y documenta un hallazgo de datos.
a) **CAR → BHAR factorial.** El CAR del event study suma retornos diarios
   aritméticos y está sesgado al alza en acciones volátiles (arrastre de
   volatilidad; Barber & Lyon 1997). En los resultados, el CAR 6m de "todos los
   insiders" (~10%) triplicaba el retorno buy-and-hold ajustado por mercado (~3,5%).
   Se sustituye como estadístico principal del event study, de la regresión de panel
   y del objetivo ML por el **BHAR factorial**: retorno compuesto de la acción menos
   el retorno esperado compuesto (FF5 + UMD, mismas betas pre-evento). El CAR se
   sigue mostrando como referencia, etiquetado como sesgado. Las carteras (base de
   H1–H6) no usan CAR y no cambian. El chequeo ML se repite con el nuevo objetivo y
   sus dos modelos se suman otra vez al conteo de trials.
b) **Supervivencia total en Yahoo.** Las 5.739 series con precio llegan todas a
   2026: Yahoo no contiene ninguna empresa deslistada. La muestra con precio está
   condicionada a sobrevivir hasta hoy. Los supuestos de retorno de deslistado
   (−30%, 0%, −100%) nunca se activan, así que esas sensibilidades no informan
   nada. El sesgo es estructural y se trata en el veredicto (§9, regla del 15%).
c) Al calcular el BHAR factorial aparecieron eventos con betas pre-evento
   degeneradas (precios planos o penny stocks), que componían retornos esperados
   absurdos. Se descartan las betas con |β| > 5 (sin AR factorial para ese
   evento) y todos los retornos por evento se winsorizan al 1%/99% dentro de cada
   partición antes de promediar. Las carteras no se ven afectadas.

### Enmienda 4 — 2026-10-21 — POST-HOLDOUT (corrección de un fallo de datos)
**Esta enmienda se escribió después de haber mirado una vez el holdout.**
En el primer holdout (config `25963eaa…`), la cartera "todos los insiders" perdió
un 108% en junio de 2025, algo imposible en una cartera long-only. La causa son
datos erróneos de Yahoo en acciones OTC de céntimos: meses de +10.000% a +47.000%
y volatilidad diaria del 1.600%. Una posición de peso ~0,07% se multiplicaba ×100
por un tick falso, y el modelo de impacto cobraba al venderla más que su propio
nocional.

Corrección, aplicada por igual a train, validation y holdout:
- Filtro de invertibilidad en cada rebalanceo, medido antes de operar: precio sin
  ajustar ≥ USD 1 y ADV20 ≥ USD 10.000.
- La volatilidad que entra en el modelo de impacto se limita al 25% diario.
- El coste de una operación nunca supera su nocional.

Nuevo config hash `dfb89f0b…`.
- Los resultados previos se conservan: dev en `results/dev_v1_pre_screen/` y
  holdout en `results/final_v1_data_bug/`, y se reportan junto a los nuevos.
- El registro de trials conserva las 95 evaluaciones anteriores y suma las nuevas:
  N solo crece, así que el DSR es más conservador.
- El holdout se re-abre con `second_look`; el fichero `HOLDOUT_UNSEALED` mantiene
  el historial de las dos miradas.
- El veredicto de dev ya estaba fijado antes de la primera mirada: H1 falla en
  validation.
