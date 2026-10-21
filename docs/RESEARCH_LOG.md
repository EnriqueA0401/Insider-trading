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
