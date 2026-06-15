# Analisi traffico, aeroporto e inquinanti

Data di aggiornamento: `2026-05-17`

## 1. Scopo del lavoro

L'obiettivo della repo non e' soltanto ottenere modelli predittivi accurati, ma
capire **quale tipo di informazione** renda prevedibili gli inquinanti in un
contesto urbano influenzato da:

- traffico urbano da spire;
- attivita' aeroportuale BLQ;
- meteo;
- memoria temporale delle serie;
- contesto multi-stazione e multi-inquinante.

La distinzione tra forecasting e interpretazione e' centrale. Un modello puo'
prevedere bene un target solo perche' sfrutta la persistenza della serie. Questo
e' utile operativamente, ma non basta per sostenere una lettura fisica o
sorgente-specifica. Per questo la repo separa quattro piani analitici:

1. forecasting single-target con e senza autoregressione del target;
2. forecasting multi-target per verificare se esista informazione condivisa tra
   stazioni e inquinanti;
3. analisi `upwind/downwind` e gradienti spaziali per testare la coerenza fisica
   del segnale aeroportuale;
4. sintesi esplicita `cross_pollutant` per confrontare in modo ordinato famiglie
   chimiche e target.

## 2. Risultato chiave della run corrente

Il risultato piu' solido della run attuale non e' solo quale modello ottenga il
`R2` piu' alto, ma **quali blocchi informativi restino importanti quando vengono
rimossi sistematicamente**.

Nell'ablazione estesa single-target `xgboost`, i contributi medi piu' forti sono:

- `meteo`: `mean delta R2 = +0.087`
- `other_pollutants`: `+0.026`
- `other_pollutants_porta_san_felice`: `+0.021`
- `rolling_features`: `+0.017`

I blocchi aeroportuali piu' raffinati hanno invece contributi medi molto piu'
piccoli:

- `airport`: `+0.001`
- `station_wind_bools`: `-0.001`
- `airport_service_type`: `-0.002`
- `airport_wind_interaction`: `-0.003`

La lettura corretta non e' quindi "l'aeroporto non conta", ma una lettura piu'
precisa:

- il contributo aeroportuale esiste;
- non domina in media l'intera matrice target-orizzonte;
- emerge soprattutto in **casi selettivi** e non come driver uniforme;
- la struttura media del problema resta guidata soprattutto da meteo, contesto
  multi-inquinante e memoria recente aggregata.

## 3. Stato attuale della repository

La repo contiene quattro blocchi analitici distinti.

### 3.1. Parte `explain`

Script:

- `explain_pollutants_by_feature_groups.py`

Funzione:

- forecasting multi-orizzonte single-target;
- confronto tra modelli;
- confronto tra setup con e senza autoregressione;
- ablazioni mirate e ablazione estesa;
- SHAP per gruppi;
- importance native di `XGBoost`;
- confronto multioutput `XGBoost`.

Output principali:

- `Analysis/slurm_full_explain/advanced_temporal_cv_scores.csv`
- `Analysis/slurm_full_explain/advanced_temporal_cv_predictions.csv`
- `Analysis/slurm_full_explain/advanced_temporal_cv_summary.csv`
- `Analysis/slurm_full_explain/advanced_ablation_summary.csv`
- `Analysis/slurm_full_explain/advanced_extended_ablation_feature_sets.csv`
- `Analysis/slurm_full_explain/advanced_extended_ablation_scores.csv`
- `Analysis/slurm_full_explain/advanced_extended_ablation_summary.csv`
- `Analysis/slurm_full_explain/advanced_extended_ablation_fold_deltas.csv`
- `Analysis/slurm_full_explain/advanced_extended_ablation_delta_summary.csv`
- `Analysis/slurm_full_explain/advanced_group_shap.csv`
- `Analysis/slurm_full_explain/advanced_xgboost_native_feature_importances.csv`
- `Analysis/slurm_full_explain/advanced_xgboost_native_feature_importances_summary.csv`
- `Analysis/slurm_full_explain/advanced_multioutput_xgboost_scores.csv`
- `Analysis/slurm_full_explain/advanced_multioutput_xgboost_summary.csv`
- `Analysis/slurm_full_explain/advanced_multioutput_ablation_summary.csv`
- `Analysis/slurm_full_explain/advanced_multioutput_extended_ablation_feature_sets.csv`
- `Analysis/slurm_full_explain/advanced_multioutput_extended_ablation_scores.csv`
- `Analysis/slurm_full_explain/advanced_multioutput_extended_ablation_summary.csv`
- `Analysis/slurm_full_explain/advanced_multioutput_extended_ablation_fold_deltas.csv`
- `Analysis/slurm_full_explain/advanced_multioutput_extended_ablation_delta_summary.csv`
- `Analysis/slurm_full_explain/advanced_multioutput_group_shap.csv`
- `Analysis/slurm_full_explain/advanced_multioutput_xgboost_native_feature_importances.csv`
- `Analysis/slurm_full_explain/advanced_multioutput_xgboost_native_feature_importances_summary.csv`
- `Analysis/slurm_full_explain/advanced_runtime_profile.csv`
- `Analysis/slurm_full_explain/pollutant_station_reference_stats.csv`
- `Analysis/slurm_full_explain/plots/`

### 3.2. Parte `upwind/downwind`

Script:

- `upwind_downwind_analysis.py`

Funzione:

- classificazione `downwind`, `upwind`, `crosswind`, `calm`;
- confronti descrittivi tra regimi;
- regressioni con interazione `BLQ x downwind`;
- matching `downwind/upwind`;
- bootstrap a blocchi;
- sensibilita' alla soglia;
- gradienti spaziali e DID multi-stazione;
- SHAP per regime.

Output principali:

- `Analysis/slurm_full_upwind/upwind_downwind_summary.csv`
- `Analysis/slurm_full_upwind/upwind_downwind_blq_effects.csv`
- `Analysis/slurm_full_upwind/upwind_downwind_distributed_lag_effects.csv`
- `Analysis/slurm_full_upwind/upwind_downwind_regression_coefficients.csv`
- `Analysis/slurm_full_upwind/upwind_downwind_distributed_lag_coefficients.csv`
- `Analysis/slurm_full_upwind/upwind_downwind_matched_summary.csv`
- `Analysis/slurm_full_upwind/upwind_downwind_matched_pairs.csv`
- `Analysis/slurm_full_upwind/upwind_downwind_bootstrap_effects.csv`
- `Analysis/slurm_full_upwind/upwind_downwind_threshold_sensitivity.csv`
- `Analysis/slurm_full_upwind/upwind_downwind_classified_hours.csv`
- `Analysis/slurm_full_upwind/upwind_downwind_blq_quantile_summary.csv`
- `Analysis/slurm_full_upwind/upwind_downwind_group_shap_by_regime.csv`
- `Analysis/slurm_full_upwind/multistation_did_summary.csv`
- `Analysis/slurm_full_upwind/multistation_did_regression_coefficients.csv`
- `Analysis/slurm_full_upwind/multistation_panel_long.csv`
- `Analysis/slurm_full_upwind/multistation_spatial_gradients.csv`
- `Analysis/slurm_full_upwind/multistation_spatial_gradient_summary.csv`
- `Analysis/slurm_full_upwind/multistation_station_wind_features.csv`
- `Analysis/slurm_full_upwind/plots/`

### 3.3. Parte `airport_response`

Script:

- `airport_response_analysis.py`

Funzione:

- curve empiriche target vs BLQ per regime;
- profili di dipendenza parziale;
- finestre evento;
- probabilita' di superamento soglia;
- gradienti descrittivi multi-stazione.

Output principali:

- `Analysis/airport_response_full/blq_empirical_response_curves.csv`
- `Analysis/airport_response_full/blq_partial_dependence_model_metrics.csv`
- `Analysis/airport_response_full/blq_partial_dependence_profiles.csv`
- `Analysis/airport_response_full/blq_event_windows_summary.csv`
- `Analysis/airport_response_full/blq_event_windows_long.csv`
- `Analysis/airport_response_full/blq_exceedance_probabilities.csv`
- `Analysis/airport_response_full/blq_spatial_gradient_response.csv`
- `Analysis/airport_response_full/plots/`

Questa parte e' **descrittiva-esplicativa**, non causale. Serve a rendere piu'
leggibile la relazione tra BLQ, vento, target e contesto urbano.

### 3.4. Parte `cross_pollutant`

Script:

- `cross_pollutant_analysis.py`

Funzione:

- confronto esplicito tra target e famiglie chimiche;
- sintesi della prevedibilita' multi-orizzonte;
- sintesi dei gruppi ablativi dominanti;
- sintesi standardizzata dei contrasti di vento.

Output principali:

- `Analysis/cross_pollutant/cross_pollutant_predictability_summary.csv`
- `Analysis/cross_pollutant/cross_pollutant_predictability_target_overview.csv`
- `Analysis/cross_pollutant/cross_pollutant_predictability_pollutant_overview.csv`
- `Analysis/cross_pollutant/cross_pollutant_ablation_group_matrix.csv`
- `Analysis/cross_pollutant/cross_pollutant_ablation_top_groups.csv`
- `Analysis/cross_pollutant/cross_pollutant_targeted_ablation_summary.csv`
- `Analysis/cross_pollutant/cross_pollutant_group_shap_summary.csv`
- `Analysis/cross_pollutant/cross_pollutant_wind_response_summary.csv`
- `Analysis/cross_pollutant/cross_pollutant_wind_response_pollutant_overview.csv`
- `Analysis/cross_pollutant/cross_pollutant_overview.csv`
- `Analysis/cross_pollutant/cross_pollutant_family_overview.csv`
- `Analysis/cross_pollutant/cross_pollutant_runtime_profile.csv`
- `Analysis/cross_pollutant/plots/`

Questa quarta analisi non ricalcola i modelli di base. Riordina e sintetizza i
risultati gia' prodotti dagli altri blocchi.

## 4. Dataset usato

File:

- `Datasets_Raw/hourly_merged_2023_2025.csv`

Caratteristiche:

- `9.792` righe orarie
- `61` colonne
- intervallo: `2024-05-29 00:00:00` -> `2025-07-10 23:00:00`
- chiave temporale: `datetime`

Questo file rappresenta l'intersezione temporale comune tra tutti i blocchi dati
utilizzati. La scelta e' corretta: evita di addestrare modelli su periodi in cui
uno dei blocchi principali manca del tutto.

### 4.1. Target analizzati

I target della run corrente sono:

- `NO2_porta_san_felice`
- `CO_porta_san_felice`
- `C6H6_porta_san_felice`
- `NO2_giardini_margherita`
- `NO2_via_chiarini`
- `O3_giardini_margherita`
- `O3_via_chiarini`

Statistiche descrittive nel dataset unificato:

| target | unita' | minimo | medio | massimo |
| --- | --- | ---: | ---: | ---: |
| `NO2_porta_san_felice` | `ug/m3` | `2.000` | `26.586` | `96.000` |
| `CO_porta_san_felice` | `mg/m3` | `0.000` | `0.468` | `2.500` |
| `C6H6_porta_san_felice` | `ug/m3` | `0.100` | `0.961` | `6.100` |
| `NO2_giardini_margherita` | `ug/m3` | `0.000` | `13.928` | `63.000` |
| `NO2_via_chiarini` | `ug/m3` | `0.000` | `15.560` | `82.000` |
| `O3_giardini_margherita` | `ug/m3` | `0.000` | `50.426` | `188.000` |
| `O3_via_chiarini` | `ug/m3` | `0.000` | `45.810` | `213.000` |

Lettura utile:

- `NO2` e' disponibile su tre stazioni e quindi e' il candidato naturale per i
  confronti spaziali;
- `O3` e' disponibile sulle due stazioni esterne ed e' il target piu' adatto a
  leggere dinamiche meteorologiche e di fondo;
- `CO` e `C6H6` sono concentrati su Porta San Felice.

### 4.2. Blocchi informativi

Il dataset integra:

- traffico aeroportuale BLQ, incluso il dettaglio per `SERVICE_TYPE_CODE`;
- traffico urbano da spire, mantenute come colonne separate;
- meteo da due sorgenti, `_aero` e `_centro`;
- altri inquinanti come contesto multi-stazione e multi-inquinante.

La selezione delle spire mantiene `20` sensori unici:

- `5` piu' vicini a BLQ;
- `5` piu' vicini a `Porta San Felice`;
- `5` piu' vicini a `Giardini Margherita`;
- `5` piu' vicini a `Via Chiarini`.

### 4.3. Feature temporali e derivate

Il pipeline costruisce:

- feature calendario:
  - `hour`, `dayofweek`, `month`, `is_weekend`
  - `hour_sin`, `hour_cos`, `month_sin`, `month_cos`
- lag:
  - `_lag_1h`, `_lag_2h`, `_lag_3h`, `_lag_6h`, `_lag_12h`, `_lag_24h`
- differenze:
  - `_diff_1h`
- rolling mean:
  - `_rolling_3h_mean`, `_rolling_6h_mean`, `_rolling_12h_mean`, `_rolling_24h_mean`
- rolling std:
  - `_rolling_3h_std`, `_rolling_6h_std`, `_rolling_12h_std`, `_rolling_24h_std`
- interazioni con il vento rispetto alla geometria aeroporto -> stazione.

Questo e' tecnicamente importante perche' il problema non e' una regressione
tabellare statica: i target dipendono da ritardo, accumulo recente, variabilita'
locale e regime di trasporto.

## 5. Metodi e logica delle analisi

### 5.1. Forecasting single-target

Domanda:

- quanto bene si riesce a prevedere ciascun target a `1h`, `3h`, `6h`, `12h`,
  `24h`?

Schema:

- validazione temporale `expanding-window` a `5` fold;
- metriche: `R2`, `MAE`, `RMSE`, `MAPE`;
- modelli confrontati:
  - `ridge`
  - `decision_tree`
  - `random_forest`
  - `extra_trees`
  - `adaboost`
  - `xgbrf`
  - `xgboost`

Le due viste sono:

- `no_target_*`: il modello non usa il passato del target;
- `with_target_*`: il modello usa anche il passato del target.

La prima vista e' piu' interpretativa. La seconda e' piu' predittiva.

### 5.2. Forecasting multioutput

Domanda:

- il contesto condiviso tra stazioni e inquinanti aggiunge informazione utile?

Metodo:

- `MultiOutputRegressor(XGBoost)`

Lo scopo non e' sostituire il single-target, ma verificare se esista un guadagno
leggibile quando i target vengono previsti insieme.

### 5.3. Ablazioni

Domanda:

- quanto perde il modello quando si rimuove un blocco coerente di feature?

Livelli usati:

- ablazioni mirate su `service_type`, `station_wind_bools` e loro rimozione
  congiunta;
- ablazione estesa su gruppi piu' ampi come `meteo`, `urban_traffic`,
  `other_pollutants`, `rolling_features`, `airport`, `airport_service_type`,
  `wind_transport`.

L'ablazione risponde a una domanda diversa dalle SHAP:

- SHAP dice cosa il modello usa;
- l'ablazione dice cosa il modello perde davvero se un blocco manca.

### 5.4. SHAP e importance native

Uso:

- `advanced_group_shap.csv`
- `advanced_multioutput_group_shap.csv`
- `advanced_xgboost_native_feature_importances_summary.csv`
- `advanced_multioutput_xgboost_native_feature_importances_summary.csv`

Scopo:

- capire quali gruppi o feature puntuali guidano le predizioni;
- distinguere il peso di autoregressione, meteo, traffico, aeroporto e contesto
  multi-inquinante.

### 5.5. Analisi `upwind/downwind`

Domanda:

- il segnale associato a BLQ e' coerente con un'ipotesi fisica di trasporto
  verso le stazioni?

Metodo:

- classificazione delle ore in `downwind`, `upwind`, `crosswind`, `calm`;
- confronti descrittivi per regime;
- regressioni con termini:
  - `blq_activity`
  - `downwind_flag`
  - `upwind_flag`
  - `blq_x_downwind`
  - `blq_x_upwind`
- matching `downwind/upwind`;
- bootstrap a blocchi;
- sensibilita' alla soglia.

`downwind` significa che l'aria si muove da BLQ verso la stazione, non
genericamente "vento favorevole".

### 5.6. Gradienti spaziali

Domanda:

- il segnale di BLQ si vede meglio nei livelli assoluti o nei gradienti tra
  stazioni?

Metodo:

- costruzione di gradienti come:
  - `NO2_psf_minus_chiarini`
  - `NO2_psf_minus_giardini`
  - `O3_chiarini_minus_giardini`
- regressioni DID multi-stazione sui gradienti.

### 5.7. Analisi `airport_response`

Domanda:

- esistono pattern empirici leggibili tra BLQ, vento e target che siano piu'
  intuitivi dei soli coefficienti?

Questa parte include:

- curve target vs BLQ per regime;
- partial dependence;
- finestre evento;
- superamento soglie alte;
- gradienti descrittivi per classi di BLQ.

E' una parte **esplicativa**, non una dimostrazione causale.

### 5.8. Analisi `cross_pollutant`

Domanda:

- in cosa `NO2`, `CO`, `C6H6` e `O3` si somigliano davvero, e in cosa divergono?

Questa parte riaggrega gli output precedenti su tre assi:

- prevedibilita';
- dipendenza dai gruppi di feature;
- risposta ai regimi di vento.

## 6. Risultati predittivi: vista senza storico del target

Questa e' la sezione piu' importante se l'obiettivo e' capire **quanto segnale
esterno** esista davvero.

La tabella seguente riporta, per ogni target e orizzonte, il **miglior setup
single-target senza autoregressione**, quindi non tutti i 1960 risultati grezzi
ma il miglior compromesso modello + feature set per ciascun caso.

| target | h | model | feature set | R2 | MAE | RMSE | MAPE |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `C6H6_porta_san_felice` | `1` | `xgboost` | `no_target_without_service_type_or_station_wind_bools` | `0.578` | `0.231` | `0.320` | `25.51` |
| `C6H6_porta_san_felice` | `3` | `xgboost` | `no_target_without_service_type_or_station_wind_bools` | `0.401` | `0.270` | `0.379` | `30.48` |
| `C6H6_porta_san_felice` | `6` | `xgboost` | `no_target_without_station_wind_bools` | `0.305` | `0.299` | `0.409` | `34.73` |
| `C6H6_porta_san_felice` | `12` | `xgboost` | `no_target_without_station_wind_bools` | `0.299` | `0.303` | `0.411` | `36.43` |
| `C6H6_porta_san_felice` | `24` | `xgboost` | `no_target_without_service_type` | `0.179` | `0.331` | `0.445` | `38.85` |
| `CO_porta_san_felice` | `1` | `xgboost` | `no_target_autoregressive` | `0.417` | `0.111` | `0.142` | `23.32` |
| `CO_porta_san_felice` | `3` | `xgboost` | `no_target_without_service_type` | `0.272` | `0.124` | `0.162` | `25.67` |
| `CO_porta_san_felice` | `6` | `xgboost` | `no_target_without_service_type` | `0.199` | `0.132` | `0.171` | `27.38` |
| `CO_porta_san_felice` | `12` | `xgboost` | `no_target_autoregressive` | `0.147` | `0.141` | `0.180` | `29.43` |
| `CO_porta_san_felice` | `24` | `xgboost` | `no_target_without_service_type_or_station_wind_bools` | `0.087` | `0.143` | `0.184` | `30.15` |
| `NO2_giardini_margherita` | `1` | `xgboost` | `no_target_without_service_type_or_station_wind_bools` | `0.526` | `3.781` | `5.104` | `38.41` |
| `NO2_giardini_margherita` | `3` | `xgboost` | `no_target_without_service_type` | `0.239` | `4.940` | `6.592` | `53.61` |
| `NO2_giardini_margherita` | `6` | `xgboost` | `no_target_without_service_type_or_station_wind_bools` | `0.116` | `5.471` | `7.094` | `66.10` |
| `NO2_giardini_margherita` | `12` | `xgboost` | `no_target_without_service_type` | `0.023` | `5.681` | `7.314` | `73.28` |
| `NO2_giardini_margherita` | `24` | `extra_trees` | `no_target_without_service_type_or_station_wind_bools` | `-0.027` | `5.921` | `7.570` | `82.94` |
| `NO2_porta_san_felice` | `1` | `xgboost` | `no_target_without_service_type` | `0.206` | `8.519` | `10.841` | `34.61` |
| `NO2_porta_san_felice` | `3` | `xgboost` | `no_target_without_service_type` | `0.070` | `9.254` | `11.723` | `38.43` |
| `NO2_porta_san_felice` | `6` | `xgboost` | `no_target_without_station_wind_bools` | `-0.007` | `9.689` | `12.140` | `40.91` |
| `NO2_porta_san_felice` | `12` | `adaboost` | `no_target_autoregressive` | `-0.062` | `10.086` | `12.495` | `45.41` |
| `NO2_porta_san_felice` | `24` | `xgboost` | `no_target_autoregressive` | `-0.003` | `9.523` | `12.025` | `42.71` |
| `NO2_via_chiarini` | `1` | `xgboost` | `no_target_without_service_type_or_station_wind_bools` | `0.293` | `5.717` | `7.974` | `30.33` |
| `NO2_via_chiarini` | `3` | `xgboost` | `no_target_without_service_type_or_station_wind_bools` | `0.123` | `6.567` | `8.895` | `37.28` |
| `NO2_via_chiarini` | `6` | `xgboost` | `no_target_without_service_type` | `0.057` | `6.868` | `9.167` | `41.64` |
| `NO2_via_chiarini` | `12` | `xgboost` | `no_target_without_service_type` | `0.059` | `6.918` | `9.194` | `42.57` |
| `NO2_via_chiarini` | `24` | `xgboost` | `no_target_autoregressive` | `0.023` | `6.989` | `9.372` | `43.79` |
| `O3_giardini_margherita` | `1` | `xgboost` | `no_target_without_service_type_or_station_wind_bools` | `0.813` | `7.061` | `9.210` | `44.02` |
| `O3_giardini_margherita` | `3` | `xgboost` | `no_target_without_service_type` | `0.614` | `10.045` | `13.053` | `66.62` |
| `O3_giardini_margherita` | `6` | `xgboost` | `no_target_without_service_type_or_station_wind_bools` | `0.427` | `12.330` | `15.882` | `82.95` |
| `O3_giardini_margherita` | `12` | `xgboost` | `no_target_without_station_wind_bools` | `0.274` | `14.041` | `17.720` | `112.20` |
| `O3_giardini_margherita` | `24` | `adaboost` | `no_target_autoregressive` | `0.173` | `16.052` | `19.251` | `147.65` |
| `O3_via_chiarini` | `1` | `xgboost` | `no_target_without_service_type_or_station_wind_bools` | `0.766` | `8.906` | `11.421` | `46.94` |
| `O3_via_chiarini` | `3` | `xgboost` | `no_target_without_service_type_or_station_wind_bools` | `0.623` | `11.171` | `14.580` | `65.55` |
| `O3_via_chiarini` | `6` | `xgboost` | `no_target_without_service_type` | `0.439` | `13.707` | `17.761` | `84.37` |
| `O3_via_chiarini` | `12` | `xgboost` | `no_target_without_station_wind_bools` | `0.422` | `14.250` | `18.067` | `102.97` |
| `O3_via_chiarini` | `24` | `xgboost` | `no_target_without_service_type` | `0.342` | `15.414` | `19.417` | `119.03` |

Lettura principale:

- `O3` e `C6H6` sono i target piu' leggibili da feature esterne;
- `CO` e' intermedio;
- `NO2`, soprattutto a Porta San Felice, resta difficile senza storico del
  target;
- `ExtraTrees` e `AdaBoost` compaiono come migliori in pochi casi specifici,
  soprattutto a orizzonte lungo.

## 7. Risultati predittivi: vista con storico del target

Questa vista misura il potenziale predittivo operativo quando la persistenza del
target viene concessa al modello.

| target | h | model | feature set | R2 | MAE | RMSE | MAPE |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `C6H6_porta_san_felice` | `1` | `xgboost` | `with_target_without_station_wind_bools` | `0.677` | `0.190` | `0.279` | `19.83` |
| `C6H6_porta_san_felice` | `3` | `xgboost` | `with_target_autoregressive` | `0.448` | `0.261` | `0.368` | `28.96` |
| `C6H6_porta_san_felice` | `6` | `xgboost` | `with_target_autoregressive` | `0.336` | `0.291` | `0.401` | `33.74` |
| `C6H6_porta_san_felice` | `12` | `xgboost` | `with_target_autoregressive` | `0.310` | `0.299` | `0.407` | `35.65` |
| `C6H6_porta_san_felice` | `24` | `xgboost` | `with_target_without_station_wind_bools` | `0.221` | `0.320` | `0.435` | `36.86` |
| `CO_porta_san_felice` | `1` | `xgboost` | `with_target_without_station_wind_bools` | `0.717` | `0.066` | `0.095` | `12.87` |
| `CO_porta_san_felice` | `3` | `xgboost` | `with_target_without_station_wind_bools` | `0.505` | `0.099` | `0.134` | `19.60` |
| `CO_porta_san_felice` | `6` | `xgboost` | `with_target_without_service_type_or_station_wind_bools` | `0.419` | `0.111` | `0.146` | `22.26` |
| `CO_porta_san_felice` | `12` | `xgboost` | `with_target_without_service_type` | `0.361` | `0.116` | `0.154` | `23.03` |
| `CO_porta_san_felice` | `24` | `xgboost` | `with_target_autoregressive` | `0.320` | `0.120` | `0.159` | `23.89` |
| `NO2_giardini_margherita` | `1` | `xgboost` | `with_target_autoregressive` | `0.781` | `2.422` | `3.561` | `21.47` |
| `NO2_giardini_margherita` | `3` | `xgboost` | `with_target_without_service_type_or_station_wind_bools` | `0.369` | `4.457` | `6.119` | `44.25` |
| `NO2_giardini_margherita` | `6` | `xgboost` | `with_target_without_service_type_or_station_wind_bools` | `0.166` | `5.316` | `6.954` | `61.81` |
| `NO2_giardini_margherita` | `12` | `xgboost` | `with_target_without_service_type_or_station_wind_bools` | `0.089` | `5.541` | `7.154` | `70.99` |
| `NO2_giardini_margherita` | `24` | `extra_trees` | `with_target_autoregressive` | `0.004` | `5.840` | `7.491` | `81.72` |
| `NO2_porta_san_felice` | `1` | `xgboost` | `with_target_without_service_type_or_station_wind_bools` | `0.788` | `4.192` | `5.723` | `15.42` |
| `NO2_porta_san_felice` | `3` | `xgboost` | `with_target_autoregressive` | `0.503` | `6.644` | `8.706` | `26.49` |
| `NO2_porta_san_felice` | `6` | `ridge` | `with_target_without_service_type_or_station_wind_bools` | `0.342` | `7.632` | `9.810` | `34.04` |
| `NO2_porta_san_felice` | `12` | `xgboost` | `with_target_autoregressive` | `0.325` | `7.770` | `9.928` | `33.70` |
| `NO2_porta_san_felice` | `24` | `xgboost` | `with_target_without_service_type` | `0.368` | `7.596` | `9.720` | `34.12` |
| `NO2_via_chiarini` | `1` | `xgboost` | `with_target_without_service_type` | `0.748` | `3.210` | `4.761` | `18.06` |
| `NO2_via_chiarini` | `3` | `xgboost` | `with_target_without_service_type_or_station_wind_bools` | `0.381` | `5.406` | `7.451` | `32.82` |
| `NO2_via_chiarini` | `6` | `xgboost` | `with_target_autoregressive` | `0.236` | `6.141` | `8.229` | `39.96` |
| `NO2_via_chiarini` | `12` | `xgboost` | `with_target_without_service_type` | `0.237` | `6.151` | `8.229` | `40.46` |
| `NO2_via_chiarini` | `24` | `xgboost` | `with_target_without_service_type_or_station_wind_bools` | `0.210` | `6.266` | `8.395` | `40.61` |
| `O3_giardini_margherita` | `1` | `xgboost` | `with_target_autoregressive` | `0.915` | `4.407` | `6.179` | `26.44` |
| `O3_giardini_margherita` | `3` | `xgboost` | `with_target_without_service_type` | `0.684` | `9.004` | `11.807` | `59.25` |
| `O3_giardini_margherita` | `6` | `xgboost` | `with_target_without_service_type` | `0.465` | `11.991` | `15.339` | `83.74` |
| `O3_giardini_margherita` | `12` | `xgboost` | `with_target_without_service_type_or_station_wind_bools` | `0.329` | `13.682` | `17.339` | `102.28` |
| `O3_giardini_margherita` | `24` | `extra_trees` | `with_target_without_service_type_or_station_wind_bools` | `0.220` | `15.059` | `18.585` | `129.57` |
| `O3_via_chiarini` | `1` | `xgboost` | `with_target_without_station_wind_bools` | `0.912` | `5.085` | `7.069` | `27.58` |
| `O3_via_chiarini` | `3` | `xgboost` | `with_target_without_service_type` | `0.668` | `10.385` | `13.633` | `61.36` |
| `O3_via_chiarini` | `6` | `xgboost` | `with_target_without_service_type_or_station_wind_bools` | `0.469` | `13.239` | `17.153` | `86.63` |
| `O3_via_chiarini` | `12` | `xgboost` | `with_target_without_station_wind_bools` | `0.436` | `13.762` | `17.593` | `107.09` |
| `O3_via_chiarini` | `24` | `extra_trees` | `with_target_without_service_type_or_station_wind_bools` | `0.352` | `15.186` | `19.033` | `121.93` |

Lettura principale:

- il salto piu' forte con autoregressione si vede su `NO2` e `CO`;
- `C6H6` migliora, ma molto meno di `NO2`;
- `O3` era gia' forte senza storico e resta forte con storico;
- a lungo orizzonte `ExtraTrees` emerge davvero per alcuni target, soprattutto
  `O3_giardini_margherita`, `O3_via_chiarini` e `NO2_giardini_margherita`.

## 8. Quale modello vince davvero

La risposta corretta non e' piu' "vince sempre `XGBoost`", ma una formulazione
piu' precisa:

- `XGBoost` resta il modello dominante nel complesso;
- il dominio e' piu' netto a breve orizzonte;
- a `24h` entrano piu' spesso in gioco `ExtraTrees` e, in un caso, `AdaBoost`.

Esempi concreti di eccezioni:

- `NO2_giardini_margherita` a `24h`: migliore `ExtraTrees`, `R2 = 0.004`;
- `O3_giardini_margherita` a `24h`: migliore `ExtraTrees`, `R2 = 0.220`;
- `O3_via_chiarini` a `24h`: migliore `ExtraTrees`, `R2 = 0.352`;
- `NO2_porta_san_felice` a `12h` senza autoregressione: migliore `AdaBoost`,
  `R2 = -0.062`, comunque in una zona difficile per tutti i modelli.

Interpretazione:

- il problema e' chiaramente non lineare;
- gli ensemble ad alberi sono la famiglia giusta;
- `XGBoost` resta il riferimento principale;
- `ExtraTrees` e' il challenger piu' credibile sui target piu' regolari o piu'
  rumorosi a lungo orizzonte.

## 9. Single-target vs multioutput

Il multioutput non sostituisce il single-target come baseline generale migliore,
ma produce alcuni miglioramenti misurabili.

Casi in cui il multioutput supera il miglior single-target:

| target | h | feature set multioutput | R2 multioutput | best single-target R2 | delta |
| --- | ---: | --- | ---: | ---: | ---: |
| `NO2_giardini_margherita` | `24` | `no_pollutant_context_without_service_type_or_station_wind_bools` | `0.070` | `0.004` | `+0.066` |
| `NO2_giardini_margherita` | `24` | `no_pollutant_context_without_station_wind_bools` | `0.051` | `0.004` | `+0.047` |
| `NO2_giardini_margherita` | `24` | `no_pollutant_context_without_service_type` | `0.048` | `0.004` | `+0.044` |
| `C6H6_porta_san_felice` | `24` | `with_pollutant_context_without_station_wind_bools` | `0.250` | `0.221` | `+0.029` |
| `NO2_giardini_margherita` | `12` | `with_pollutant_context` | `0.118` | `0.089` | `+0.028` |
| `O3_via_chiarini` | `24` | `with_pollutant_context_without_service_type_or_station_wind_bools` | `0.371` | `0.352` | `+0.019` |
| `C6H6_porta_san_felice` | `3` | `with_pollutant_context` | `0.465` | `0.448` | `+0.017` |
| `NO2_via_chiarini` | `6` | `with_pollutant_context_without_service_type` | `0.247` | `0.236` | `+0.011` |

Conclusione:

- il contesto multi-stazione e multi-inquinante contiene informazione reale;
- il vantaggio non e' uniforme;
- i guadagni piu' netti si vedono su `NO2_giardini_margherita`,
  `C6H6_porta_san_felice` e `O3_via_chiarini`.

## 10. Ablazioni: cosa conta davvero

### 10.1. Gerarchia media dei gruppi

Ablazione estesa single-target `xgboost`, media su target e orizzonti:

| gruppo rimosso | mean delta R2 | lettura |
| --- | ---: | --- |
| `meteo` | `+0.087` | contributo medio piu' forte |
| `other_pollutants` | `+0.026` | secondo blocco medio piu' importante |
| `other_pollutants_porta_san_felice` | `+0.021` | contesto locale forte su PSF |
| `rolling_features` | `+0.017` | memoria recente aggregata molto utile |
| `other_pollutants_giardini_margherita` | `+0.006` | contributo locale positivo ma secondario |
| `urban_traffic` | `+0.004` | contributo medio positivo ma contenuto |
| `diff_features` | `+0.003` | contributo piccolo ma reale |
| `lag_features` | `+0.002` | meno forti delle rolling |
| `wind_transport` | `+0.002` | contributo medio piccolo |
| `airport` | `+0.001` | positivo ma debole in media |
| `station_wind_bools` | `-0.001` | quasi nullo in media |
| `airport_service_type` | `-0.002` | selettivo, non uniforme |
| `airport_wind_interaction` | `-0.003` | selettivo, non uniforme |

### 10.2. Dove il blocco aeroporto aiuta davvero

Il blocco `airport` aggregato mostra i guadagni piu' leggibili soprattutto su:

- `O3_giardini_margherita` a `24h`: `delta R2 = +0.074`
- `C6H6_porta_san_felice` a `24h`: `+0.047`
- `O3_giardini_margherita` a `12h`: `+0.039`
- `NO2_porta_san_felice` a `24h`: `+0.027`

La scomposizione `airport_service_type` emerge soprattutto su:

- `C6H6_porta_san_felice` a `24h`: `+0.043`
- `NO2_porta_san_felice` a `12h`: `+0.037`
- `C6H6_porta_san_felice` a `12h`: `+0.026`
- `NO2_porta_san_felice` a `1h`: `+0.023`
- `NO2_porta_san_felice` a `6h`: `+0.021`
- `CO_porta_san_felice` a `3h`: `+0.016`

I booleani `station_wind_bools` mostrano i casi piu' chiari su:

- `CO_porta_san_felice` a `24h`: `+0.081`
- `NO2_giardini_margherita` a `24h`: `+0.053`
- `CO_porta_san_felice` a `12h`: `+0.030`

La rimozione congiunta `service_type + station_wind_bools` concentra le perdite
piu' leggibili su:

- `CO_porta_san_felice` a `24h`: `+0.093`
- `NO2_giardini_margherita` a `24h`: `+0.052 / +0.053`
- `NO2_porta_san_felice` a `12h`: `+0.031`
- `NO2_porta_san_felice` a `1h-3h` senza autoregressione: `+0.028`, `+0.028`

### 10.3. Lettura corretta dell'ablazione

Tre punti:

1. i blocchi aeroportuali raffinati non sono decorativi;
2. il loro contributo e' selettivo, non medio-strutturale;
3. la struttura robusta del problema resta guidata da:
   - `meteo`
   - `other_pollutants`
   - `rolling_features`

## 11. SHAP e feature importance

### 11.1. Lettura per gruppi

Le SHAP di gruppo confermano la struttura gia' vista nelle ablazioni.

| target | gruppo 1 | gruppo 2 | gruppo 3 | lettura |
| --- | --- | --- | --- | --- |
| `NO2_porta_san_felice` | `rolling_features (12.62)` | `target_autoregressive (7.70)` | `meteo (7.41)` | memoria recente dominante |
| `CO_porta_san_felice` | `rolling_features (0.191)` | `target_autoregressive (0.154)` | `meteo (0.101)` | persistenza + meteo |
| `C6H6_porta_san_felice` | `rolling_features (0.448)` | `meteo (0.281)` | `lag_features / other_pollutants (~0.223)` | target piu' leggibile da fattori esterni |
| `NO2_giardini_margherita` | `rolling_features (7.65)` | `meteo (6.88)` | `target_autoregressive (3.23)` | meteo e memoria pesano insieme |
| `NO2_via_chiarini` | `rolling_features (10.39)` | `target_autoregressive (6.57)` | `meteo (5.94)` | dinamica recente molto forte |
| `O3_giardini_margherita` | `rolling_features (31.92)` | `meteo (29.55)` | `target_autoregressive (23.27)` | struttura fortemente regolare |
| `O3_via_chiarini` | `rolling_features (33.66)` | `meteo (33.01)` | `target_autoregressive (22.65)` | meteo e rolling dominano |

### 11.2. Cosa aggiungono i `service_type`

Le feature `blq_service_*` compaiono davvero nelle importance:

- su `NO2_porta_san_felice` emergono soprattutto cargo, mail e combined;
- su `NO2_via_chiarini` emergono cargo, combined e charter;
- su `NO2_giardini_margherita` compaiono cargo, charter, scheduled e mail;
- su `O3` entrano spesso charter e cargo;
- su `CO` e `C6H6` l'effetto esiste ma resta piu' contenuto.

### 11.3. Lettura finale della parte interpretativa

- le SHAP confermano che i nuovi blocchi aeroportuali non sono fittizi;
- le ablazioni chiariscono che il loro impatto e' selettivo;
- la repo, letta correttamente, non supporta una narrativa semplice del tipo
  "BLQ domina il sistema".

## 12. Risultati `upwind/downwind`

### 12.1. Contrasto descrittivo `downwind - upwind`

| target | unita' | downwind mean | upwind mean | downwind - upwind |
| --- | --- | ---: | ---: | ---: |
| `NO2_porta_san_felice` | `ug/m3` | `28.75` | `29.65` | `-0.90` |
| `CO_porta_san_felice` | `mg/m3` | `0.554` | `0.468` | `+0.086` |
| `C6H6_porta_san_felice` | `ug/m3` | `1.135` | `0.973` | `+0.163` |
| `NO2_giardini_margherita` | `ug/m3` | `17.37` | `14.27` | `+3.10` |
| `NO2_via_chiarini` | `ug/m3` | `17.26` | `16.91` | `+0.35` |
| `O3_giardini_margherita` | `ug/m3` | `38.22` | `55.82` | `-17.61` |
| `O3_via_chiarini` | `ug/m3` | `34.11` | `55.57` | `-21.46` |

Primo messaggio:

- `CO` e `C6H6` a PSF sono piu' alti in downwind;
- `NO2_porta_san_felice` no;
- `NO2_giardini_margherita` si';
- `O3` sulle stazioni esterne va nella direzione opposta.

### 12.2. Matching `downwind/upwind`

| target | mean diff downwind - upwind | p-value | lettura |
| --- | ---: | ---: | --- |
| `NO2_porta_san_felice` | `-1.33` | `0.0030` | piu' basso in downwind anche dopo matching |
| `CO_porta_san_felice` | `+0.0186` | `0.0122` | segnale positivo piccolo ma robusto |
| `C6H6_porta_san_felice` | `+0.0152` | `0.3870` | differenza non robusta |
| `NO2_giardini_margherita` | `+0.843` | `0.0024` | segnale positivo robusto |
| `NO2_via_chiarini` | `-0.625` | `0.0555` | ambiguo |
| `O3_giardini_margherita` | `-5.66` | `<0.001` | molto piu' basso in downwind |
| `O3_via_chiarini` | `-8.98` | `<0.001` | molto piu' basso in downwind |

### 12.3. Bootstrap a blocchi

| target | mean effect downwind - upwind | CI95 | lettura |
| --- | ---: | --- | --- |
| `NO2_porta_san_felice` | `-1.03` | attraversa `0` | segno negativo ma non robusto |
| `CO_porta_san_felice` | `+0.087` | tutto positivo | segnale positivo stabile |
| `C6H6_porta_san_felice` | `+0.164` | tutto positivo | segnale positivo abbastanza stabile |
| `NO2_giardini_margherita` | `+3.08` | tutto positivo | segnale positivo netto |
| `NO2_via_chiarini` | `+0.36` | attraversa `0` | effetto debole/incerto |
| `O3_giardini_margherita` | `-17.72` | tutto negativo | segnale negativo molto forte |
| `O3_via_chiarini` | `-21.52` | tutto negativo | segnale negativo molto forte |

### 12.4. `high_downwind - low_downwind`

| target | effect high_downwind - low_downwind | lettura |
| --- | ---: | --- |
| `NO2_porta_san_felice` | `+2.72` | cresce nelle ore downwind ad alta BLQ |
| `CO_porta_san_felice` | `-0.136` | segno opposto a una relazione monotona semplice |
| `C6H6_porta_san_felice` | `-0.174` | segno opposto a una relazione monotona semplice |
| `NO2_giardini_margherita` | `-3.17` | segno opposto |
| `NO2_via_chiarini` | `-6.22` | segno opposto |
| `O3_giardini_margherita` | `+34.15` | fortissimo aumento |
| `O3_via_chiarini` | `+41.15` | fortissimo aumento |

Questa tabella e' importante perche' mostra che BLQ, chimica locale, mixing
atmosferico e traffico non si riducono a una relazione monotona banale.

### 12.5. Sensibilita' alla soglia

| target | diff @0.30 | diff @0.50 | diff @0.70 | diff @0.85 | lettura |
| --- | ---: | ---: | ---: | ---: | --- |
| `NO2_porta_san_felice` | `-1.16` | `-0.90` | `-0.93` | `-0.73` | sempre negativo |
| `CO_porta_san_felice` | `+0.076` | `+0.086` | `+0.083` | `+0.044` | sempre positivo |
| `C6H6_porta_san_felice` | `+0.138` | `+0.163` | `+0.153` | `+0.096` | sempre positivo |
| `NO2_giardini_margherita` | `+2.89` | `+3.10` | `+3.43` | `+2.59` | sempre positivo |
| `NO2_via_chiarini` | `+0.50` | `+0.35` | `-0.54` | `-3.22` | cambia segno |
| `O3_giardini_margherita` | `-18.44` | `-17.61` | `-15.23` | `-6.60` | sempre negativo |
| `O3_via_chiarini` | `-22.94` | `-21.46` | `-17.30` | `-5.48` | sempre negativo |

### 12.6. Regressioni `BLQ x downwind`

Messaggio chiave:

- il termine `blq_x_downwind` non produce una firma semplice e coerente per tutti
  i target;
- `NO2_porta_san_felice` non supporta la narrativa piu' ingenua;
- `O3` sulle stazioni esterne mostra pattern robusti ma di segno opposto.

### 12.7. Gradienti spaziali

I gradienti DID mostrano che:

- esistono differenze spaziali legate al regime di vento;
- ma non tutte si allineano a una narrativa monotona "piu' vicino alla traiettoria
  aeroporto -> piu' alto";
- il sistema reale dipende da stazione, inquinante e orizzonte.

### 12.8. Lettura complessiva della parte fisica

- `NO2_porta_san_felice`: non mostra il pattern aeroportuale semplice atteso;
- `CO_porta_san_felice`: segnale downwind positivo piccolo ma stabile;
- `C6H6_porta_san_felice`: segnale positivo descrittivo, meno convincente dopo
  matching;
- `NO2_giardini_margherita`: uno dei casi piu' netti a favore di un effetto
  downwind positivo;
- `NO2_via_chiarini`: ambiguo;
- `O3`: pattern downwind robustamente negativo sulle stazioni esterne.

Conclusione fisica prudente:

- non emerge una firma aeroportuale unica e generalizzata;
- esistono segnali compatibili con un ruolo di BLQ e del vento;
- l'effetto cambia molto tra inquinanti e stazioni;
- una spiegazione monocausale non e' supportata.

## 13. Confronto esplicito tra inquinanti

La nuova analisi `cross_pollutant` ordina in una sintesi unica cio' che nelle
sezioni precedenti era distribuito tra forecasting, ablazioni e contrasti di
vento.

### 13.1. Sintesi per target

| target | gain autoregressivo medio R2 | top group 1 | delta | top group 2 | delta | downwind-upwind std units | matched std units | bootstrap std units | soglia stabile |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `C6H6_porta_san_felice` | `0.046` | `other_pollutants` | `0.096` | `meteo` | `0.054` | `0.247` | `0.023` | `0.248` | `1` |
| `CO_porta_san_felice` | `0.240` | `meteo` | `0.261` | `other_pollutants` | `0.100` | `0.296` | `0.064` | `0.301` | `1` |
| `NO2_giardini_margherita` | `0.106` | `meteo` | `0.192` | `urban_traffic` | `0.010` | `0.323` | `0.088` | `0.321` | `1` |
| `NO2_porta_san_felice` | `0.424` | `other_pollutants` | `0.095` | `rolling_features` | `0.056` | `-0.066` | `-0.098` | `-0.076` | `1` |
| `NO2_via_chiarini` | `0.251` | `meteo` | `0.131` | `other_pollutants` | `0.081` | `0.034` | `-0.060` | `0.035` | `0` |
| `O3_giardini_margherita` | `0.062` | `meteo` | `0.272` | `rolling_features` | `0.070` | `-0.471` | `-0.151` | `-0.474` | `1` |
| `O3_via_chiarini` | `0.049` | `meteo` | `0.090` | `other_pollutants` | `0.052` | `-0.548` | `-0.229` | `-0.549` | `1` |

### 13.2. Sintesi per famiglia chimica

| pollutant | best no-auto R2 1h | best no-auto R2 24h | best with-auto R2 1h | best with-auto R2 24h | gain autoregressivo medio | raw downwind-upwind std units | matched std units | bootstrap std units | high_downwind-low_downwind std units | soglia stabile | top groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `C6H6` | `0.578` | `0.179` | `0.677` | `0.221` | `0.046` | `0.247` | `0.023` | `0.248` | `-0.264` | `1.00` | `other_pollutants`, `meteo`, `rolling_features` |
| `CO` | `0.417` | `0.087` | `0.717` | `0.320` | `0.240` | `0.296` | `0.064` | `0.301` | `-0.466` | `1.00` | `meteo`, `other_pollutants`, `rolling_features` |
| `NO2` | `0.342` | `-0.002` | `0.772` | `0.194` | `0.261` | `0.097` | `-0.023` | `0.093` | `-0.242` | `0.67` | `meteo`, `other_pollutants`, `rolling_features` |
| `O3` | `0.790` | `0.258` | `0.914` | `0.286` | `0.056` | `-0.510` | `-0.190` | `-0.512` | `0.982` | `1.00` | `meteo`, `other_pollutants`, `other_pollutants` |

### 13.3. Lettura finale del confronto

- `NO2` e' la famiglia piu' autoregressiva;
- `CO` e' intermedio ma ancora molto dipendente dallo storico;
- `C6H6` e' il target piu' leggibile da fattori esterni a Porta San Felice;
- `O3` e' il piu' meteorologico, regolare e robusto nel confronto standardizzato
  dei regimi di vento.

## 14. Cosa si puo' sostenere davvero

### 14.1. Cose supportate dai risultati

- le feature esterne contengono segnale reale;
- il valore di questo segnale cambia molto tra target;
- `NO2` e' fortemente guidato dalla persistenza;
- `CO` ha un comportamento intermedio;
- `C6H6` e' piu' leggibile da fattori esterni;
- `O3` e' molto regolare e fortemente meteorologico;
- il contesto multi-stazione contiene informazione reale;
- `service_type` aggiunge segnale predittivo in casi selettivi;
- il vento e la geometria aeroporto -> stazione spiegano parte del quadro, ma in
  modo non uniforme.

### 14.2. Cose non supportate

- causalita' forte;
- attribuzione pulita di dominanza sorgente;
- una narrativa semplice del tipo "BLQ aumenta sempre il target in downwind";
- una firma unica dell'aeroporto valida per tutti gli inquinanti.

## 15. Limiti interpretativi

I limiti principali sono:

- performance predittiva non implica causalita';
- autoregressione e interpretazione sono in tensione;
- traffico, meteo, calendario e altri inquinanti sono correlati;
- il multioutput dimostra dipendenza informativa, non causalita' tra stazioni;
- `airport_response` resta una parte esplicativa, non inferenziale;
- anche `upwind/downwind` resta osservazionale, non sperimentale.

## 16. Conclusione attuale

La repo ora racconta in modo coerente una storia piu' precisa di quella
sostenibile nelle versioni iniziali.

1. Il sistema contiene segnale predittivo reale proveniente da traffico, meteo,
   aeroporto e contesto multi-stazione.
2. Il peso relativo di questi blocchi cambia molto tra target.
3. La struttura media del problema e' guidata soprattutto da:
   - `meteo`
   - `other_pollutants`
   - `rolling_features`
4. I blocchi aeroportuali raffinati aggiungono informazione reale, ma soprattutto
   in casi selettivi.
5. `NO2` e' la famiglia piu' autoregressiva.
6. `CO` e' intermedio.
7. `C6H6` e' il target piu' leggibile da variabili esterne a Porta San Felice.
8. `O3` e' il target piu' meteorologico, regolare e robustamente distinguibile
   anche nei contrasti di vento.

La conclusione piu' difendibile non e' quindi "BLQ domina" ne' "BLQ non conta",
ma una formulazione piu' sobria:

- l'aeroporto e' una parte informativa del sistema;
- il suo effetto e' selettivo, non uniforme;
- la dinamica media resta governata soprattutto da meteo, dipendenze tra
  inquinanti e memoria recente aggregata;
- non emerge una firma aeroportuale unica e semplice valida per tutti i target.

## 17. Dove stanno i risultati completi

Il file presente qui e' una sintesi strutturata e completa dei risultati piu'
importanti. Le matrici complete, con tutte le righe e tutte le metriche grezze,
restano nei CSV.

Per leggere **tutti i risultati** senza perdita di dettaglio:

- metriche complete single-target:
  - `Analysis/slurm_full_explain/advanced_temporal_cv_summary.csv`
- predizioni out-of-sample:
  - `Analysis/slurm_full_explain/advanced_temporal_cv_predictions.csv`
- metriche complete multioutput:
  - `Analysis/slurm_full_explain/advanced_multioutput_xgboost_summary.csv`
- ablazioni mirate:
  - `Analysis/slurm_full_explain/advanced_ablation_summary.csv`
- ablazioni estese single-target:
  - `Analysis/slurm_full_explain/advanced_extended_ablation_delta_summary.csv`
- ablazioni estese multioutput:
  - `Analysis/slurm_full_explain/advanced_multioutput_extended_ablation_delta_summary.csv`
- SHAP e importance:
  - `Analysis/slurm_full_explain/advanced_group_shap.csv`
  - `Analysis/slurm_full_explain/advanced_multioutput_group_shap.csv`
  - `Analysis/slurm_full_explain/advanced_xgboost_native_feature_importances_summary.csv`
  - `Analysis/slurm_full_explain/advanced_multioutput_xgboost_native_feature_importances_summary.csv`
- contrasti e regressioni `upwind/downwind`:
  - `Analysis/slurm_full_upwind/upwind_downwind_summary.csv`
  - `Analysis/slurm_full_upwind/upwind_downwind_blq_effects.csv`
  - `Analysis/slurm_full_upwind/upwind_downwind_matched_summary.csv`
  - `Analysis/slurm_full_upwind/upwind_downwind_bootstrap_effects.csv`
  - `Analysis/slurm_full_upwind/upwind_downwind_threshold_sensitivity.csv`
  - `Analysis/slurm_full_upwind/multistation_did_summary.csv`
- sintesi comparative:
  - `Analysis/cross_pollutant/cross_pollutant_overview.csv`
  - `Analysis/cross_pollutant/cross_pollutant_family_overview.csv`

Questo file e questi CSV, letti insieme, coprono sia la parte descrittiva sia la
parte quantitativa completa.
