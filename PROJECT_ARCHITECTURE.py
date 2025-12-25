# =============================================================================
# PROGETTO: SAND BATTERY 3D THERMAL SIMULATION
# =============================================================================
# 
# Documento di Architettura e Piano di Sviluppo
# Data: 14 Dicembre 2025
# Versione: 1.0
#
# =============================================================================

"""
================================================================================
                        ARCHITETTURA DEL SISTEMA
================================================================================

OBIETTIVO:
Simulare il comportamento termico 3D di una Sand Battery utilizzando il metodo
delle differenze finite (FDM - Finite Difference Method) su una mesh 3D
strutturata.

================================================================================
STRUTTURA DEL PROGETTO
================================================================================

battery_simulation/
├── original_prompt.txt              # Prompt originale dell'utente
├── materials_database.py            # Database proprietà materiali
├── PROJECT_ARCHITECTURE.py          # Questo file - architettura
├── docs/
│   ├── 01_THEORY.md                 # Teoria scambio termico
│   ├── 02_NUMERICAL_METHODS.md      # Metodi numerici (FDM)
│   ├── 03_GEOMETRY.md               # Geometria della batteria
│   └── 04_GUI_DESIGN.md             # Design dell'interfaccia
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── mesh.py                  # Generazione mesh 3D
│   │   ├── materials.py             # Gestione materiali
│   │   ├── geometry.py              # Definizione geometria cilindrica
│   │   ├── boundary_conditions.py   # Condizioni al contorno
│   │   └── heat_transfer.py         # Equazioni scambio termico
│   ├── solver/
│   │   ├── __init__.py
│   │   ├── matrix_builder.py        # Costruzione matrice sistema
│   │   ├── steady_state.py          # Solver stazionario
│   │   ├── transient.py             # Solver transitorio
│   │   └── sparse_utils.py          # Utilities matrici sparse
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── power_balance.py         # Bilancio potenze
│   │   ├── energy_balance.py        # Bilancio energie
│   │   ├── exergy_analysis.py       # Analisi exergetica
│   │   └── performance_metrics.py   # Metriche computazionali
│   └── visualization/
│       ├── __init__.py
│       ├── vtk_renderer.py          # Rendering 3D con VTK
│       ├── slice_viewer.py          # Visualizzatore sezioni
│       ├── colormap.py              # Scale di colori
│       └── plots.py                 # Grafici 1D/2D
├── gui/
│   ├── __init__.py
│   ├── main_window.py               # Finestra principale
│   ├── input_panel.py               # Pannello input parametri
│   ├── viewer_3d.py                 # Widget visualizzazione 3D
│   ├── plot_widgets.py              # Widget grafici
│   └── simulation_control.py        # Controllo simulazione
├── tests/
│   ├── test_mesh.py
│   ├── test_solver.py
│   └── test_validation.py
├── config/
│   ├── default_config.yaml          # Configurazione default
│   └── materials.yaml               # Materiali in formato YAML
├── main.py                          # Entry point applicazione
└── requirements.txt                 # Dipendenze Python

================================================================================
SCELTA DEL LINGUAGGIO E LIBRERIE
================================================================================

LINGUAGGIO PRINCIPALE: Python
MOTIVAZIONE:
- Facilità di sviluppo GUI
- Ottime librerie scientifiche
- Interoperabilità con librerie C/C++/Fortran tramite wrapper

ACCELERAZIONE COMPUTAZIONALE:
1. NumPy + SciPy     - Operazioni vettorizzate e solver sparsi (LAPACK/BLAS)
2. Numba             - JIT compilation per loop critici (velocità ~C)
3. CuPy (opzionale)  - GPU acceleration con CUDA

VISUALIZZAZIONE 3D:
1. PyVista (VTK wrapper) - Visualizzazione mesh 3D interattiva
2. Matplotlib            - Grafici 2D

GUI:
1. PyQt6               - Framework GUI principale
2. PyQtGraph           - Grafici real-time veloci

ALTERNATIVE CONSIDERATE:
- Julia: Più veloce ma meno librerie GUI mature
- C++: Troppo complesso per prototipazione rapida
- MATLAB: Licenza costosa

================================================================================
MODELLO FISICO
================================================================================

EQUAZIONE GOVERNANTE:
Equazione del calore in 3D (conduzione + sorgenti)

    ρ * cp * ∂T/∂t = ∇·(k∇T) + Q

Dove:
- ρ = densità [kg/m³]
- cp = calore specifico [J/(kg·K)]
- T = temperatura [K]
- t = tempo [s]
- k = conducibilità termica [W/(m·K)]
- Q = sorgente/pozzo di calore [W/m³]

CASO STAZIONARIO (∂T/∂t = 0):
    ∇·(k∇T) + Q = 0
    
Discretizzato: A * T = b
- A = matrice sparsa dei coefficienti
- T = vettore temperature incognite
- b = vettore termini noti (sorgenti + boundary)

CASO TRANSITORIO:
    ρ * cp * (T^(n+1) - T^n) / Δt = ∇·(k∇T^(n+1)) + Q
    
Schema implicito per stabilità incondizionata.

================================================================================
DISCRETIZZAZIONE (METODO DIFFERENZE FINITE)
================================================================================

MESH:
- Griglia 3D cartesiana uniforme
- N punti per lato → N³ nodi totali
- Passo spaziale: Δx = Δy = Δz = L / (N-1)

NODO CENTRALE (i,j,k):
Scambio con 6 nodi adiacenti:
- (i+1,j,k), (i-1,j,k)  →  direzione x
- (i,j+1,k), (i,j-1,k)  →  direzione y
- (i,j,k+1), (i,j,k-1)  →  direzione z

DISCRETIZZAZIONE LAPLACIANO:
Per conduttività uniforme:

∇²T ≈ (T_{i+1} + T_{i-1} + T_{j+1} + T_{j-1} + T_{k+1} + T_{k-1} - 6*T_{ijk}) / Δx²

Per conduttività variabile (interfacce materiali):
Si usa la media armonica della conduttività:

k_eff = 2 * k1 * k2 / (k1 + k2)

================================================================================
STRUTTURA DATI MESH
================================================================================

Ogni nodo (i,j,k) contiene:

class Node:
    # Indici
    i, j, k: int                    # Posizione nella griglia
    index: int                      # Indice lineare = i + j*Nx + k*Nx*Ny
    
    # Posizione fisica
    x, y, z: float                  # Coordinate [m]
    
    # Proprietà materiale
    material_id: int                # ID materiale
    rho: float                      # Densità [kg/m³]
    cp: float                       # Calore specifico [J/(kg·K)]
    k: float                        # Conducibilità [W/(m·K)]
    
    # Stato
    T: float                        # Temperatura [°C o K]
    T_old: float                    # Temperatura step precedente
    
    # Sorgente
    Q: float                        # Potenza volumetrica [W/m³]
    
    # Velocità (per fluidi)
    vx, vy, vz: float              # Componenti velocità [m/s]
    
    # Flag
    is_boundary: bool              # Nodo al contorno
    boundary_type: str             # 'dirichlet', 'neumann', 'convection'
    boundary_value: float          # Valore condizione al contorno

================================================================================
GEOMETRIA DELLA BATTERIA
================================================================================

VOLUME DI CONTROLLO:
- Dominio cubico: L x L x L
- Centro in (L/2, L/2, L/2)

STRUTTURA RADIALE (dal centro verso l'esterno):

1. FLUIDO NEI TUBI (r < R_tubo)
   - Materiale: acqua o aria
   - Ha vettore velocità
   - Scambio convettivo con parete tubo

2. PARETE TUBI (R_tubo < r < R_tubo + s_tubo)
   - Materiale: acciaio inox
   - Conduzione radiale

3. MATRICE SABBIA + RESISTENZE (R_tubo + s_tubo < r < R_sand)
   - Materiale: sabbia (con packing factor)
   - Resistenze: sorgenti di calore Q
   - Scambio con aria nei pori

4. PARETE INTERNA ACCIAIO (R_sand < r < R_sand + s_acciaio_int)
   - Materiale: acciaio

5. ISOLAMENTO (R_sand + s_acciaio_int < r < R_isol)
   - Materiale: lana minerale
   - Bassa conducibilità

6. PARETE ESTERNA (R_isol < r < R_ext)
   - Materiale: acciaio

7. ARIA ESTERNA (r > R_ext, fuori dal cilindro)
   - Materiale: aria
   - Condizione al contorno: convezione

8. TERRENO (sotto il cilindro)
   - Materiale: terreno
   - Condizione: temperatura costante in profondità

================================================================================
TIPI DI ANALISI
================================================================================

1. ANALISI STAZIONARIA (STEADY-STATE)
   - Trova distribuzione T quando dT/dt = 0
   - Risolve sistema lineare: A * T = b
   - Metodo: Solver iterativo (Conjugate Gradient + Preconditioner)

2. ANALISI TRANSITORIA (TRANSIENT)
   - Evoluzione T(t) nel tempo
   - Schema temporale: Backward Euler (implicito)
   - Ad ogni timestep: (M/Δt + A) * T^(n+1) = M/Δt * T^n + b
   - Dove M = matrice masse (diagonale con ρ*cp*V)

3. ANALISI EXERGETICA
   - Exergia = E * (1 - T0/T)
   - Distruzione exergia per irreversibilità

================================================================================
BILANCI DI POTENZA
================================================================================

POTENZE IN INGRESSO:
- P_resistenze = Σ Q_i * V_i   [W]  (sorgenti volumetriche)
- P_elettrica = P_resistenze / η_elettrica

POTENZE IN USCITA:
- P_tubi = ṁ * cp * (T_out - T_in)  [W]  (calore estratto dai tubi)

POTENZE PERSE:
- P_dispersione_laterale = ∫ h*(T_s - T_∞) dA  [W]
- P_dispersione_fondo = k_terreno * A * (T_base - T_terreno) / L_terreno
- P_dispersione_top = simile a laterale

BILANCIO:
- Stazionario: P_in = P_out + P_perdite
- Transitorio: P_in = P_out + P_perdite + d(E_stored)/dt

================================================================================
FASI DI SVILUPPO
================================================================================

FASE 1: CORE ENGINE (Priorità ALTA)
- [ ] 1.1 Struttura mesh 3D
- [ ] 1.2 Assegnazione materiali a nodi
- [ ] 1.3 Costruzione matrice sistema
- [ ] 1.4 Solver stazionario base
- [ ] 1.5 Test con caso semplice (cubo isotermo)

FASE 2: GEOMETRIA (Priorità ALTA)
- [ ] 2.1 Generatore geometria cilindrica
- [ ] 2.2 Definizione strati radiali
- [ ] 2.3 Posizionamento tubi e resistenze
- [ ] 2.4 Condizioni al contorno

FASE 3: VISUALIZZAZIONE (Priorità MEDIA)
- [ ] 3.1 Rendering 3D mesh con PyVista
- [ ] 3.2 Colormap temperatura
- [ ] 3.3 Sezioni interattive
- [ ] 3.4 Grafici 1D

FASE 4: GUI (Priorità MEDIA)
- [ ] 4.1 Layout finestra principale
- [ ] 4.2 Pannello input parametri
- [ ] 4.3 Integrazione viewer 3D
- [ ] 4.4 Controllo simulazione

FASE 5: ANALISI AVANZATA (Priorità BASSA inizialmente)
- [ ] 5.1 Solver transitorio
- [ ] 5.2 Bilanci potenza/energia
- [ ] 5.3 Analisi exergetica
- [ ] 5.4 Performance metrics

FASE 6: OTTIMIZZAZIONE (Priorità BASSA)
- [ ] 6.1 Parallelizzazione con Numba
- [ ] 6.2 GPU support (CuPy)
- [ ] 6.3 Preconditioner avanzati

================================================================================
IPOTESI E SEMPLIFICAZIONI DA VALIDARE CON L'UTENTE
================================================================================

1. MESH CARTESIANA vs CILINDRICA
   IPOTESI: Uso mesh cartesiana su dominio cubico, con geometria cilindrica
            definita assegnando materiali diversi ai nodi.
   PRO: Più semplice da implementare, stessa Δx ovunque
   CONTRO: Approssimazione superfici curve, più nodi necessari
   ALTERNATIVA: Mesh cilindrica (r, θ, z) - più precisa ma più complessa

2. TUBI E RESISTENZE
   IPOTESI: Modellati come regioni con proprietà diverse, non come geometrie
            esatte. Resistenze = nodi con sorgente Q costante.
   DOMANDA: OK modellare così o serve geometria esatta dei tubi?

3. CONVEZIONE INTERNA
   IPOTESI: Fluido nei tubi trattato con condizione al contorno convettiva
            (coefficiente h dato), non risolvo fluidodinamica.
   DOMANDA: Serve CFD per flusso nei tubi o basta correlazione h = f(Re, Pr)?

4. ARIA NEI PORI DELLA SABBIA
   IPOTESI: Trattata come medium effettivo (conducibilità effettiva che
            include sabbia + aria nei pori).
   DOMANDA: OK o serve modello bifase più dettagliato?

5. CONDIZIONI AL CONTORNO
   IPOTESI: 
   - Aria esterna: convezione (h dato, T_∞ dato)
   - Terreno: temperatura costante a profondità fissata
   - Simmetria: nessuna (simulo tutto il dominio)
   DOMANDA: Ci sono altre condizioni da considerare?

6. TIMESTEP TRANSITORIO
   IPOTESI: Δt adattivo basato su criterio CFL/stabilità.
   DOMANDA: Preferisci Δt fisso o adattivo?

================================================================================
REQUISITI SISTEMA
================================================================================

MEMORIA STIMATA:
- N = 50  →   125.000 nodi  →  ~10 MB (gestibile)
- N = 100 → 1.000.000 nodi  →  ~100 MB (ok)
- N = 200 → 8.000.000 nodi  →  ~1 GB (richiede ottimizzazione)

TEMPO CALCOLO STIMATO (solver stazionario):
- N = 50:  ~1-5 secondi
- N = 100: ~30-120 secondi
- N = 200: ~10-30 minuti

GPU ACCELERATION:
Può ridurre tempi di 10-100x per mesh grandi.

================================================================================
"""

# =============================================================================
# PROSSIMI PASSI
# =============================================================================

"""
FASE 1 - PRIMA IMPLEMENTAZIONE:

1. Creare la struttura delle cartelle
2. Implementare classe Mesh3D base
3. Implementare assegnazione materiali
4. Costruire matrice sistema per caso semplice
5. Testare con un cubo con T fissata ai bordi

DOMANDE PER L'UTENTE PRIMA DI PROCEDERE:

1. Confermi che mesh cartesiana va bene per iniziare?
   (Poi eventualmente passiamo a cilindrica)

2. Per i tubi: ti va bene che li modelliamo come "regioni cilindriche"
   con una condizione al contorno convettiva, senza risolvere il flusso
   interno dettagliato?

3. Per le resistenze: le modelliamo come sorgenti di potenza distribuite
   (W/m³) in una regione cilindrica?

4. Che risoluzione iniziale preferisci? (es. N=50 per test veloci)

5. Il fluido nei tubi è acqua, aria o altro?

6. Vuoi iniziare con l'analisi stazionaria e poi aggiungere il transitorio?
"""
