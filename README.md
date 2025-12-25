# Battery Simulation Project

## Descrizione

Sistema di simulazione termica 3D per Sand Battery basato sul metodo delle differenze finite (FDM).

## Struttura del Progetto

```
battery_simulation/
├── src/
│   ├── core/           # Motore di calcolo
│   │   ├── mesh.py     # Classe Mesh3D
│   │   ├── materials.py # Gestione materiali
│   │   └── geometry.py # Definizione geometria batteria
│   ├── solver/         # Solutori numerici
│   │   ├── matrix_builder.py
│   │   └── steady_state.py
│   ├── analysis/       # Post-processing
│   │   └── power_balance.py
│   └── visualization/  # Visualizzazione PyVista
│       └── renderer.py
├── gui/                # Interfaccia PyQt6
│   └── main_window.py
├── config/             # Configurazioni
│   └── default_config.yaml
├── docs/               # Documentazione
│   ├── 01_THEORY.md
│   └── 02_FDM_DISCRETIZATION.md
├── tests/              # Unit tests
├── materials_database.py
└── PROJECT_ARCHITECTURE.py
```

## Installazione

```bash
# Creare ambiente virtuale
python -m venv venv
venv\Scripts\activate  # Windows

# Installare dipendenze
pip install numpy scipy pyvista pyqt6 pyvistaqt numba pyyaml
```

## Uso Rapido

```python
from src.core.mesh import Mesh3D
from src.core.materials import MaterialManager
from src.solver.steady_state import SteadyStateSolver

# Creare mesh
mesh = Mesh3D(Lx=10, Ly=10, Lz=8, N=50)

# Assegnare materiali
materials = MaterialManager()
mesh.assign_material_cylindrical(materials)

# Risolvere
solver = SteadyStateSolver(mesh)
T = solver.solve()
```

## Requisiti

- Python 3.10+
- NumPy
- SciPy
- PyVista
- PyQt6
- Numba (opzionale, per accelerazione)

## Licenza

MIT License
