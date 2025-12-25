"""
mesh.py - Classe Mesh3D per la discretizzazione del dominio

Implementa una mesh cartesiana 3D strutturata con:
- Allocazione memoria per campi scalari (temperatura, materiali, sorgenti)
- Mapping indici 3D <-> 1D
- Supporto per mesh non uniforme (futuro)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
from enum import IntEnum


class MaterialID(IntEnum):
    """Identificatori dei materiali"""
    AIR = 0
    SAND = 1
    INSULATION = 2
    STEEL = 3
    TUBES = 4
    HEATERS = 5
    GROUND = 6
    CONCRETE = 7


class BoundaryType(IntEnum):
    """Tipi di condizione al contorno"""
    INTERNAL = 0        # Nodo interno
    DIRICHLET = 1       # Temperatura fissa
    NEUMANN = 2         # Flusso imposto
    CONVECTION = 3      # Convezione (Robin)
    SYMMETRY = 4        # Simmetria (flusso nullo)


@dataclass
class NodeProperties:
    """Proprietà termiche di un singolo nodo"""
    k: float = 1.0          # Conducibilità termica [W/(m·K)]
    rho: float = 1000.0     # Densità [kg/m³]
    cp: float = 1000.0      # Calore specifico [J/(kg·K)]
    Q: float = 0.0          # Sorgente di calore volumetrica [W/m³]
    
    @property
    def alpha(self) -> float:
        """Diffusività termica [m²/s]"""
        return self.k / (self.rho * self.cp)


@dataclass
class Mesh3D:
    """
    Mesh cartesiana 3D strutturata con spaziatura UNIFORME (dx = dy = dz).
    
    Attributes:
        Lx, Ly, Lz: Dimensioni del dominio [m]
        Nx, Ny, Nz: Numero di celle per direzione (calcolato automaticamente se uniform=True)
        dx, dy, dz: Spaziatura delle celle [m] (uguali se uniform=True)
        d: Spaziatura uniforme [m]
    """
    
    # Dimensioni dominio
    Lx: float = 10.0
    Ly: float = 10.0
    Lz: float = 8.0
    
    # Numero di celle (usato solo se uniform=False)
    Nx: int = 50
    Ny: int = 50
    Nz: int = 40
    
    # Spaziatura target (usato se uniform=True)
    target_spacing: float = 0.2  # [m] - spaziatura desiderata
    uniform: bool = True  # Se True, forza dx = dy = dz
    
    # Campi calcolati dopo __post_init__
    dx: float = field(init=False)
    dy: float = field(init=False)
    dz: float = field(init=False)
    N_total: int = field(init=False)
    
    # Coordinate dei centri cella
    x: np.ndarray = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)
    z: np.ndarray = field(init=False, repr=False)
    
    # Griglie 3D di coordinate
    X: np.ndarray = field(init=False, repr=False)
    Y: np.ndarray = field(init=False, repr=False)
    Z: np.ndarray = field(init=False, repr=False)
    
    # Campi scalari
    material_id: np.ndarray = field(init=False, repr=False)
    boundary_type: np.ndarray = field(init=False, repr=False)
    
    # Proprietà termiche (campi 3D)
    k: np.ndarray = field(init=False, repr=False)
    rho: np.ndarray = field(init=False, repr=False)
    cp: np.ndarray = field(init=False, repr=False)
    Q: np.ndarray = field(init=False, repr=False)
    
    # Temperatura (soluzione)
    T: np.ndarray = field(init=False, repr=False)
    
    # Condizioni al contorno
    bc_h: np.ndarray = field(init=False, repr=False)      # Coefficiente convettivo
    bc_T_inf: np.ndarray = field(init=False, repr=False)  # Temperatura esterna
    bc_q: np.ndarray = field(init=False, repr=False)      # Flusso imposto
    
    # Spaziatura uniforme (calcolata in __post_init__)
    d: float = field(init=False)
    
    def __post_init__(self):
        """Inizializza la mesh e alloca la memoria"""
        
        if self.uniform:
            # Se Nx è stato fornito esplicitamente (diverso dal default 50), 
            # usalo per determinare la spaziatura. Altrimenti usa target_spacing.
            if self.Nx != 50:
                self.d = self.Lx / self.Nx
            else:
                self.d = self.target_spacing
            
            # Calcola numero di celle per avere spaziatura uniforme
            self.Nx = max(3, int(np.round(self.Lx / self.d)))
            self.Ny = max(3, int(np.round(self.Ly / self.d)))
            self.Nz = max(3, int(np.round(self.Lz / self.d)))
            
            # Spaziatura finale uniforme basata su Lx
            self.d = self.Lx / self.Nx
            self.dx = self.d
            self.dy = self.d
            self.dz = self.d
            
            # Aggiusta le dimensioni del dominio per garantire uniformità esatta
            self.Ly = self.Ny * self.d
            self.Lz = self.Nz * self.d
        else:
            # Mesh non uniforme (come prima)
            self.dx = self.Lx / self.Nx
            self.dy = self.Ly / self.Ny
            self.dz = self.Lz / self.Nz
            self.d = self.dx  # Per compatibilità
        
        self.N_total = self.Nx * self.Ny * self.Nz
        
        # Coordinate dei centri cella (1D)
        self.x = np.linspace(self.dx/2, self.Lx - self.dx/2, self.Nx)
        self.y = np.linspace(self.dy/2, self.Ly - self.dy/2, self.Ny)
        self.z = np.linspace(self.dz/2, self.Lz - self.dz/2, self.Nz)
        
        # Griglie 3D (meshgrid)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        
        # Inizializza campi a valori di default
        shape = (self.Nx, self.Ny, self.Nz)
        
        # ID materiale (default: aria)
        self.material_id = np.zeros(shape, dtype=np.int8)
        
        # Tipo boundary (default: interno)
        self.boundary_type = np.zeros(shape, dtype=np.int8)
        self._set_boundary_nodes()
        
        # Proprietà termiche (default: aria)
        self.k = np.ones(shape, dtype=np.float64) * 0.026      # Aria
        self.rho = np.ones(shape, dtype=np.float64) * 1.2      # Aria
        self.cp = np.ones(shape, dtype=np.float64) * 1005.0    # Aria
        self.Q = np.zeros(shape, dtype=np.float64)              # Nessuna sorgente
        
        # Temperatura iniziale
        self.T = np.ones(shape, dtype=np.float64) * 20.0       # Ambiente
        
        # Condizioni al contorno
        self.bc_h = np.zeros(shape, dtype=np.float64)
        self.bc_T_inf = np.ones(shape, dtype=np.float64) * 20.0
        self.bc_q = np.zeros(shape, dtype=np.float64)
    
    def _set_boundary_nodes(self):
        """Identifica automaticamente i nodi di bordo"""
        # Facce del dominio
        self.boundary_type[0, :, :] = BoundaryType.CONVECTION    # X = 0
        self.boundary_type[-1, :, :] = BoundaryType.CONVECTION   # X = Lx
        self.boundary_type[:, 0, :] = BoundaryType.CONVECTION    # Y = 0
        self.boundary_type[:, -1, :] = BoundaryType.CONVECTION   # Y = Ly
        self.boundary_type[:, :, 0] = BoundaryType.DIRICHLET     # Z = 0 (terreno)
        self.boundary_type[:, :, -1] = BoundaryType.CONVECTION   # Z = Lz (aria)
    
    # =========================================================================
    # METODI DI INDICIZZAZIONE
    # =========================================================================
    
    def ijk_to_linear(self, i: int, j: int, k: int) -> int:
        """Converte indici 3D in indice lineare"""
        return i + j * self.Nx + k * self.Nx * self.Ny
    
    def linear_to_ijk(self, p: int) -> Tuple[int, int, int]:
        """Converte indice lineare in indici 3D"""
        k = p // (self.Nx * self.Ny)
        remainder = p % (self.Nx * self.Ny)
        j = remainder // self.Nx
        i = remainder % self.Nx
        return i, j, k
    
    def get_position(self, i: int, j: int, k: int) -> Tuple[float, float, float]:
        """Restituisce le coordinate (x, y, z) del centro cella"""
        return self.x[i], self.y[j], self.z[k]
    
    def find_cell(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """Trova la cella che contiene il punto (x, y, z)"""
        i = int(x / self.dx)
        j = int(y / self.dy)
        k = int(z / self.dz)
        
        # Clamp agli indici validi
        i = max(0, min(i, self.Nx - 1))
        j = max(0, min(j, self.Ny - 1))
        k = max(0, min(k, self.Nz - 1))
        
        return i, j, k
    
    # =========================================================================
    # METODI DI ASSEGNAZIONE MATERIALI
    # =========================================================================
    
    def set_material_region_box(self, 
                                 material: MaterialID,
                                 x_min: float, x_max: float,
                                 y_min: float, y_max: float,
                                 z_min: float, z_max: float,
                                 props: NodeProperties):
        """Assegna un materiale a una regione rettangolare"""
        
        i_min, j_min, k_min = self.find_cell(x_min, y_min, z_min)
        i_max, j_max, k_max = self.find_cell(x_max, y_max, z_max)
        
        # Assicura che max >= min
        i_max = max(i_min, i_max)
        j_max = max(j_min, j_max)
        k_max = max(k_min, k_max)
        
        # Slicing per l'assegnazione
        self.material_id[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1] = material
        self.k[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1] = props.k
        self.rho[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1] = props.rho
        self.cp[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1] = props.cp
        self.Q[i_min:i_max+1, j_min:j_max+1, k_min:k_max+1] = props.Q
    
    def set_material_cylinder(self,
                               material: MaterialID,
                               center_x: float, center_y: float,
                               r_inner: float, r_outer: float,
                               z_min: float, z_max: float,
                               props: NodeProperties):
        """Assegna un materiale a una regione cilindrica (anello)"""
        
        k_min_idx = int(z_min / self.dz)
        k_max_idx = int(z_max / self.dz)
        
        # Itera su tutti i punti e verifica se sono nel cilindro
        for i in range(self.Nx):
            for j in range(self.Ny):
                x_c, y_c = self.x[i], self.y[j]
                r = np.sqrt((x_c - center_x)**2 + (y_c - center_y)**2)
                
                if r_inner <= r <= r_outer:
                    for k in range(max(0, k_min_idx), min(self.Nz, k_max_idx + 1)):
                        self.material_id[i, j, k] = material
                        self.k[i, j, k] = props.k
                        self.rho[i, j, k] = props.rho
                        self.cp[i, j, k] = props.cp
                        self.Q[i, j, k] = props.Q
    
    # =========================================================================
    # METODI DI CONDIZIONI AL CONTORNO
    # =========================================================================
    
    def set_convection_bc(self, face: str, h: float, T_inf: float):
        """
        Imposta condizione al contorno convettiva su una faccia.
        
        Args:
            face: 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
            h: Coefficiente convettivo [W/(m²·K)]
            T_inf: Temperatura del fluido [°C]
        """
        if face == 'x_min':
            self.bc_h[0, :, :] = h
            self.bc_T_inf[0, :, :] = T_inf
            self.boundary_type[0, :, :] = BoundaryType.CONVECTION
        elif face == 'x_max':
            self.bc_h[-1, :, :] = h
            self.bc_T_inf[-1, :, :] = T_inf
            self.boundary_type[-1, :, :] = BoundaryType.CONVECTION
        elif face == 'y_min':
            self.bc_h[:, 0, :] = h
            self.bc_T_inf[:, 0, :] = T_inf
            self.boundary_type[:, 0, :] = BoundaryType.CONVECTION
        elif face == 'y_max':
            self.bc_h[:, -1, :] = h
            self.bc_T_inf[:, -1, :] = T_inf
            self.boundary_type[:, -1, :] = BoundaryType.CONVECTION
        elif face == 'z_min':
            self.bc_h[:, :, 0] = h
            self.bc_T_inf[:, :, 0] = T_inf
            self.boundary_type[:, :, 0] = BoundaryType.CONVECTION
        elif face == 'z_max':
            self.bc_h[:, :, -1] = h
            self.bc_T_inf[:, :, -1] = T_inf
            self.boundary_type[:, :, -1] = BoundaryType.CONVECTION
        else:
            raise ValueError(f"Faccia non valida: {face}")
    
    def set_fixed_temperature_bc(self, face: str, T: float):
        """
        Imposta temperatura fissa (Dirichlet) su una faccia del dominio.
        
        Args:
            face: 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
            T: Temperatura fissa [°C]
        """
        if face == 'x_min':
            self.T[0, :, :] = T
            self.bc_T_inf[0, :, :] = T
            self.boundary_type[0, :, :] = BoundaryType.DIRICHLET
        elif face == 'x_max':
            self.T[-1, :, :] = T
            self.bc_T_inf[-1, :, :] = T
            self.boundary_type[-1, :, :] = BoundaryType.DIRICHLET
        elif face == 'y_min':
            self.T[:, 0, :] = T
            self.bc_T_inf[:, 0, :] = T
            self.boundary_type[:, 0, :] = BoundaryType.DIRICHLET
        elif face == 'y_max':
            self.T[:, -1, :] = T
            self.bc_T_inf[:, -1, :] = T
            self.boundary_type[:, -1, :] = BoundaryType.DIRICHLET
        elif face == 'z_min':
            self.T[:, :, 0] = T
            self.bc_T_inf[:, :, 0] = T
            self.boundary_type[:, :, 0] = BoundaryType.DIRICHLET
        elif face == 'z_max':
            self.T[:, :, -1] = T
            self.bc_T_inf[:, :, -1] = T
            self.boundary_type[:, :, -1] = BoundaryType.DIRICHLET
        else:
            raise ValueError(f"Faccia non valida: {face}. Usare: x_min, x_max, y_min, y_max, z_min, z_max")
    
    # =========================================================================
    # METODI UTILITY
    # =========================================================================
    
    def get_info(self) -> Dict[str, Any]:
        """Restituisce informazioni sulla mesh"""
        return {
            'dimensions': (self.Lx, self.Ly, self.Lz),
            'cells': (self.Nx, self.Ny, self.Nz),
            'spacing': (self.dx, self.dy, self.dz),
            'total_nodes': self.N_total,
            'memory_MB': self._estimate_memory() / 1e6
        }
    
    def _estimate_memory(self) -> float:
        """Stima la memoria utilizzata in bytes"""
        n_fields = 10  # Numero di array 3D
        bytes_per_float = 8
        bytes_per_int = 1
        
        float_memory = n_fields * self.N_total * bytes_per_float
        int_memory = 2 * self.N_total * bytes_per_int
        
        return float_memory + int_memory
    
    def get_temperature_slice(self, axis: str, position: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estrae una sezione 2D del campo di temperatura.
        
        Args:
            axis: 'x', 'y', o 'z'
            position: Posizione della sezione [m]
            
        Returns:
            coord1, coord2, T_slice: Coordinate e temperatura sulla sezione
        """
        if axis == 'x':
            idx = int(position / self.dx)
            idx = max(0, min(idx, self.Nx - 1))
            return self.Y[idx, :, :], self.Z[idx, :, :], self.T[idx, :, :]
        elif axis == 'y':
            idx = int(position / self.dy)
            idx = max(0, min(idx, self.Ny - 1))
            return self.X[:, idx, :], self.Z[:, idx, :], self.T[:, idx, :]
        elif axis == 'z':
            idx = int(position / self.dz)
            idx = max(0, min(idx, self.Nz - 1))
            return self.X[:, :, idx], self.Y[:, :, idx], self.T[:, :, idx]
        else:
            raise ValueError(f"Asse non valido: {axis}")
    
    def flatten_field(self, field: np.ndarray) -> np.ndarray:
        """Converte un campo 3D in vettore 1D (ordine column-major / Fortran)"""
        return field.ravel(order='F')
    
    def unflatten_field(self, vector: np.ndarray) -> np.ndarray:
        """Converte un vettore 1D in campo 3D"""
        return vector.reshape((self.Nx, self.Ny, self.Nz), order='F')


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    # Test creazione mesh
    mesh = Mesh3D(Lx=10, Ly=10, Lz=8, Nx=50, Ny=50, Nz=40)
    
    print("=== Mesh3D Info ===")
    info = mesh.get_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test indicizzazione
    print("\n=== Test Indicizzazione ===")
    i, j, k = 10, 20, 15
    p = mesh.ijk_to_linear(i, j, k)
    i2, j2, k2 = mesh.linear_to_ijk(p)
    print(f"  (i,j,k) = ({i},{j},{k}) -> p = {p} -> ({i2},{j2},{k2})")
    assert (i, j, k) == (i2, j2, k2), "Errore indicizzazione!"
    
    # Test posizione
    x, y, z = mesh.get_position(25, 25, 20)
    print(f"  Centro cella (25,25,20): ({x:.2f}, {y:.2f}, {z:.2f}) m")
    
    print("\n=== Test Completato ===")
