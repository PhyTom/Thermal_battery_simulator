"""
materials.py - Gestione del database materiali

Fornisce:
- Proprietà termiche di tutti i materiali
- Calcolo proprietà effettive (mezzo poroso)
- Interpolazione con temperatura
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Callable
from enum import Enum


class MaterialType(Enum):
    """Categorie di materiali"""
    STORAGE = "storage"
    STRUCTURAL = "structural"
    INSULATION = "insulation"
    FLUID = "fluid"


@dataclass
class ThermalProperties:
    """
    Proprietà termiche di un materiale.
    
    Tutti i valori possono essere costanti o funzioni della temperatura.
    """
    name: str
    k: float                    # Conducibilità termica [W/(m·K)]
    rho: float                  # Densità [kg/m³]
    cp: float                   # Calore specifico [J/(kg·K)]
    T_max: float = 1000.0       # Temperatura massima operativa [°C]
    emissivity: float = 0.9     # Emissività superficiale
    
    # Funzioni per proprietà dipendenti dalla temperatura (opzionali)
    k_func: Optional[Callable[[float], float]] = None
    cp_func: Optional[Callable[[float], float]] = None
    
    def get_k(self, T: float) -> float:
        """Conducibilità a temperatura T"""
        if self.k_func is not None:
            return self.k_func(T)
        return self.k
    
    def get_cp(self, T: float) -> float:
        """Calore specifico a temperatura T"""
        if self.cp_func is not None:
            return self.cp_func(T)
        return self.cp
    
    @property
    def alpha(self) -> float:
        """Diffusività termica [m²/s]"""
        return self.k / (self.rho * self.cp)
    
    @property
    def volumetric_heat_capacity(self) -> float:
        """Capacità termica volumetrica [J/(m³·K)]"""
        return self.rho * self.cp


# =============================================================================
# DATABASE MATERIALI
# =============================================================================

# Materiali di stoccaggio (sabbia, rocce)
STORAGE_MATERIALS: Dict[str, ThermalProperties] = {
    "silica_sand": ThermalProperties(
        name="Sabbia Silicea",
        k=0.35,
        rho=1500,
        cp=800,
        T_max=1200,
        emissivity=0.9
    ),
    "olivine": ThermalProperties(
        name="Olivina",
        k=3.5,
        rho=3300,
        cp=900,
        T_max=1400,
        emissivity=0.85
    ),
    "steatite": ThermalProperties(
        name="Steatite (Pietra Ollare)",
        k=3.0,
        rho=2700,
        cp=980,
        T_max=1200,
        emissivity=0.9
    ),
    "basalt": ThermalProperties(
        name="Basalto",
        k=1.7,
        rho=2900,
        cp=850,
        T_max=1100,
        emissivity=0.9
    ),
    "magnetite": ThermalProperties(
        name="Magnetite",
        k=4.5,
        rho=5150,
        cp=650,
        T_max=600,
        emissivity=0.95
    ),
    "quartzite": ThermalProperties(
        name="Quarzite",
        k=5.0,
        rho=2650,
        cp=900,
        T_max=1400,
        emissivity=0.85
    ),
    "granite": ThermalProperties(
        name="Granito",
        k=2.5,
        rho=2700,
        cp=800,
        T_max=800,
        emissivity=0.9
    ),
}

# Materiali strutturali
STRUCTURAL_MATERIALS: Dict[str, ThermalProperties] = {
    "carbon_steel": ThermalProperties(
        name="Acciaio al Carbonio",
        k=50.0,
        rho=7850,
        cp=490,
        T_max=400,
        emissivity=0.8
    ),
    "stainless_steel": ThermalProperties(
        name="Acciaio Inossidabile 304",
        k=16.0,
        rho=8000,
        cp=500,
        T_max=800,
        emissivity=0.6
    ),
    "concrete": ThermalProperties(
        name="Calcestruzzo",
        k=1.4,
        rho=2400,
        cp=880,
        T_max=300,
        emissivity=0.9
    ),
}

# Materiali isolanti
INSULATION_MATERIALS: Dict[str, ThermalProperties] = {
    "rock_wool": ThermalProperties(
        name="Lana di Roccia",
        k=0.04,
        rho=100,
        cp=840,
        T_max=700,
        emissivity=0.9
    ),
    "glass_wool": ThermalProperties(
        name="Lana di Vetro",
        k=0.035,
        rho=30,
        cp=840,
        T_max=400,
        emissivity=0.9
    ),
    "calcium_silicate": ThermalProperties(
        name="Silicato di Calcio",
        k=0.07,
        rho=200,
        cp=840,
        T_max=1000,
        emissivity=0.9
    ),
    "ceramic_fiber": ThermalProperties(
        name="Fibra Ceramica",
        k=0.12,
        rho=130,
        cp=1130,
        T_max=1260,
        emissivity=0.9
    ),
    "perlite": ThermalProperties(
        name="Perlite Espansa",
        k=0.05,
        rho=100,
        cp=900,
        T_max=900,
        emissivity=0.9
    ),
}

# Fluidi
FLUID_MATERIALS: Dict[str, ThermalProperties] = {
    "air": ThermalProperties(
        name="Aria",
        k=0.026,
        rho=1.2,
        cp=1005,
        T_max=2000,
        emissivity=0.0
    ),
    "water": ThermalProperties(
        name="Acqua",
        k=0.60,
        rho=1000,
        cp=4186,
        T_max=100,
        emissivity=0.95
    ),
    "thermal_oil": ThermalProperties(
        name="Olio Diatermico",
        k=0.12,
        rho=900,
        cp=2100,
        T_max=350,
        emissivity=0.9
    ),
}


# =============================================================================
# CLASSE MATERIAL MANAGER
# =============================================================================

class MaterialManager:
    """
    Gestore centrale per i materiali della simulazione.
    
    Permette di:
    - Accedere alle proprietà dei materiali
    - Calcolare proprietà effettive per mezzi porosi
    - Gestire materiali custom
    """
    
    def __init__(self):
        # Unisci tutti i database
        self.materials: Dict[str, ThermalProperties] = {}
        self.materials.update(STORAGE_MATERIALS)
        self.materials.update(STRUCTURAL_MATERIALS)
        self.materials.update(INSULATION_MATERIALS)
        self.materials.update(FLUID_MATERIALS)
        
        # Materiale di default per i pori
        self.pore_fluid = "air"
    
    def get(self, name: str) -> ThermalProperties:
        """Restituisce le proprietà di un materiale"""
        if name not in self.materials:
            raise KeyError(f"Materiale non trovato: {name}")
        return self.materials[name]
    
    def list_materials(self, category: Optional[MaterialType] = None) -> list:
        """Lista i materiali disponibili"""
        if category == MaterialType.STORAGE:
            return list(STORAGE_MATERIALS.keys())
        elif category == MaterialType.STRUCTURAL:
            return list(STRUCTURAL_MATERIALS.keys())
        elif category == MaterialType.INSULATION:
            return list(INSULATION_MATERIALS.keys())
        elif category == MaterialType.FLUID:
            return list(FLUID_MATERIALS.keys())
        else:
            return list(self.materials.keys())
    
    def add_custom_material(self, key: str, props: ThermalProperties):
        """Aggiunge un materiale custom al database"""
        self.materials[key] = props
    
    # =========================================================================
    # CALCOLO PROPRIETÀ EFFETTIVE PER MEZZO POROSO
    # =========================================================================
    
    def compute_effective_properties(self, 
                                      solid: str, 
                                      porosity: float,
                                      fluid: Optional[str] = None) -> ThermalProperties:
        """
        Calcola le proprietà effettive di un mezzo poroso.
        
        Args:
            solid: Nome del materiale solido
            porosity: Porosità (frazione di vuoti, 0-1)
            fluid: Nome del fluido nei pori (default: aria)
            
        Returns:
            ThermalProperties: Proprietà effettive del mezzo
        """
        if fluid is None:
            fluid = self.pore_fluid
        
        solid_props = self.get(solid)
        fluid_props = self.get(fluid)
        
        phi = porosity
        
        # Conducibilità effettiva (media geometrica)
        k_eff = (solid_props.k ** (1 - phi)) * (fluid_props.k ** phi)
        
        # Densità e capacità termica volumetrica (media aritmetica)
        rho_eff = (1 - phi) * solid_props.rho + phi * fluid_props.rho
        
        # Capacità termica (media pesata sulla massa)
        cp_eff = ((1 - phi) * solid_props.rho * solid_props.cp + 
                  phi * fluid_props.rho * fluid_props.cp) / rho_eff
        
        return ThermalProperties(
            name=f"{solid_props.name} (poroso, φ={phi:.2f})",
            k=k_eff,
            rho=rho_eff,
            cp=cp_eff,
            T_max=min(solid_props.T_max, fluid_props.T_max),
            emissivity=solid_props.emissivity
        )
    
    def compute_packed_bed_properties(self,
                                       solid: str,
                                       packing_fraction: float = 0.63) -> ThermalProperties:
        """
        Calcola proprietà di un letto impaccato di particelle.
        
        Args:
            solid: Nome del materiale delle particelle
            packing_fraction: Frazione di solido (1 - porosità)
            
        Returns:
            ThermalProperties effettive
        """
        porosity = 1 - packing_fraction
        return self.compute_effective_properties(solid, porosity)
    
    # =========================================================================
    # UTILITÀ
    # =========================================================================
    
    def get_energy_density(self, 
                            material: str, 
                            T_high: float, 
                            T_low: float,
                            packing_fraction: float = 1.0) -> float:
        """
        Calcola la densità energetica [MJ/m³] per un materiale.
        
        Args:
            material: Nome del materiale
            T_high: Temperatura alta [°C]
            T_low: Temperatura bassa [°C]
            packing_fraction: Frazione di impaccamento
        """
        if packing_fraction < 1.0:
            props = self.compute_packed_bed_properties(material, packing_fraction)
        else:
            props = self.get(material)
        
        delta_T = T_high - T_low
        energy_density = props.rho * props.cp * delta_T / 1e6  # MJ/m³
        
        return energy_density
    
    def compare_materials(self, 
                          materials: list, 
                          T_high: float = 400.0, 
                          T_low: float = 100.0) -> dict:
        """Confronta le densità energetiche di diversi materiali"""
        results = {}
        for mat in materials:
            ed = self.get_energy_density(mat, T_high, T_low, packing_fraction=0.63)
            results[mat] = ed
        return results


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    manager = MaterialManager()
    
    print("=== Materiali di Stoccaggio ===")
    for name in manager.list_materials(MaterialType.STORAGE):
        props = manager.get(name)
        print(f"  {props.name}:")
        print(f"    k = {props.k:.2f} W/(m·K)")
        print(f"    ρ = {props.rho:.0f} kg/m³")
        print(f"    cp = {props.cp:.0f} J/(kg·K)")
        print(f"    ρ·cp = {props.volumetric_heat_capacity/1e6:.2f} MJ/(m³·K)")
    
    print("\n=== Proprietà Effettive Steatite (φ=0.37) ===")
    eff = manager.compute_packed_bed_properties("steatite", packing_fraction=0.63)
    print(f"  {eff.name}:")
    print(f"    k_eff = {eff.k:.3f} W/(m·K)")
    print(f"    ρ_eff = {eff.rho:.0f} kg/m³")
    print(f"    cp_eff = {eff.cp:.0f} J/(kg·K)")
    
    print("\n=== Confronto Densità Energetica (400→100°C, φ=0.63) ===")
    materials = ["silica_sand", "olivine", "steatite", "basalt", "magnetite"]
    comparison = manager.compare_materials(materials, T_high=400, T_low=100)
    for mat, ed in sorted(comparison.items(), key=lambda x: -x[1]):
        print(f"  {mat}: {ed:.1f} MJ/m³")
    
    print("\n=== Test Completato ===")
