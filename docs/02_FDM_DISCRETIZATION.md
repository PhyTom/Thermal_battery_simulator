# Discretizzazione FDM - Finite Difference Method

## 1. Introduzione al Metodo delle Differenze Finite

Il metodo delle differenze finite (FDM) approssima le derivate con rapporti incrementali,
trasformando le PDE in sistemi di equazioni algebriche.

---

## 2. Approssimazione delle Derivate

### 2.1 Derivata Prima

**Differenza in avanti (forward):**
$$\frac{\partial T}{\partial x}\bigg|_i \approx \frac{T_{i+1} - T_i}{\Delta x} + O(\Delta x)$$

**Differenza all'indietro (backward):**
$$\frac{\partial T}{\partial x}\bigg|_i \approx \frac{T_i - T_{i-1}}{\Delta x} + O(\Delta x)$$

**Differenza centrata:**
$$\frac{\partial T}{\partial x}\bigg|_i \approx \frac{T_{i+1} - T_{i-1}}{2\Delta x} + O(\Delta x^2)$$

### 2.2 Derivata Seconda

**Differenza centrata:**
$$\frac{\partial^2 T}{\partial x^2}\bigg|_i \approx \frac{T_{i+1} - 2T_i + T_{i-1}}{\Delta x^2} + O(\Delta x^2)$$

---

## 3. Laplaciano 3D (Mesh Cartesiana Uniforme)

Per una mesh con spaziatura uniforme $\Delta = \Delta x = \Delta y = \Delta z$:

$$\nabla^2 T_{i,j,k} \approx \frac{1}{\Delta^2}[T_{i+1,j,k} + T_{i-1,j,k} + T_{i,j+1,k} + T_{i,j-1,k} + T_{i,j,k+1} + T_{i,j,k-1} - 6T_{i,j,k}]$$

### 3.1 Forma Matriciale

Per ogni nodo interno, l'equazione stazionaria $\nabla^2 T = -Q/k$ diventa:

$$\frac{1}{\Delta^2}[T_{E} + T_{W} + T_{N} + T_{S} + T_{U} + T_{D} - 6T_{P}] = -\frac{Q_P}{k}$$

Dove:
- P = nodo Principale (i,j,k)
- E = East (i+1,j,k), W = West (i-1,j,k)
- N = North (i,j+1,k), S = South (i,j-1,k)  
- U = Up (i,j,k+1), D = Down (i,j,k-1)

Riarrangiando:

$$-6T_P + T_E + T_W + T_N + T_S + T_U + T_D = -\frac{Q_P \Delta^2}{k}$$

---

## 4. Materiali con Conducibilità Variabile

Quando $k$ non è costante (interfacce tra materiali diversi), si usa la **media armonica**
per la conducibilità all'interfaccia:

$$k_{i+1/2} = \frac{2 k_i k_{i+1}}{k_i + k_{i+1}}$$

### 4.1 Schema Conservativo

Per garantire la conservazione dell'energia:

$$\nabla \cdot (k \nabla T) \approx \frac{1}{\Delta^2}\left[k_{E}(T_E - T_P) + k_W(T_W - T_P) + ...\right]$$

Dove $k_E = k_{i+1/2,j,k}$, ecc.

L'equazione per il nodo P diventa:

$$(k_E + k_W + k_N + k_S + k_U + k_D) T_P - k_E T_E - k_W T_W - k_N T_N - k_S T_S - k_U T_U - k_D T_D = Q_P \Delta^2$$

---

## 5. Condizioni al Contorno

### 5.1 Dirichlet (Temperatura Fissa)

Se il nodo di bordo ha $T_{bordo} = T_{prescritta}$:

**Opzione 1:** Inserire direttamente nel sistema
- Riga corrispondente: $1 \cdot T_{bordo} = T_{prescritta}$
- Contributo spostato nel vettore RHS delle equazioni vicine

**Opzione 2:** Nodo fantasma
- Si crea un nodo virtuale oltre il bordo
- Si impone la condizione e si elimina il nodo fantasma

### 5.2 Neumann (Flusso Nullo - Adiabatico)

Per un bordo adiabatico su $x = x_{max}$:

$$\frac{\partial T}{\partial x}\bigg|_{bordo} = 0$$

Approssimazione: $T_{i+1} = T_i$ (nodo fantasma riflesso)

L'equazione del nodo di bordo usa il valore riflesso.

### 5.3 Robin (Convezione)

Per convezione sul bordo:

$$-k \frac{\partial T}{\partial n}\bigg|_{bordo} = h(T_{bordo} - T_\infty)$$

Approssimando la derivata:

$$-k \frac{T_{bordo} - T_{interno}}{\Delta/2} = h(T_{bordo} - T_\infty)$$

Risolvendo per $T_{bordo}$:

$$T_{bordo} = \frac{k \cdot T_{interno} + h \cdot \frac{\Delta}{2} \cdot T_\infty}{k + h \cdot \frac{\Delta}{2}}$$

**Schema matriciale:**
$$\left(1 + \frac{h \Delta}{2k}\right) T_{bordo} - T_{interno} = \frac{h \Delta}{2k} T_\infty$$

---

## 6. Schema Implicito per Transitorio

### 6.1 Metodo Backward Euler (Implicito)

$$\rho c_p \frac{T^{n+1} - T^n}{\Delta t} = k \nabla^2 T^{n+1} + Q$$

Riarrangiando:

$$T^{n+1} - \frac{k \Delta t}{\rho c_p} \nabla^2 T^{n+1} = T^n + \frac{Q \Delta t}{\rho c_p}$$

Definendo $\alpha = k/(\rho c_p)$ e $Fo = \alpha \Delta t / \Delta^2$:

$$T_P^{n+1}(1 + 6 Fo) - Fo(T_E^{n+1} + T_W^{n+1} + ...) = T_P^n + \frac{Q_P \Delta t}{\rho c_p}$$

**Vantaggi:**
- Incondizionatamente stabile
- Permette $\Delta t$ grandi

**Svantaggi:**
- Richiede soluzione di sistema lineare ad ogni step

### 6.2 Metodo Crank-Nicolson

$$\rho c_p \frac{T^{n+1} - T^n}{\Delta t} = \frac{k}{2}(\nabla^2 T^{n+1} + \nabla^2 T^n) + Q$$

**Vantaggi:**
- Accuratezza $O(\Delta t^2)$
- Incondizionatamente stabile

---

## 7. Struttura della Matrice

### 7.1 Sistema Lineare

Il sistema FDM può essere scritto come:

$$\mathbf{A} \cdot \mathbf{T} = \mathbf{b}$$

Dove:
- $\mathbf{A}$ = matrice dei coefficienti (sparsa)
- $\mathbf{T}$ = vettore delle temperature incognite
- $\mathbf{b}$ = vettore dei termini noti

### 7.2 Struttura della Matrice A

Per una griglia 3D con $N_x \times N_y \times N_z$ nodi:

- Dimensione: $N \times N$ dove $N = N_x \cdot N_y \cdot N_z$
- Struttura a bande con 7 diagonali (3D)
- Matrice sparsa (molti zeri)

**Pattern delle diagonali:**
- Diagonale principale: coefficiente del nodo P
- Diagonali ±1: coefficienti E/W
- Diagonali ±$N_x$: coefficienti N/S
- Diagonali ±$N_x N_y$: coefficienti U/D

### 7.3 Esempio di Riga della Matrice

Per un nodo interno con indice lineare $p$:

```
A[p, p-Nx*Ny] = -k_D / Δ²     (contributo Down)
A[p, p-Nx]    = -k_S / Δ²     (contributo South)
A[p, p-1]     = -k_W / Δ²     (contributo West)
A[p, p]       = (k_E+k_W+k_N+k_S+k_U+k_D) / Δ²  (diagonale)
A[p, p+1]     = -k_E / Δ²     (contributo East)
A[p, p+Nx]    = -k_N / Δ²     (contributo North)
A[p, p+Nx*Ny] = -k_U / Δ²     (contributo Up)

b[p] = Q_p
```

---

## 8. Linearizzazione degli Indici

### 8.1 Da 3D a 1D

Per passare da indici $(i, j, k)$ a indice lineare $p$:

$$p = i + j \cdot N_x + k \cdot N_x \cdot N_y$$

### 8.2 Da 1D a 3D

Per il passaggio inverso:

$$k = p // (N_x \cdot N_y)$$
$$j = (p - k \cdot N_x \cdot N_y) // N_x$$
$$i = p - k \cdot N_x \cdot N_y - j \cdot N_x$$

---

## 9. Implementazione Efficiente

### 9.1 Matrici Sparse

Usare `scipy.sparse` per matrici sparse:
- `csr_matrix`: Compressed Sparse Row - efficiente per moltiplicazione
- `csc_matrix`: Compressed Sparse Column - efficiente per slicing
- `lil_matrix`: List of Lists - efficiente per costruzione

### 9.2 Solutori

**Diretti:**
- `scipy.sparse.linalg.spsolve`: Fattorizzazione LU sparsa
- Robusto ma memoria O(N^1.5) per 3D

**Iterativi:**
- `scipy.sparse.linalg.cg`: Conjugate Gradient (matrice simmetrica positiva definita)
- `scipy.sparse.linalg.gmres`: General Minimal Residual
- `scipy.sparse.linalg.bicgstab`: BiCGStab

**Precondizionatori:**
- ILU (Incomplete LU)
- Jacobi
- SSOR

### 9.3 Accelerazione Numba

```python
from numba import jit, prange

@jit(nopython=True, parallel=True)
def build_matrix_coefficients(k_field, dx, Nx, Ny, Nz):
    # Calcolo parallelo dei coefficienti
    for k in prange(Nz):
        for j in range(Ny):
            for i in range(Nx):
                # ... calcolo coefficienti ...
```

---

## 10. Validazione

### 10.1 Soluzioni Analitiche

Per validare il codice, confrontare con soluzioni note:

**Caso 1D stazionario:**
$$T(x) = T_0 + \frac{q''}{k} x - \frac{Q}{2k} x^2$$

**Caso 3D con sorgente uniforme:**
Sfera in dominio infinito, soluzione analitica per confronto.

### 10.2 Bilancio Energetico

Verificare che:
$$P_{in} = P_{out} + \Delta E_{stored}$$

Con tolleranza < 1% per mesh sufficientemente fine.

---

## 11. Convergenza di Mesh

### 11.1 Studio di Convergenza

Eseguire la simulazione con mesh progressivamente più fini:
- Mesh 1: N = 20
- Mesh 2: N = 40
- Mesh 3: N = 80

Verificare che $||T_{N} - T_{N/2}||$ diminuisca con ordine atteso (secondo ordine per schema centrato).

### 11.2 Criterio di Arresto

Convergenza raggiunta quando:
$$\frac{||T_{N} - T_{N/2}||_\infty}{||T_{N}||_\infty} < \epsilon$$

Con $\epsilon$ tipicamente 0.01 (1%).
