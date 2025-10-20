import numpy as np

def verificar_condiciones(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> bool:
    
    m, n = A.shape
    if b.shape[0] != m:
        raise ValueError(f"La dimensión de b ({b.shape[0]}) no coincide con las filas de A ({m})")
    if x.shape[0] != n:
        raise ValueError(f"La dimensión de x ({x.shape[0]}) no coincide con las columnas de A ({n})")
    try:
        Ax = A @ x
        condicion_1_vec = (Ax <= b)
        condicion_1_satisfecha = np.all(condicion_1_vec)
        condicion_2_vec = (x >= 0)
        condicion_2_satisfecha = np.all(condicion_2_vec)
        return condicion_1_satisfecha and condicion_2_satisfecha

    except Exception as e:
        print(f"Error durante el cálculo: {e}")
        return False
