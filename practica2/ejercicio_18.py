import sys
import mosek

def solve_sdp_primal(objective_sense):
    
    numvar = 1 
    numcon = 6  
    BARVARDIM = [3] ## Matrices 3x3

    ##Defino los límites de las restricciones (b_i)
    bkc = [mosek.boundkey.fx] * numcon
    blc = [5.0, -12.0, 27.0, 0.0, 1.0, 10.0]
    buc = [5.0, -12.0, 27.0, 0.0, 1.0, 10.0]

    with mosek.Env() as env:
        with env.Task(0, 0) as task:
            task.appendvars(numvar)
            task.appendcons(numcon)
            task.appendbarvars(BARVARDIM)

            
            ## 'y' es una variable libre
            task.putvarbound(0, mosek.boundkey.fr, 0.0, 0.0)
            task.putcj(0, 1.0)
            if objective_sense == 'minimize':
                task.putobjsense(mosek.objsense.minimize)
            else:
                task.putobjsense(mosek.objsense.maximize)

            task.putarow(2, [0], [2.0])
            task.putarow(3, [0], [-1.0])

            ## Creo las matrices esparsas A_i que "seleccionan" un elemento de X
            
            # i=0: X_00 = 5. A_0 tiene 1.0 en (0,0)
            symA0 = task.appendsparsesymmat(BARVARDIM[0], [0], [0], [1.0])
            
            # i=1: X_10 = -12. A_1 tiene 0.5 en (1,0)
            symA1 = task.appendsparsesymmat(BARVARDIM[0], [1], [0], [0.5])
            
            # i=2: X_11 + ... = 27. A_2 tiene 1.0 en (1,1)
            symA2 = task.appendsparsesymmat(BARVARDIM[0], [1], [1], [1.0])
            
            # i=3: X_20 + ... = 0. A_3 tiene 0.5 en (2,0)
            symA3 = task.appendsparsesymmat(BARVARDIM[0], [2], [0], [0.5])
            
            # i=4: X_21 = 1. A_4 tiene 0.5 en (2,1)
            symA4 = task.appendsparsesymmat(BARVARDIM[0], [2], [1], [0.5])
            
            # i=5: X_22 = 10. A_5 tiene 1.0 en (2,2)
            symA5 = task.appendsparsesymmat(BARVARDIM[0], [2], [2], [1.0])

            task.putbaraij(0, 0, [symA0], [1.0]) # Constr 0, Var X_0
            task.putbaraij(1, 0, [symA1], [1.0]) # Constr 1, Var X_0
            task.putbaraij(2, 0, [symA2], [1.0]) # Constr 2, Var X_0
            task.putbaraij(3, 0, [symA3], [1.0]) # Constr 3, Var X_0
            task.putbaraij(4, 0, [symA4], [1.0]) # Constr 4, Var X_0
            task.putbaraij(5, 0, [symA5], [1.0]) # Constr 5, Var X_0

            for i in range(numcon):
                task.putconbound(i, bkc[i], blc[i], buc[i])
            task.optimize()
            solsta = task.getsolsta(mosek.soltype.itr)

            if solsta == mosek.solsta.optimal:
                xx = task.getxx(mosek.soltype.itr)
                return xx[0]
            else:
                return None
if __name__ == "__main__":
    y_min = solve_sdp_primal('minimize')
    y_max = solve_sdp_primal('maximize')

    print(f"Resolviendo 'minimizar: y'...")
    if y_min is not None:
        print(f"Resultado (y_min): {y_min:.8f}")
    else:
        print("El problema de minimización no tiene solución óptima.")

    print(f"\nResolviendo 'maximizar: y'...")
    if y_max is not None:
        print(f"Resultado (y_max): {y_max:.8f}")
    else:
        print("El problema de maximización no tiene solución óptima.")

    if y_min is not None and y_max is not None:
        print("\n---")
        print("Determinación del espectraedro (conjunto factible para y):")
        print(f"El conjunto factible para 'y' es el intervalo: [{y_min:.8f}, {y_max:.8f}]")