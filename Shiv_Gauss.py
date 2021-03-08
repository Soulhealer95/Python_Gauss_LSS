"""
/*		Gauss Elimination Using LU and Partial Pivoting
 * Uses Gauss Elimination to Solve System of Linear Equations
 * Relies on Numpy Module
 * 
 *
 * Author:					Dated				
 * Shivam S.					03-Feb-21
 * Student, McMasterU
 *
 */
"""

import numpy as np


"""
/*			Upper_TLS
 * Desc: 
 * Solve Upper Triangular Linear Systems Equation
 * 		Ax = b
 *
 * Args:
 * @int		size		Size of Square Matrix
 * @float	MatrixA  	Square Matrix (A)
 * @float	VectorB		Array with (b) Values
 *
 * Returns:
 * N/A
 */
"""
def Upper_TLS(size, MatrixA, VectorB):
    # Generate temp array Vector for x
    x = []
    for i in range(0, size):
        x.append(0)

    # Loop backwards to compute x
    for i in range(size-1,-1,-1):
        if MatrixA[i][i] == 0:
            print("Division by Zero! Exiting!")
            return
        x[i] = round(VectorB[i] / MatrixA[i][i], 3)

        for j in range(0, i+1):
            VectorB[j] = VectorB[j] - (MatrixA[j][i] * x[i])

    return x




"""
/*			Gauss_LU
 * Desc: 
 * Solves Linear System Equation with Gauss Elimination
 * Can not be used if first term of Matrix is 0
 *
 * Prints the Matrix and Result
 * Relies on Upper_TLS()
 * 		
 * Args:
 * @int		size		Size of Square Matrix
 * @float	MatrixA  	Square Matrix
 * @float	VectorB		Array with (b) Values
 *
 * Returns:
 * N/A
 */

"""


def Gauss_LU(size, MatrixA, VectorB):

    # Initializing Matrices
    MatrixL = np.zeros((size,size))

    if MatrixA[0][0] == 0:
        hnt("Gauss_LU(): LU Factorization Impossible since A[0][0] = 0!")
        return

    # Calculate
    for k in range(0, size):

        #Compute Multiplier
        for i in range(k+1, size):
            MatrixL[i][k] = MatrixA[i][k] / MatrixA[k][k]
            MatrixA[i][k] = 0
            

        # Perform Eliminations
        for j in range(k+1, size):
            for l in range(k+1, size):
                MatrixA[l][j] = MatrixA[l][j] - (MatrixL[l][k] * MatrixA[k][j])

            VectorB[j] = VectorB[j] - (MatrixL[j][k] * VectorB[k])

    # Format diagonal in MatrixL to "1"
    for i in range(0,size):
        MatrixL[i][i] = 1

    return Upper_TLS(size, MatrixA, VectorB)



"""
/*			Gauss_PP
 * Desc: 
 * Solves Linear System Equation with Gauss Elimination
 * Using Partial Pivoting and Upper_TLS function
 * Prints the result
 * 		
 * Args:
 * @int		size		Size of Square Matrix
 * @float	MatrixA  	Square Matrix
 * @float	VectorB		Array with (b) Values
 *
 * Returns:
 * N/A
 */

"""
def Gauss_PP(size, MatrixA, VectorB):

    # Initializing Matrices
    MatrixL = np.zeros((size,size))
    
    # Calculate
    for k in range(0, size):

        # Assume current row is highest
        high_row = k

        # Find biggest value in the current column
        for j in range(k+1, size):
            if abs(MatrixA[high_row][k]) < abs(MatrixA[j][k]):
                high_row = j

        # If a row with higher value was found, swap rows
        if high_row != k:
            MatrixA[[ k, high_row ]] = MatrixA[[ high_row, k ]]
            MatrixL[[ k, high_row ]] = MatrixL[[ high_row, k ]]
            VectorB[k], VectorB[high_row] = VectorB[high_row], VectorB[k]
            print(f"Gauss_PP(): Row Swap {k} and {high_row} Completed!\n A:\n{MatrixA}\n b:{VectorB}T\n L:\n{MatrixL}\n")

        # skip if current column is 0
        if MatrixA[k][k] == 0:
            continue

        #Compute Multiplier
        for i in range(k+1, size):
            MatrixL[i][k] = MatrixA[i][k] / MatrixA[k][k]
            MatrixA[i][k] = 0
            

        # Perform Eliminations
        for j in range(k+1, size):
            for l in range(k+1, size):
                MatrixA[l][j] = MatrixA[l][j] - (MatrixL[l][k] * MatrixA[k][j])
                print(f"Gauss_PP(): Elimination at {j}-{k} A:\n{MatrixA}\n")

            VectorB[j] = VectorB[j] - (MatrixL[j][k] * VectorB[k])
            print(f"Gauss_PP(): Elimination at {j} Vector B = {VectorB}T\n")

    # Format diagonal in MatrixL to "1"
    for i in range(0,size):
        MatrixL[i][i] = 1

    print(f"Gauss_PP(): Pre-Results\n A:\n {MatrixA}\n b:\n {VectorB}T\n L:\n {MatrixL}\n")
    Upper_TLS(size, MatrixA, VectorB)





