#import Shiv_Gauss
import random, Shiv_Gauss, numpy as np 

TOLERANCE = 0.1


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
/*		    Normalize
 * Desc: 
 * Calculates The Infinite Norm of a Vector
 * 		
 * Args:
 * @int		size		Size of Square Matrix
 * @float	VectorB  	Square Matrix
 *
 * Returns:
 * @float       Maxiumum absolute Value
 */

"""
def Normalize(size, Vector):
    temp = []
    tsum = 0
    # Iterate over rows to get Inifinite Norm
    for i in range(0,size):
        temp.append(abs(Vector[i]))
    return max(temp)
    
"""
/*		Inverse Iteration
 * Desc: 
 * Calculates the smallest positive Eigenvalue 
 * and Eigenvector for a given Matrix
 *
 * Prints each iteration to show convergence
 * Relies on Gauss_LU() and its dependencies
 * 		
 * Args:
 * @int		size		Size of Square Matrix
 * @float	MatrixA  	Square Matrix
 *
 * Returns:
 * @float       EigenVector (of size 'size')
 * @float       EigenValue
 */

"""
def InverseEigen(size, MatrixA):
    # Create a Random Vector of Size
    x = []
    for i in range(0,size):
        x.append(random.randrange(-5,5))
    print(f"Initial Vector: {x}\n Matrix A:\n {MatrixA}")

    # Store version of MatrixA Since Gauss Changes it
    MatrixA_tmp = np.empty_like(MatrixA)
    MatrixA_tmp[:] = MatrixA

    tol = TOLERANCE
    diff = TOLERANCE+1

    yn_cur = 0
    ynorm = Normalize(size, x)
    k = 0

    print("k\t|\t xT\t\t| ||Yk||inf")
    while diff > tol:

        yn_cur = ynorm
        # Solve for Ay = x
        y = Gauss_LU(size, MatrixA, x)
        
        # Generate the next Vector
        ynorm = Normalize(size, y)
        for i in range(0, size):
            x[i] = float(format(y[i] / ynorm, ".4f"))
        
        # Print Current Iteration
        print(f"{k}\t|\t {x} \t|\t {ynorm}")

        # Update difference and restore MatrixA since Gauss function updates it
        diff = ( abs(ynorm - yn_cur) / yn_cur ) * 100
        k += 1
        MatrixA[:] = MatrixA_tmp

    return x, ynorm


def main():
 # Get Inputs
    size = int(input("Enter Size of the Array:"))

    if size < 2:
        print("Invalid Size Provided! Must be atleast a 2x2 matrix!")
        quit()

    # Generate a Matrix of given size
    MatA = np.zeros((size,size))
    print("Getting Values for Matrix A...")
    for i in range(0,size):
        for j in range(0, size):
            MatA[i][j] = float(input(f"Provide Value for A[{i}][{j}]: "))

    # Get and Print the Results
    EigenVector, REigenValue = InverseEigen(size, MatA)
    print("EigenVector is ", EigenVector, " and EigenValue is ", 1/REigenValue)

   
if __name__ == "__main__":
    main()



