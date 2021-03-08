#import Shiv_Gauss
import random, Shiv_Gauss, numpy as np 

TOLERANCE = 0.01


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
            return -1
        x[i] = round(VectorB[i] / MatrixA[i][i], 3)

        for j in range(0, i+1):
            VectorB[j] = VectorB[j] - (MatrixA[j][i] * x[i])

    return x


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

            VectorB[j] = VectorB[j] - (MatrixL[j][k] * VectorB[k])

    # Format diagonal in MatrixL to "1"
    for i in range(0,size):
        MatrixL[i][i] = 1

    return Upper_TLS(size, MatrixA, VectorB)



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
        return Gauss_PP(size, MatrixA, VectorB)

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
/*		    Norm
 * Desc: 
 * Calculates The Infinite Norm of a Vector
 * 		
 * Args:
 * @int		size		Size of Square Matrix
 * @float	VectorB  	Square Matrix
 *
 * Returns:
 * @float       Result of the Formula
 */

"""

def Norm(size, Vector):
    #print (f"Normalizing: {Vector}")
    temp = []
    tsum = 0
    # Iterate over rows to get Inifinite Norm
    for i in range(0,size):
        temp.append(abs(Vector[i]))
    return max(temp)

"""
/*		Sigma-Calc
 * Desc: 
 * Calculates The Sigma Value from a Matrix and Vector 
 * Uses formula:
 *          Sigma = ( x(t) * A * x ) / ( x(t) * x ) 
 *          Where x     -> Vector 
 *                x(t)  -> Transpose of The Vector
 *                A     -> Matrix
 *
 * Relies on numpy
 * 		
 * Args:
 * @int		size		Size of Square Matrix
 * @float	MatrixA  	Square Matrix
 *
 * Returns:
 * @float       Result of the Formula
 */

"""
def sigma1_calc(MatrixA, VectorB):
    # Calculate sigma1 = xT * A * x / xT * x
    BT = np.transpose(VectorB)
    num =  np.matmul(BT, np.matmul(MatrixA, VectorB))
    denom = np.matmul(BT, VectorB)
    #print(f"sigma with : {num} and {denom}")

    return num /denom


"""
/*		Rayleigh-Quotient Iteration
 * Desc: 
 * Calculates Rayleigh-Quotient and uses that to find 
 * Eigenvalue and Eigenvector
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
def Ray(size, MatrixA):
    # Create a Random Vector of Size
    x = []
    for i in range(0,size):
        x.append(random.randrange(-5,5))
    print(f"Initial Vector: {x}\n Matrix A:\n {MatrixA}")

    tol = TOLERANCE
    diff = TOLERANCE+1

    sigma1 = sigma1_calc(MatrixA, x)
    ynorm = Norm(size, x)
    k = 0

    # Generate Identity Matrix I
    I = np.zeros((size,size))
    for i in range(0,size):
        I[i][i] = 1

    print("k\t|\t\t\txT\t\t\t\t\t\t\t|\t Yinf \t\t\t|\tsigma1")
    while diff > tol:

        sigma1_cur = sigma1

        # Shift the MatrixA with sigma 
        sub_arr = sigma1 * np.array(I)
        MatrixA_new = np.subtract( MatrixA, sub_arr )

        # Solve for Ay = x
        y = Gauss_LU(size, MatrixA_new, x)

        # If it was a singular matrix Exit now
        if y == -1:
            break
        
        # Generate the next Vector
        ynorm = Norm(size, y)
        for i in range(0, size):
            x[i] = float(format(y[i] / ynorm, ".4f"))
        
        # Print Current Iteration
        print(f"{k}\t|\t\t\t{x}\t\t\t\t\t|\t {ynorm:.4f} \t\t|\t{sigma1}")

        # Update iteration
        k += 1
        # Calculate sigma1 = xT * A * x / xT * x
        sigma1 = sigma1_calc(MatrixA, x)
        diff = abs( (sigma1 - sigma1_cur) / sigma1_cur ) * 100

    return x, sigma1





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

    # Get the EigenValue
    EigenVector, EigenValue = Ray(size, MatA)
    print("EigenVector is ", EigenVector, " and EigenValue is ", round(EigenValue,4))

   
if __name__ == "__main__":
    main()



