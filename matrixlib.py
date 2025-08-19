import copy
import math
import cmath

class Matrix:
    """
    Attributes:
        data: 2 dimensional array of integers/floats/complex.
        dim: tuple encoding dimensions of matrix.
    
    Methods:
        transpose()
            Returns transpose of a matrix.
    """
    def __init__(self, data):
        if type(data) != list:
            raise TypeError("Matrix must be a list of lists with same length")
        if data == []:
            raise TypeError("Matrix must be a list of lists with same length")
        
        m = len(data)       #m is the number of rows
        n = len(data[0])    #n is the number of columns

        for row in data:    #if some row has not the same length as others do
            if len(row) != n:
                   raise TypeError("Matrix must be a list of lists with same length")

        self.data = data
        self.dim = (m, n)

    def __repr__(self):             #outputs each row of a matrix on separate line
        s = ""        
        for row in self.data:
            s += str(row) + "\n"
        s = s[:-1]
        return s

    def __eq__(self, other):
        if self.data != other.data:
            return False
        return True

    def __add__(self, other):       #matrix addition
        if self.dim == other.dim:
            M = []
            for i in range(self.dim[0]):
                M.append(list(map(sum, zip(self.data[i], other.data[i])))) #sums i-th row of self with i-th row of other
            return Matrix(M)
        else:
            raise ValueError("Dimensions of matrices are not the same")

    def __mul__(self, other):
        if (type(self) == Matrix) and ((type(other) == int) or (type(other) == float) or (type(other) == complex)): #scalar multiplication
            return Matrix([[x*other for x in row] for row in self.data])
        if (type(self) == Matrix) and (type(other) == Matrix):   #standard matrix multiplication algorithm
            if self.dim[1] != other.dim[0]:
                raise ValueError("Dimensions of matrices not compatible")
            M = Empty(self.dim[0], other.dim[1])
            for i in range(self.dim[0]):               
                for j in range(other.dim[1]):
                    for k in range(other.dim[0]):
                        M.data[i][j] += self.data[i][k]*other.data[k][j]
            return M
                          
        else:
            raise TypeError("Unsupported type for multiplication")

    def __sub__(self, other):     #matrix subtraction  
        if self.dim == other.dim:
            return self + other*(-1)
        else:
            raise ValueError("Dimensions of matrices are not the same")

    def transpose(self):
        """ Returns transpose of a matrix."""
        T = Matrix([[self.data[i][j] for i in range(self.dim[0])] for j in range(self.dim[1])])
        self.data = T.data
        self.dim = (T.dim[0], T.dim[1])
        return T

class Sparse:
    """
    Attributes:
        data: array of arrays with 3 elements [row, column, value], value can be integer/float/complex.
        dim: tuple encoding dimensions of matrix.
    
    Methods:
        transpose()
            Returns transpose of a matrix.
    """
    def __init__(self, data, dim):
        if (data != []) and ((max([x[0] for x in data]) >= dim[0]) or (max([x[1] for x in data]) >= dim[1]) or (min([x[1] for x in data]) < 0) or (min([x[0] for x in data]) < 0)):
            raise ValueError("Element indeces are not compatible with matrix dimensions")

        self.data = data    #list of 3 element lists [row, column, value]
        self.dim = dim      #user needs to specify dimensions of matrix

    def __repr__(self):
        A = Empty(self.dim[0], self.dim[1])

        for x in self.data:
            A.data[x[0]][x[1]] = x[2] 
        
        s = ""        
        for row in A.data:
            s += str(row) + "\n"
        s = s[:-1]
        return s

    def __eq__(self, other):
        if (self.data != other.data) or (self.dim != other.dim):
            return False
        return True        

    def __add__(self, other):
        if self.dim == other.dim:
            A = Sparse([], self.dim)
            B = Sparse([], other.dim)
            M = Sparse([], self.dim)
            A.data = self.data.copy()
            B.data = other.data.copy()

            i = 0
            j = 0
            while (i < len(A.data)) and (j < len(B.data)):
                if (A.data[i][0], A.data[i][1]) < (B.data[j][0], B.data[j][1]):
                    M.data.append(A.data[i])
                    i += 1

                elif (A.data[i][0], A.data[i][1]) > (B.data[j][0], B.data[j][1]):
                    M.data.append(B.data[j])
                    j += 1

                elif (A.data[i][0], A.data[i][1]) == (B.data[j][0], B.data[j][1]):
                    M.data.append([A.data[i][0], A.data[i][1], A.data[i][2] + B.data[j][2]])
                    i += 1
                    j += 1

            return M

        else:
            raise ValueError("Matrices are not of a same type")

    def __mul__(self, other):
        if (type(self) == Sparse) and ((type(other) == int) or (type(other) == float)): #multiplying by scalar
            A = Sparse([], self.dim)
            for x in self.data:
                A.data.append([x[0], x[1], x[2]*other])
            return A
        
        if (type(self) == Sparse) and (type(other) == Sparse):  #matrix multiplication
            if (self.dim[1] == other.dim[0]):
                A = Empty(self.dim[0], self.dim[1])
                B = Empty(other.dim[0], other.dim[1])
                for x in self.data:
                    A.data[x[0]][x[1]] = x[2]
                for x in other.data:
                    B.data[x[0]][x[1]] = x[2]
                
                return A * B
            else:
                raise ValueError("Dimensions of matrices not compatible")
        else:
            raise TypeError("Unsupported type for multiplication") 

    def __sub__(self, other):     #matrix subtraction  
        if self.dim == other.dim:
            return self + other*(-1)
        else:
            raise ValueError("Dimensions of matrices are not the same")          

    def transpose(self):
        """ Returns transpose of a sparse matrix."""
        T = Sparse([[x[1], x[0], x[2]] for x in self.data], (self.dim[1], self.dim[0]))
        self.data = T.data
        self.dim = T.dim
        return T                            

#------------------Functions-------------------------#

def Identity(n: int):
    """
    Returns n x n identity matrix.

    Examples:
    >>> print(Identity(3))
    [1, 0, 0]
    [0, 1, 0]
    [0, 0, 1]
    """
    if n > 0:
        return Matrix([[int(i == j) for j in range(n)] for i in range(n)])  
    else:
        raise ValueError("The order of matrix has to greater than zero")  

def Empty(m: int, n: int):
    """
    Returns m x n matrix filled with zeros.

    Examples:
    >>> print(Empty(3, 2))
    [0, 0]
    [0, 0]
    [0, 0]
    """
    return Matrix([[0 for j in range(n)] for i in range(m)])

def dot(a: Matrix, b: Matrix): 
    """
    Returns dot product of vectors a and b

    Examples:
    >>> a = Matrix([[1], [2], [3]])
    >>> b = Matrix([[4], [5], [6]])
    >>> print(dot(a, b))
    32
    """
    if (a.dim[1] != 1) or (b.dim[1] != 1):
        raise ValueError("dot function is defined for vectors only")
    c = Empty(a.dim[0], 1)
    c.data = a.data.copy()
    c.transpose()
    return (c*b).data[0][0]

def hadamard_product(A: Matrix, B: Matrix):
    """
    Returns the hadamard product of matrices A and B

    Examples:
    >>> print(hadamard_product(Matrix([[1, 3, 4], [5, 4, 2], [7, 5, 3]]), Matrix([[4, 5, 2], [9, 7, 1], [2, 2, 2]])))
    [4, 15, 8]
    [45, 28, 2]
    [14, 10, 6]
    """
    if (A.dim[0] == B.dim[0]) and (A.dim[1] == B.dim[1]):
        C = Empty(A.dim[0], A.dim[1])
        C.data = [[A.data[i][j]*B.data[i][j] for j in range(A.dim[1])] for i in range(A.dim[0])]
        return C
    else:
        raise ValueError("Matrices must be of same type")

def trace(A: Matrix):
    """
    Returns trace of a square matrix A

    Examples:
    >>> print(trace(Matrix([[1, 4, 7], [2, -1, 3], [3, 4, 0]])))
    0
    """
    if (A.dim[0] != A.dim[1]):
        raise ValueError("Matrix is not square")
    return sum([A.data[i][i] for i in range(A.dim[0])])

def LUP(A: Matrix):     
    """
    Returns LUP decomposition of square matrix + number of row exchanges
    
    Examples:
    >>> L, U, P, row_ex = LUP(Matrix([[1, 4, -3], [-2, 8, 5], [3, 4, 7]]))
    >>> print(L)
    [1, 0, 0]
    [-2.0, 1, 0]
    [3.0, -0.5, 1]
    >>> print(U)
    [1, 4, -3]
    [0.0, 16.0, -1.0]
    [0.0, 0.0, 15.5]
    >>> print(P)
    [1, 0, 0]
    [0, 1, 0]
    [0, 0, 1]
    >>> print(row_ex)
    0
    """    
    if A.dim[0] != A.dim[1]:
        raise ValueError("Matrix is not square")

    L = Identity(A.dim[0])      #upper triangular matrix
    U = Identity(A.dim[0])      #lower triangular matrix
    P = Identity(A.dim[0])      #permutation matrix
    row_ex = 0                  #number of row exchanges, useful for computing determinant

    for i in range(A.dim[0]):
        zero_column = False
        if A.data[i][i] == 0:   #if the diagonal elements is zero we need to swap it with other non-zero elements in the same column
            for j in range(i, A.dim[0]):
                if A.data[j][i] != 0:
                    A.data[i], A.data[j] = A.data[j], A.data[i]
                    P.data[i], P.data[j] = P.data[j], P.data[i]
                    row_ex += 1
                    break
            else:
                zero_column = True
        if zero_column != True: #if the current column is not zero, then we do standard gaussian elimination
            for j in range(i+1, A.dim[0]):
                L.data[j][i] = A.data[j][i]/A.data[i][i]
                A.data[j] = [A.data[j][k]-(A.data[j][i]/A.data[i][i])*A.data[i][k] for k in range(A.dim[0])]    #each row from index i+1, ..., j is updated such that A[j][i] = 0
    
    U = A 
    return L, U, P, row_ex       

def det(A: Matrix): 
    """
    Returns the determinant of the matrix A
    
    Examples:
    >>> print(det(Matrix([[1, 2, 3], [4, 5, 6], [7, 3, 1]])))
    -6.0
    """
    if A.dim[0] != A.dim[1]: #if the matrix is not of type n x n
        raise ValueError("Matrix is not square")
    L, U, P, row_ex = LUP(A)    #LUP decomposition of matrix
    D = 1
    for i in range(A.dim[0]):   #fist we multiply D with diagonals of L and U
        D *= L.data[i][i]*U.data[i][i]
    return D * (-1)**row_ex     #then we multiply D by the determinant of the permutation matrix

def inv(A: Matrix): 
    """
    Returns the inverse of a regular matrix A

    Examples:
    >>> print(inv(Matrix([[1, 0, 5], [2, 1, 6], [3, 4, 0]])))
    [-24.0, 20.0, -5.0]
    [18.0, -15.0, 4.0]
    [5.0, -4.0, 1.0]
    """
    if A.dim[0] != A.dim[1]: #if the matrix is not of type n x n
        raise ValueError("Matrix is not square")
    A_inv = Empty(A.dim[0], A.dim[1])
    L, U, P, row_ex = LUP(A)    #LUP decomposition of matrix
    for i in range(U.dim[0]):
        if U.data[i][i] == 0:   #if there is a zero on the diagonal of the upper triangular matrix, then
            raise ValueError("Matrix is singular")
    x = [0] * A.dim[0]
    y = [0] * A.dim[0]    
    for i in range(A.dim[0]):
        b = [P.data[k][i] for k in range(P.dim[0])]  #this is the canonical vector e_i multiplied by permutation matrix P (Pe_i)
        for j in range(A.dim[0]):   #solving Ly = b
            y[j] = b[j] - sum(L.data[j][k] * y[k] for k in range(j))
        for j in range(A.dim[0]-1, -1, -1): #solving Ux = y
            x[j] = (y[j] - sum(U.data[j][k] * x[k] for k in range(j+1, A.dim[0]))) / U.data[j][j]
            A_inv.data[j][i] = x[j]
    return A_inv

def solve(A: Matrix, b: Matrix): 
    """
    Returns the solutions to the equation Ax = b

    Examples:
    >>> print(solve(Matrix([[1, 2, 3], [3, 2, 1], [3, 1, 2]]), Matrix([[14], [10], [11]])))
    [1.0]
    [2.0]
    [3.0]
    """    
    if A.dim[0] != A.dim[1]: #if the matrix is not of type n x n
        raise ValueError("Matrix is not square")
    if (b.dim[0] != A.dim[0]) or (b.dim[1] != 1):
        raise ValueError("Vector b is of a wrong type")
    L, U, P, row_ex = LUP(A)    #LUP decomposition of matrix
    for i in range(U.dim[0]):
        if U.data[i][i] == 0:   #if there is a zero on the diagonal of the upper triangular matrix, then
            raise ValueError("Matrix is singular")
    x = [0] * A.dim[0]
    y = [0] * A.dim[0]
    z = P*b
    for i in range(A.dim[0]):   #solving Ly = z
        y[i] = z.data[i][0] - sum(L.data[i][j] * y[j] for j in range(i))
    for i in range(A.dim[0]-1, -1, -1): #solving Ux = y
        x[i] = (y[i] - sum(U.data[i][j] * x[j] for j in range(i+1, A.dim[0]))) / U.data[i][i]
    x = Matrix([x])     #
    x.transpose()       #transformes x into a n x 1 matrix
    return x

def REF(A: Matrix):     
    """
    Returns row echelon form of a matrix
    
    Examples:
    >>> print(REF(Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))
    [1, 2, 3]
    [0.0, -3.0, -6.0]
    [0.0, 0.0, 0.0]
    """
    for i in range(min(A.dim[0], A.dim[1])):
        zero_column = False
        if A.data[i][i] == 0:   #if the diagonal element is zero we need to swap it with other non-zero elements in the same column
            for j in range(i, A.dim[0]):
                if A.data[j][i] != 0:
                    A.data[i], A.data[j] = A.data[j], A.data[i]
                    break
            else:
                zero_column = True
        if zero_column != True: #if the current column is not zero, then we do standard gaussian elimination
            for j in range(i+1, A.dim[0]):
                A.data[j] = [A.data[j][k]-(A.data[j][i]/A.data[i][i])*A.data[i][k] for k in range(A.dim[1])]    #each row from index i+1, ..., j is updated such that A[j][i] = 0
    return A

def cholesky(A: Matrix): 
    """
    Returns cholesky decomposition of a positive definite matrix (if it exists)

    Examples:
    >>> print(cholesky(Matrix([[9, 6, 3], [6, 8, 2], [3, 2, 5]])))
    [3.0, 0, 0]
    [2.0, 2.0, 0]
    [1.0, 0.0, 2.0]
    """
    if A.dim[0] != A.dim[1]: #if the matrix is not of type n x n
        raise ValueError("Matrix is not square")
    B = Empty(A.dim[0], A.dim[1])
    B.data = A.data.copy()
    B.transpose()
    if A != B:
        raise ValueError("Matrix is not symmetric")

    L = Empty(A.dim[0], A.dim[1]) #lower triangular matrix

    for i in range(A.dim[0]):
        for j in range(i+1):
            S = 0
            if i == j:  #calculating L[i][i]
                for k in range(j):
                    S += L.data[j][k]*L.data[j][k]
                if (A.data[i][i] - S > 0):
                    L.data[i][i] = (A.data[i][i] - S)**0.5
                else:
                    raise ValueError("Matrix is not positive definite")
            else:       #calculating L[i][j]
                for k in range(j):
                    S += L.data[i][k] * L.data[j][k]
                L.data[i][j] = (A.data[i][j] - S) / L.data[j][j]
    return L

def qr(A: Matrix): 
    """
    Returns QR decomposition of a square matrix

    Examples:
    >>> Q, R = qr(Matrix([[1,0,-1],[-2,1,4], [1, 3, 3]]))
    >>> print(Q)
    [0.4082482904638631, -0.8164965809277261, 0.4082482904638631]
    [-0.05314940034527339, 0.42519520276218714, 0.9035398058696477]
    [0.9113223768657689, 0.39056673294246624, -0.1301889109808269]
    >>> print(R)
    [2.449489742783178, 0.408248290463863, -2.4494897427831788]
    [0, 3.13581462037113, 4.464549629002965]
    [0, 0, 0.26037782196164777]
    """
    if A.dim[0] != A.dim[1]: #if the matrix is not of type n x n
        raise ValueError("Matrix is not square")
    Q = Empty(A.dim[0], A.dim[1])
    R = Empty(A.dim[0], A.dim[1])
    for i in range(A.dim[0]):
        u = Matrix([[A.data[k][i]] for k in range(A.dim[0])]) #i-th column vector of A
        for j in range(i):
            v = Matrix([[Q.data[k][j]] for k in range(Q.dim[0])]) #j-th column vector of Q
            R.data[j][i] = dot(Matrix([[A.data[k][i]] for k in range(A.dim[0])]), v)
            u = u - v * R.data[j][i]
        norm = (dot(u, u))**0.5 
        if norm < 1e-10:    #checks if u is zero
            for j in range(Q.dim[0]):
                Q.data[j][i] = 0
            R.data[i][i] = 0
        else:
            for j in range(Q.dim[0]):
                Q.data[j][i] = (u * (1/norm)).data[j][0]   
            R.data[i][i] = norm
    Q.transpose()
    return Q, R

def fast_mul(A: Matrix, B: Matrix): 
    """
    Uses Strassen algorithm for multiplying matrices A and B
    
    Examples:
    >>> print(fast_mul(Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), Matrix([[2, 3, 1], [7, 4, 2], [2, 2, 9]])))
    [22, 17, 32]
    [55, 44, 68]
    [88, 71, 104]
    """    
    if (A.dim[1] != B.dim[0]):
        raise ValueError("Matrix dimensions not compatible")

    c = max(A.dim[0], A.dim[1], B.dim[0], B.dim[1])
    k = 1
    while k < c:    #finding the smallest n such that c <= 2**n
        k *= 2

    A_ = Empty(k, k)
    B_ = Empty(k, k)

    for i in range(k):  #making A_, B_ square matrices of order k = 2**n
        for j in range(k):
            if (i < A.dim[0]) and (j < A.dim[1]):
                A_.data[i][j] = A.data[i][j]
            else:
                A_.data[i][j] = 0
            if (i < B.dim[0]) and (j < B.dim[1]):
                B_.data[i][j] = B.data[i][j]
            else:
                B_.data[i][j] = 0

    def recmul(A_: Matrix, B_: Matrix):
        if (A_.dim[0] == 1) and (B_.dim[0] == 1) and (A_.dim[1] == 1) and (B_.dim[1] == 1): #base case, 1x1 matrix multiplication
            return A_*B_               

        k = A_.dim[0]
        
        Ac = Empty(k, k)
        Bc = Empty(k, k)
        Ac.data = A_.data.copy()
        Bc.data = B_.data.copy()

        A11 = Matrix([[Ac.data[i][j] for j in range(k//2)] for i in range(k//2)])   #separating matrices into blocks
        A12 = Matrix([[Ac.data[i][j] for j in range(k//2, k)] for i in range(k//2)])
        A21 = Matrix([[Ac.data[i][j] for j in range(k//2)] for i in range(k//2, k)])
        A22 = Matrix([[Ac.data[i][j] for j in range(k//2, k)] for i in range(k//2, k)])

        B11 = Matrix([[Bc.data[i][j] for j in range(k//2)] for i in range(k//2)])
        B12 = Matrix([[Bc.data[i][j] for j in range(k//2, k)] for i in range(k//2)])
        B21 = Matrix([[Bc.data[i][j] for j in range(k//2)] for i in range(k//2, k)])
        B22 = Matrix([[Bc.data[i][j] for j in range(k//2, k)] for i in range(k//2, k)])

        M1 = recmul(A11 + A22, B11 + B22)
        M2 = recmul(A21 + A22, B11)     
        M3 = recmul(A11, B12 - B22)
        M4 = recmul(A22, B21 - B11)
        M5 = recmul(A11 + A12, B22)
        M6 = recmul(A21 - A11, B11 + B12)
        M7 = recmul(A12 - A22, B21 + B22)

        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6 

        C = Empty(k, k)
        for i in range(k):
            if (i < k//2):
                C.data[i] = C11.data[i] + C12.data[i]
            else:
                C.data[i] = C21.data[i-k//2] + C22.data[i-k//2]
        return C

    C_ = recmul(A_, B_)
    C = Matrix([[C_.data[i][j] for j in range(B.dim[1])] for i in range(A.dim[0])]) #the product of m x n and n x l matrix has dimensions m x l
    return C

def fast_inv(A: Matrix): 
    """
    Returns inverse of a regular matrix A with dimensions 2**K x 2**K

    Examples:
    >>> print(fast_inv(Matrix([[2, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]])))
    [0.5, 0.0, -0.5, 0.5]
    [0.0, 0.0, 1.0, -1.0]
    [-0.5, 1.0, -0.5, 0.5]
    [0.5, -1.0, 0.5, 0.5]
    """
    if (A.dim[0] == 1) and (A.dim[1] == 1): #base case
        return Matrix([[1/A.data[0][0]]])

    if (A.dim[0] & (A.dim[0]-1) != 0) or (A.dim[0] == 0) or (A.dim[1] & (A.dim[1]-1) != 0) or (A.dim[1] == 0):
        raise ValueError("Dimensions of matrix are not a power of 2")

    k = A.dim[0]
    A11 = Matrix([[A.data[i][j] for j in range(k//2)] for i in range(k//2)])   #separating matrices into blocks
    A12 = Matrix([[A.data[i][j] for j in range(k//2, k)] for i in range(k//2)])
    A21 = Matrix([[A.data[i][j] for j in range(k//2)] for i in range(k//2, k)])
    A22 = Matrix([[A.data[i][j] for j in range(k//2, k)] for i in range(k//2, k)])
      
    M_A = A22 - fast_mul(fast_mul(A21, fast_inv(A11)), A12)

    C11 = fast_inv(A11) + fast_mul(fast_mul(fast_mul(fast_mul(fast_inv(A11), A12), fast_inv(M_A)), A21), fast_inv(A11))
    C12 = fast_mul(fast_mul(fast_inv(A11), A12), fast_inv(M_A))*(-1)
    C21 = fast_mul(fast_mul(fast_inv(M_A), A21), fast_inv(A11))*(-1)
    C22 = fast_inv(M_A)

    C = Empty(k, k)
    for i in range(k):
        if (i < k//2):
            C.data[i] = C11.data[i] + C12.data[i]
        else:
            C.data[i] = C21.data[i-k//2] + C22.data[i-k//2]
    return C

def fast_LUP(A: Matrix): 
    """
    Returns LUP decomposition if a matrix is of a type 2**K x n and regular

    Examples:
    >>> L, U, P = fast_LUP(Matrix([[3, -7, -2, 2], [-3, 5, 1, 0], [6, -4, 0, -5], [-9, 5, -5, 12]]))
    >>> print(L)
    [1, 0, 0, 0]
    [-1.0, 1, 0, 0]
    [2.0, -4.999999999999999, 1, 0]
    [-3.0, 7.999999999999999, 3.000000000000006, 1]
    >>> print(U)
    [3, -7, -2.0, 2]
    [0, -2.0, -1.0, 2.0]
    [0, 0, -0.9999999999999982, 0.9999999999999982]
    [0, 0, 0, -0.9999999999999991]
    >>> print(P)
    [1, 0, 0, 0]
    [0, 1, 0, 0]
    [0, 0, 1, 0]
    [0, 0, 0, 1]
    """    
    if A.dim[0] > A.dim[1]:
        raise ValueError("Matrix is singular")

    if A.dim[0] == 1:   #Matrix with single row
        i = 0
        while (i < A.dim[1]) and (A.data[0][i] == 0):
            i += 1

        if i < A.dim[1]:    #we switch the first element with non-zero element and return LUP decomposition
            L = Identity(1)
            P = Identity(A.dim[1])
            P.data[0], P.data[i] = P.data[i], P.data[0]

            P_inv = Empty(P.dim[0], P.dim[1])
            P_inv.data = P.data.copy()
            P_inv.transpose()
            U = fast_mul(A, P_inv)
            return L, U, P
        
        else:
            raise ValueError("Matrix is singular")
    
    if (A.dim[0] & (A.dim[0]-1) != 0) or A.dim[0] == 0:    #checks of number of rows is a power of 2
        raise ValueError("Number of rows is not a power of 2")

    k = A.dim[0]

    B = Empty(k//2, A.dim[1])
    C = Empty(k//2, A.dim[1])

    B.data = [A.data[i] for i in range(k//2)]
    C.data = [A.data[i] for i in range(k//2, k)]
    L1, U1, P1 = fast_LUP(B)

    P1_inv = Empty(P1.dim[0], P1.dim[1])
    P1_inv.data = P1.data.copy()
    P1_inv.transpose()

    D = fast_mul(C, P1_inv)
    E = G = Empty(k//2, k//2)
    
    E = Matrix([[U1.data[i][j] for j in range(k//2)] for i in range(k//2)])   #separating matrices into blocks
    F = Matrix([[U1.data[i][j] for j in range(k//2, U1.dim[1])] for i in range(k//2)]) 
    G = Matrix([[D.data[i][j] for j in range(k//2)] for i in range(k//2)]) 
    H = Matrix([[D.data[i][j] for j in range(k//2, D.dim[1])] for i in range(k//2)])

    J = H - fast_mul(fast_mul(G, fast_inv(E)), F)
    L2, U2, P2 = fast_LUP(J)
    
    P2_inv = Empty(P2.dim[0], P2.dim[1])
    P2_inv.data = P2.data.copy()
    P2_inv.transpose()

    L = Empty(A.dim[0], A.dim[0])
    U = Empty(A.dim[0], A.dim[1])
    P = Empty(A.dim[1], A.dim[1])

    L21 = fast_mul(G, fast_inv(E)) 
    U12 = fast_mul(F, P2_inv)
    
    for i in range(k):
        if (i < k//2):
            L.data[i] = L1.data[i] + [0]*(k//2)
            U.data[i] = E.data[i] + U12.data[i]
        else:
            L.data[i] = L21.data[i-k//2] + L2.data[i-k//2]
            U.data[i] = [0]*(k//2) + U2.data[i-k//2]

    for i in range(A.dim[1]):
        if (i < k//2):
            P.data[i] = [0]*i + [1] + [0]*(A.dim[1]-i-1)
        else:
            P.data[i] = [0]*(k//2) + P2.data[i-k//2]
    P = P * P1
    return L, U, P
         
def eig(A, ite=50):  
    """
    Returns the eigenvalues of square matrix A in a list using QR algorithm

    Examples:
    >>> print(eig(Matrix([[2, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]])))
    [2.8793852415718155, 2.0000000000000036, 0.6527036409420062, -0.5320888825138217]
    """
    for i in range(ite):
        Q, R = qr(A)
        Q_T = Empty(Q.dim[0], Q.dim[1])
        Q_T.data = Q.data.copy()
        Q_T.transpose()
        A = Q*A*Q_T
    return [A.data[i][i] for i in range(A.dim[0])]  #the eigenvalues are on the diagonal

def powm(A: Matrix, n: int):
    """
    Returns the square matrix A to the power of n

    Examples:
    >>> print(powm(Matrix([[1, 2, 3], [1, 2, 3], [1, 1, 1]]), 4))
    [126, 195, 264]
    [126, 195, 264]
    [69, 107, 145]
    """
    if (n == 0) and (A != Empty(A.dim[0], A.dim[0])):
        return Identity(A.dim[0])

    if (n == 0) and (A == Empty(A.dim[0], A.dim[0])):
        raise ValueError("Undefined expression")

    if (n < 0):
        A = inv(A)
        n *= -1
    
    B = Identity(A.dim[0])
    if (n > 0):
        while n > 1:
            if (n % 2) == 1:
                B = B * A
                n -= 1
            A = A * A
            n /= 2
    return A * B

def sinm(A: Matrix, ite = 25):
    """
    Returns the (matrix) sine of a square matrix A

    Examples:
    >>> print(sinm(Matrix([[1, 1, 0], [1, 0, 1], [0, 1, 1]])))
    [0.5835894705445261, 0.5835894705445261, -0.25788151426337047]
    [0.5835894705445261, -0.25788151426337047, 0.5835894705445261]
    [-0.25788151426337047, 0.5835894705445261, 0.5835894705445261]
    """
    B = Empty(A.dim[0], A.dim[0])   #matrix sine stored here
    A_2 = Empty(A.dim[0], A.dim[0]) #A squared
    A_2.data = A.data.copy()
    A_2 = A_2 * A_2
    f = 1                           #odd number factorial

    for i in range(1, ite + 1): #power series calculation
        if i > 1:
            f *= (2*i-1)*(2*i - 2)
            A = A * A_2
            B = B + A * (1 / f) * ((-1)**(i-1))
        else:
            B = A
    return B 

def cosm(A: Matrix, ite = 25):
    """
    Returns the (matrix) cosine of a square matrix A

    Examples:
    >>> print(cosm(Matrix([[1, 0, 1], [1, 1, 1], [0, 1, 1]]))) 
    [0.6729735557184887, -0.25504795732268243, -0.6868620683472132]
    [-0.6868620683472132, 0.4179255983958064, -0.9419100256698957]
    [-0.25504795732268243, -0.6868620683472132, 0.4179255983958064]
    """
    C = Identity(A.dim[0])  #matrix power stored here
    B = Identity(A.dim[0])  #matrix sine stored here
    A_2 = Empty(A.dim[0], A.dim[0])
    A_2.data = A.data.copy()
    A_2 = A_2 * A_2
    f = 1                   #even number factorial

    for i in range(1, ite + 1): #power series calculation
        if i > 1:
            f *= (2*i-2)*(2*i - 3)
            C = C * A_2
            B = B + C * (1 / f) * ((-1)**(i-1))

    return B 

def expm(A: Matrix, ite = 25):
    """
    Returns the (matrix) exponential of a square matrix A

    Examples:
    >>> print(expm(Matrix([[1, 2, 0], [2, 2, 1], [0, 2, 0]])))
    [15.315254263497797, 20.825554710978267, 4.977932111577207]
    [20.825554710978267, 30.70596373056413, 7.923811299700528]
    [9.955864223154414, 15.847622599401056, 4.902476908008666]
    """  
    C = Identity(A.dim[0])  #matrix power stored here
    B = Identity(A.dim[0])  #matrix exp stored here
    f = 1                   #factorial

    for i in range(1, ite + 1): #power series calculation
        f *= i
        C = C * A
        B = B + C * (1 / f)

    return B

def fft(v: Matrix):
    """
    Returns the discrete Fourier transform of vector v with 2**K elements, using the Cooley-Turkey algorithm. 

    Examples:
    >>> print(fft(Matrix([[1], [2], [3], [4]])))
    [(10+0j)]
    [(-2+2j)]
    [(-2+0j)]
    [(-1.9999999999999998-2j)]
    """
    if (v.dim[0] & (v.dim[0]-1) != 0) or v.dim[0] == 0:    #checks of number of elements is a power of 2
        raise ValueError("Number of elements is not a power of 2")
    u = Empty(v.dim[0], v.dim[1])
    u.data = v.data.copy()
    if u.dim[0] == 1:   #base case
        return u
                        #recursive case
    N = u.dim[0]
    u.transpose()
    b = Matrix([[u.data[0][2*i] for i in range(N // 2)]])     #even number indexed elements of u
    c = Matrix([[u.data[0][2*i+1] for i in range(N // 2)]])   #odd number indexed elements of u
    b.transpose()
    c.transpose()
    
    b = fft(b)
    c = fft(c)
    A = [[0] for i in range(N)]

    for k in range(N // 2):
        e = cmath.exp(-2 * math.pi * k * complex(0, 1) / N)
        A[k][0] = b.data[k][0] + e * c.data[k][0]
        A[k + (N // 2)][0] = b.data[k][0] - e * c.data[k][0]

    return Matrix(A)

def ifft(v: Matrix):
    """
    Returns the inverse discrete Fourier transform of vector v with 2**K elements, using the Cooley-Turkey algorithm.

    Examples:
    >>> print(ifft(Matrix([[10], [-2 + 2j], [-2], [-2 - 2j]])))
    [(1+0j)]
    [(2+6.123233995736766e-17j)]
    [(3+0j)]
    [(4-6.123233995736766e-17j)]
    """
    if (v.dim[0] & (v.dim[0]-1) != 0) or v.dim[0] == 0:    #checks of number of elements is a power of 2
        raise ValueError("Number of elements is not a power of 2")

    def recifft(v: Matrix):
        u = Empty(v.dim[0], v.dim[1])
        u.data = v.data.copy()
        if u.dim[0] == 1:   #base case
            return u
                            #recursive case
        N = u.dim[0]
        u.transpose()
        b = Matrix([[u.data[0][2*i] for i in range(N // 2)]])     #even number indexed elements of u
        c = Matrix([[u.data[0][2*i+1] for i in range(N // 2)]])   #odd number indexed elements of u
        b.transpose()
        c.transpose()
        
        b = recifft(b)
        c = recifft(c)
        A = [[0] for i in range(N)]

        for k in range(N // 2):
            e = cmath.exp(2 * math.pi * k * complex(0, 1) / N)
            A[k][0] = b.data[k][0] + e * c.data[k][0]
            A[k + (N // 2)][0] = b.data[k][0] - e * c.data[k][0]
        A = Matrix(A)
        return A

    return recifft(v) * (1 / v.dim[0])


def conv(u: Matrix, v: Matrix):
    """
    Returns the convolution of vectors u and v with the same number of elements using FFT.

    Examples:
    >>> print(conv(Matrix([[1], [2], [4]]), Matrix([[1], [0], [1]])))
    [1.0]
    [2.0]
    [5.0]
    [2.0]
    [4.0]
    """
    if u.dim[0] != v.dim[0]:
        raise ValueError("Vectors must have identical number of elements")
    
    k = 1
    while k < 2*u.dim[0]:    #finding the smallest n such that c <= 2**n
        k *= 2
    
    U = Empty(1, k)
    V = Empty(1, k)

    u.transpose()
    v.transpose()
    U.data = [u.data[0] + [0 for i in range(k - u.dim[1])]] #u and v filled with zeros to have k elements
    V.data = [v.data[0] + [0 for i in range(k - v.dim[1])]]
    U.transpose()
    V.transpose()

    U = fft(U)
    V = fft(V)
    B = ifft(hadamard_product(U, V))
    C = Empty(2*u.dim[1] - 1, 1)

    for x in B.data:
        if (type(x[0]) == complex) and (x[0].imag < 1e-10):
            x[0] = x[0].real

    for i in range(2*u.dim[1] - 1):
        C.data[i] = B.data[i]

    return C

def spmv(A: Sparse, v: Matrix):
    """
    Returns the product A*v, where A is Sparse and v is vector (n x 1 Matrix)

    Examples:
    >>> print(spmv(Sparse([(0, 0, 1), (0, 1, 2), (1, 1, -2), (1, 2, 3), (2, 2, 4)], (3, 3)), Matrix([[1], [-1], [2]])))
    [-1]
    [8]
    [8]
    """
    if (A.dim[1] != v.dim[0]):
        raise ValueError("Dimensions are not compatible")

    u = Empty(A.dim[0], 1)
    i = 0
    for x in A.data:
        if x[0] == i:
            u.data[i][0] += v.data[x[1]][0] * x[2]
        else:
            i = x[0]
            u.data[i][0] += v.data[x[1]][0] * x[2]

    return u        

def sREF(M: Sparse):    
    """
    Returns reduced echelon form of (Sparse) matrix M

    Examples:
    >>> print(sREF(Sparse([[0, 1, 1], [0, 2, 2], [1, 0, 1], [2, 1, 3], [2, 2, 6]], (3, 4))))
    [1, 0, 0, 0]
    [0, 1, 2, 0]
    [0, 0, 0, 0]
    """ 
    A = Sparse([], (M.dim[0], M.dim[1]))
    A.data = M.data.copy()

    if A.data == []: #Base case, zero matrix is in row echelon form
        return A
                     #Recursive case
    column = A.dim[1]-1
    index = 0   
    for i in range(len(A.data)):
        if (A.data[i][1] < column):
            column = A.data[i][1]
            index = i
        if (A.data[i][1] == 0):
            break

    C_row = [i for i in range(len(A.data)) if A.data[i][0] == A.data[index][0]]
    first_row = [i for i in range(len(A.data)) if A.data[i][0] == 0]
    if A.data[index][0] > 0: #swapping with first row
        for i in first_row:
            A.data[i][0] = A.data[index][0]
        for i in C_row:
            A.data[i][0] = 0
    A.data = sorted(A.data, key = lambda x: (x[0], x[1]))

    C_row = [i for i in range(len(A.data)) if (A.data[i][1] == A.data[0][1]) and (A.data[i][0] != 0)]
    first_row = [i for i in range(len(A.data)) if A.data[i][0] == 0]

    for i in C_row: #elimination
        f_row = first_row.copy()
        row = [j for j in range(len(A.data)) if (A.data[j][0] == A.data[i][0])]
        a = f_row.pop(0)        
        b = row.pop(0)
        c = -A.data[b][2]/A.data[a][2]
        end = False
        while end == False:
            if f_row == []:
                end = True
            if A.data[a][1] > A.data[b][1]:
                if row != []:
                    b = row.pop(0)
            elif A.data[a][1] == A.data[b][1]:
                A.data[b][2] += c*A.data[a][2]
                if f_row != []:
                    a = f_row.pop(0)
                if row != []:
                    b = row.pop(0)
            elif A.data[a][1] < A.data[b][1]:
                A.data.append([A.data[b][0], A.data[a][1], c*A.data[a][2]])
                if f_row != []:                
                    a = f_row.pop(0)
                if row != []:
                    b = row.pop(0)
    
    A.data = sorted(A.data, key = lambda x: (x[0], x[1]))

    for x in A.data:    #zero elements deletion
        if abs(x[2]) < 1e-10:
            A.data.remove(x)

    R = Sparse([[A.data[i][0]-1, A.data[i][1]-1, A.data[i][2]] for i in range(len(A.data)) if (A.data[i][0] != 0)], (A.dim[0]-1, A.dim[1]-1))
    B = sREF(R)
    C = Sparse([], (A.dim[0], A.dim[1]))
    C.data = [A.data[i] for i in range(len(A.data)) if A.data[i][0] == 0] + [[B.data[i][0] + 1, B.data[i][1] + 1, B.data[i][2]] for i in range(len(B.data))]  
    return C

