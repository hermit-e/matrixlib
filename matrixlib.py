import copy

class Matrix:
    """
    Attributes:
        data: 2 dimensional array of integers/floats.
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
        if (type(self) == Matrix) and ((type(other) == int) or (type(other) == float)): #scalar multiplication
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
