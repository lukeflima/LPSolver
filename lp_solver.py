import math

def is_optimal(matrix):
    c = matrix[-1]
    return all(c[i] >= -1e-9 for i in range(len(c)-1))


def choose_next_basis(A, dual):
    c = [row[-1] for row in A[:-1]] + [0] if dual else A[-1]
    
    next = min(range(len(c) - 1), key=lambda i: c[i])
    
    if c[next] <= -1e-9:
        return next
    
    return -1


def choose_leaving_var(matrix, next_basis_var, using_psdeo, dual):
    n = len(matrix)
    m = len(matrix[0])-1

    k = 2 if using_psdeo else 1
    ratios = []
    if dual:
        for j in range(m-k):
            if matrix[next_basis_var][j] < -1e-9 and abs(matrix[-1][j]) > 1e-9:
                ratios.append((j, -matrix[-1][j]/ matrix[next_basis_var][j]))
    else:
        for i in range(n-k):
            if matrix[i][next_basis_var] > 1e-9:
                ratios.append((i, matrix[i][-1] / matrix[i][next_basis_var]))
    
    if len(ratios) == 0:
        return -1
    
    leaving = min(ratios, key=lambda x: x[1])[0]

    return leaving



def pivot(matrix, basis, next_basis_var, leaving, dual):
    if dual:
        next_basis_var, leaving = leaving, next_basis_var
    
    n = len(matrix)
    m = len(matrix[0])-1
    
    basis[leaving] = next_basis_var

    if matrix[leaving][next_basis_var] != 1:
        factor = matrix[leaving][next_basis_var]
        for j in range(len(matrix[leaving])):                            
            matrix[leaving][j] /= factor
    for i in range(n):
        if i != leaving and abs(matrix[i][next_basis_var]) >= 1e-9:
            factor = matrix[i][next_basis_var]
            for j in range(m+1):                            
                matrix[i][j] -= factor * matrix[leaving][j]


def simplex(matrix, basis=None, phase=1, dual=False):
    n = len(matrix)
    m = len(matrix[0])-1
    
    if basis is None:
        basis = [m + i for i in range(n-1)]
        
        pseudo_objective = [0]*(m + n)
        for i in range(n-1):
            for j in range(m):
                pseudo_objective[j] -= matrix[i][j]
            pseudo_objective[-1] -= matrix[i][-1]

            matrix[i] = matrix[i][:-1] + [0] * (i) +[1] + [0]*(n - i - 2) + [matrix[i][-1]]
        
        matrix[-1] = matrix[-1][:-1] + [0]*(n-1) + [matrix[-1][-1]]
        m += n - 1
        
        matrix.append(pseudo_objective)
        n += 1
        
    while True:
        next_basis_var = choose_next_basis(matrix, dual)
        if next_basis_var != -1:
            leaving = choose_leaving_var(matrix, next_basis_var, using_psdeo=phase == 1, dual=dual)
            if leaving == -1:
                break
            pivot(matrix, basis, next_basis_var, leaving, dual)
        else:
            break

    #remove artificial variables from matrix
    if phase == 1:
        if abs(matrix[-1][-1]) > 1e-9:
            print("Must be 0 for feasable solution")
            exit(1)
        
        new_matrix = []
        m -= len(basis)
        n -= 1
        for i in range(n):
            new_matrix.append(matrix[i][:m] + [matrix[i][-1]])
        matrix = new_matrix   

        for i in range(len(basis)):
            if basis[i] >= m:
                next_basis_var = -1
                for j in range(m):
                    if abs(matrix[i][j]) > 1e-9:
                        next_basis_var = j
                        break
                
                if next_basis_var == -1:
                    continue
                pivot(matrix, basis, next_basis_var, i, dual)
    
    return -matrix[-1][-1], basis, matrix

    
def solve_lp(c, A, b, basis=None, dual=False, c0=0):
    matrix = [A[i] + [b[i]] for i in range(len(b))]
    matrix.append(c+[c0])
    if basis is None:
        sol, basis, matrix = simplex(matrix, basis, 1, dual)
        if is_optimal(matrix):
            return sol, basis, matrix
    return simplex(matrix, basis, 2, dual)


def solve_mixedin(c, A, b, x=None, c0=0):
    if x is None or all(i == 0 for i in x):
        return solve_lp(c, A, b, c0=c0)
    
    matrix = [A[i] + [b[i]] for i in range(len(b))]
    matrix.append(c+[c0])
    basis = None
    iterations = 0
    dual = False
    while True:
        A = [row[:-1] for row in matrix[:-1]]
        b = [row[-1] for row in matrix[:-1]]
        c = matrix[-1][:-1]
        c0 = matrix[-1][-1]
        sol, basis, matrix = solve_lp(c, A, b, basis, dual, c0)
        if iterations > 100:
            break
        
        is_integer_solution = all(abs(matrix[i][-1] - round(matrix[i][-1])) < 1e-9 for i in range(len(matrix) -1))
        if is_integer_solution:
            if is_optimal(matrix):
                break
            else:
                dual = False
                continue
        else:
            # find a basic integer variable that is not integer to cutting plane
            for i in range(len(basis)):
                if x[i] and abs(matrix[i][-1] - round(matrix[i][-1])) >= 1e-9:
                    new_row = [0.0]* (len(matrix[0]))
                    for j in range(len(matrix[i])):
                        new_row[j] = -matrix[i][j] + math.floor(matrix[i][j])
                    new_row[-1] = -matrix[i][-1] + math.floor(matrix[i][-1])
                    new_row.insert(-1, 1.0)  # new slack variable
                    
                    # adding new col
                    for i in range(len(matrix)):
                        matrix[i].insert(-1, 0.0)
                    
                    matrix.insert(-1, new_row)
                    basis.append(len(matrix[0]) -2)

                    dual = new_row[-1] < -1e-9
                    
                    break
            iterations += 1
    
    return sol, basis, matrix