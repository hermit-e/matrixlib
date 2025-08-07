# matrixlib

## 1. Zadání problému

Vytvořit knihovnu s maticovými operacemi jako je např. sčítání, násobení, inverze, determinanty, řešení soustav lineárních rovnic apod. 

## 2. Uživatelská část

Knihovna používá třídu <strong>Matrix</strong>, pro vytvoření matice musí uživatel zadat jednotlivé prvky (typu <strong>int</strong>, nebo <strong>float</strong>) do dvojrozměrného pole:

```python 
A = Matrix([[1, 2.71],
            [3, 4.32]])
```
### Vypsání matice

Pro vypsání matice stačí použít funkci <strong>print()</strong>

```python 
print(A)
```

Výstup:

```python 
[1, 2.71]
[3, 4.32]
```

### Rozměr matice

Rozměr matice lze získat pomocí atributu <strong>dim</strong>, jako tuple <strong>(m, n)</strong>, kde <strong>m</strong> je počet řádků a <strong>n</strong> je počet sloupců matice.


```python 
A = Matrix([[1, 2.71],
            [3, 4.32]])
print(A.dim)
```

Výstup:

```python 
(2, 2)
```

### Sčítání matic

Pro součet dvou matic stačí použít operátor <strong>+</strong>

```python 
A = Matrix([[1, 2, 3],
            [4, 5, 6]])

B = Matrix([[1, 1, 1],
            [1, 1, 1]])

print(A + B)
```

Výstup:

```python 
[2, 3, 4]
[5, 6, 7]
```

### Odečítání matic

Pro odečtení dvou matic stačí použít operátor <strong>-</strong>

```python 
A = Matrix([[1, 2, 3],
            [4, 5, 6]])

B = Matrix([[1, 1, 1],
            [1, 1, 1]])

print(A - B)
```

Výstup:

```python 
[0, 1, 2]
[3, 4, 5]
```

### Násobení matic

Pro násobení dvou matic stačí použít operátor <strong>*</strong>

```python 
A = Matrix([[1, 2],
            [4, 5]])

B = Matrix([[2, 3],
            [2, 2]])

print(A * B)
```

Výstup:

```python 
[6, 7]
[18, 22]
```

### Rovnost matic

Rovnost matic lze porovnávat pomocí operátoru <strong>==</strong>

```python 
A = Matrix([[1, 2],
            [4, 5]])

B = Matrix([[2, 3],
            [2, 2]])

print(A == B)
```

Výstup:

```python 
False
```

### Transpozice matice

Matici lze transponovat pomocí metody <strong>transpose</strong>


```python 
A = Matrix([[1, 2, 3],
            [4, 5, 6]])

A.transpose()

print(A)
```

Výstup:

```python 
[1, 4]
[2, 5]
[3, 6]
```

### Jednotková matice

Jednotkovou matici řádu <strong>n</strong> lze vytvořit pomocí funkce <strong>Identity</strong>


```python 
print(Identity(3))
```

Výstup


```python 
[1, 0, 0]
[0, 1, 0]
[0, 0, 1]
```

### Nulová matice

Nulovou matici typu <strong>m</strong> x <strong>n</strong> lze vytvořit pomocí funkce <strong>Empty</strong>

```python 
print(Empty(3, 2))
```

Výsledek:

```python 
[0, 0]
[0, 0]
[0, 0]
```

### Skalární součin

Skalární součin dvou vektorů (zde bráno jako matice typu <strong>m</strong> x <strong>1</strong>) lze vypočítat pomocí funkce <strong>dot</strong>

```python 
a = Matrix([[1], [2], [3]])
b = Matrix([[4], [5], [6]])
print(dot(a, b))
```

Výstup:

```python 
32
```

### LUP rozklad matice

Rozložení čtvercové matice na součin dolní trojúhelníkové (L), horní trojúhelníkové (U) a permutační matice (P), lze vypočítat pomocí funkce <strong>LUP</strong>, funkce navíc vrátí počet prohození řádků

```python 
L, U, P, row_ex = LUP(Matrix([[1, 4, -3], [-2, 8, 5], [3, 4, 7]]))
print(L)
print(U)
print(P)
```

Výstup:

```python 
[1, 0, 0]
[-2.0, 1, 0]
[3.0, -0.5, 1]
[1, 4, -3]
[0.0, 16.0, -1.0]
[0.0, 0.0, 15.5]
[1, 0, 0]
[0, 1, 0]
[0, 0, 1]
```

### Determinant matice

Determinant čtvercové matice lze vypočítat pomocí funkce <strong>det</strong>

```python 
print(det(Matrix([[1, 2, 3], [4, 5, 6], [7, 3, 1]])))
```

Výstup:

```python 
-6.0
```

### Inverzní matice

Inverzní matici k čtvercové matici lze vypočítat pomocí funkce <strong>inv</strong>

```python 
print(inv(Matrix([[1, 0, 5], [2, 1, 6], [3, 4, 0]])))
```

Výstup:

```python 
[-24.0, 20.0, -5.0]
[18.0, -15.0, 4.0]
[5.0, -4.0, 1.0]
```

### Řešení soustavy lineárních rovnic

Pro vyřešení soustavy lineárních rovnic musí uživatel zadat matici soustavy a vektor pravých stran do funkce <strong>solve</strong>

```python 
print(solve(Matrix([[1, 2, 3], [3, 2, 1], [3, 1, 2]]), Matrix([[14], [10], [11]])))
```

Výstup:

```python 
[1.0]
[2.0]
[3.0]
```

### Řádkově odstupňovaný tvar matice

Řádkově odstupňovaný tvar matice lze vypočítat pomocí funkce <strong>REF</strong>

```python 
print(REF(Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))
```

Výstup:

```python 
[1, 2, 3]
[0.0, -3.0, -6.0]
[0.0, 0.0, 0.0]
```

### Choleského rozklad

Pro výpočet Choleského rozkladu pozitivně definitní matice lze použít funkce <strong>cholesky</strong>

```python 
print(cholesky(Matrix([[9, 6, 3], [6, 8, 2], [3, 2, 5]])))
```

Výstup:
```python
[3.0, 0, 0]
[2.0, 2.0, 0]
[1.0, 0.0, 2.0]
```

### QR rozklad

Pro výpočet QR rozkladu čtvercové matice (<strong>Q</strong> je ortogonální matice, <strong>R</strong> je horní trojúhelníková matice) lze použít funkci <strong>qr</strong>

```python 
Q, R = qr(Matrix([[1,0,-1],[-2,1,4], [1, 3, 3]]))
print(Q)
print(R)
```

Výstup:

```python
[0.4082482904638631, -0.8164965809277261, 0.4082482904638631]
[-0.05314940034527339, 0.42519520276218714, 0.9035398058696477]
[0.9113223768657689, 0.39056673294246624, -0.1301889109808269]
[2.449489742783178, 0.408248290463863, -2.4494897427831788]
[0, 3.13581462037113, 4.464549629002965]
[0, 0, 0.26037782196164777]
```

### Rychlé násobení matic

Pro rychlé vynásobení dvou matic lze využít funkci <strong>fast_mul</strong>

```python 
print(fast_mul(Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), Matrix([[2, 3, 1], [7, 4, 2], [2, 2, 9]])))
```

Výstup:

```python 
[22, 17, 32]
[55, 44, 68]
[88, 71, 104]
```

### Rychlý výpočet inverzní matice

Pro rychlý výpočet inverzní matice (regulární, řádu mocniny dvou) lze použít funkci <strong>fast_inv</strong>

```python 
print(fast_inv(Matrix([[2, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]])))
```

Výstup:

```python 
[0.5, 0.0, -0.5, 0.5]
[0.0, 0.0, 1.0, -1.0]
[-0.5, 1.0, -0.5, 0.5]
[0.5, -1.0, 0.5, 0.5]
```

### Rychlý LUP rozklad

Pro rychlý výpočet LUP rozkladu matice (regulární, počet řádků je mocnina dvou) lze použít funkci <strong>fast_LUP</strong>

```python 
L, U, P = fast_LUP(Matrix([[3, -7, -2, 2], [-3, 5, 1, 0], [6, -4, 0, -5], [-9, 5, -5, 12]]))
print(L)
print(U)
print(P)
```

Výstup:

```python 
[1, 0, 0, 0]
[-1.0, 1, 0, 0]
[2.0, -4.999999999999999, 1, 0]
[-3.0, 7.999999999999999, 3.000000000000006, 1]
[3, -7, -2.0, 2]
[0, -2.0, -1.0, 2.0]
[0, 0, -0.9999999999999982, 0.9999999999999982]
[0, 0, 0, -0.9999999999999991]
[1, 0, 0, 0]
[0, 1, 0, 0]
[0, 0, 1, 0]
[0, 0, 0, 1]
```

### Výpočet vlastních čísel

Pro výpočet vlastních čísel matice lze využít funkci <strong>eig</strong>, uživatel může specifikovat počet iterací, jinak je počet iterací nastaven na 50.


```python 
print(eig(Matrix([[2, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]]), 50))
```

Výstup:

```python 
[2.8793852415718155, 2.0000000000000036, 0.6527036409420062, -0.5320888825138217]
```

## 3. Programátorská část
