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

## 3. Programátorská část
