# matrixlib

## 1. Zadání problému

Vytvořit knihovnu s maticovými operacemi jako je např. sčítání, násobení, inverze, determinanty, řešení soustav lineárních rovnic apod. 

## 2. Uživatelská část

Knihovna používá třídy <strong>Matrix</strong> a <strong>Sparse</strong>, pro vytvoření matice musí uživatel zadat jednotlivé prvky (typu <strong>int</strong>, nebo <strong>float</strong>) do dvojrozměrného pole:

```python 
A = Matrix([[1, 2.71],
            [3, 4.32]])
```

Pro vytvoření řídké matice musí uživatel zadat nenulové prvky společně s indexy řádku a sloupce do pole (seřazené podle (řádek, sloupec)) a specifikovat rozměr matice:

```python 
B = Sparse([[0, 0, 2], [1, 1, -1], [1, 2, 4]], (4, 3))
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

Rozměr matice lze získat pomocí atributu <strong>dim</strong>, jako tuple <strong>(m, n)</strong>, kde <strong>m</strong> je počet řádků a <strong>n</strong> je počet sloupců matice. Lze použít i pro matice typu Sparse.


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

Pro součet dvou matic stačí použít operátor <strong>+</strong>. Lze použít i pro matice typu Sparse.

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

Pro odečtení dvou matic stačí použít operátor <strong>-</strong>. Lze použít i pro matice typu Sparse.

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

Pro násobení dvou matic stačí použít operátor <strong>*</strong>. Lze použít i pro matice typu Sparse.

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

Rovnost matic lze porovnávat pomocí operátoru <strong>==</strong>. Lze použít i pro matice typu Sparse.

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

Matici lze transponovat pomocí metody <strong>transpose</strong>. Lze použít i pro matice typu Sparse.


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

### Hadamardův součin

Hadamardův součin dvou matic stejného typu lze vypočítat pomocí funkce <strong>hadamard_product</strong>

```python 
print(hadamard_product(Matrix([[1, 3, 4], [5, 4, 2], [7, 5, 3]]), Matrix([[4, 5, 2], [9, 7, 1], [2, 2, 2]])))
```

Výstup:

```python 
[4, 15, 8]
[45, 28, 2]
[14, 10, 6]
```

### Stopa matice

Stopu čtvercové matice lze vypočítat pomocí funkce <strong>trace</strong>

```python 
print(trace(Matrix([[1, 4, 7], [2, -1, 3], [3, 4, 0]])))
```

Výstup:

```python 
0
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

### Rychlé umocňování matice

Pro výpočet celočíselné mocniny čtvercové matice lze použít funkci <strong>powm</strong>.

```python 
print(powm(Matrix([[1, 2, 3], [1, 2, 3], [1, 1, 1]]), 4))
```

Výstup:

```python 
[126, 195, 264]
[126, 195, 264]
[69, 107, 145]
```

### Sinus z matice

Pro výpočet sinu z čtvercové matice lze použít funkci <strong>sinm</strong>. Uživatel může specifikovat počet iterací.

```python 
print(sinm(Matrix([[1, 1, 0], [1, 0, 1], [0, 1, 1]]), 25))
```

Výstup:

```python 
[0.5835894705445261, 0.5835894705445261, -0.25788151426337047]
[0.5835894705445261, -0.25788151426337047, 0.5835894705445261]
[-0.25788151426337047, 0.5835894705445261, 0.5835894705445261]
```

### Cosinus z matice

Pro výpočet cosinu z čtvercové matice lze použít funkci <strong>cosm</strong>. Uživatel může specifikovat počet iterací.

```python 
print(cosm(Matrix([[1, 0, 1], [1, 1, 1], [0, 1, 1]]), 25)) 
```

Výstup:

```python 
[0.6729735557184887, -0.25504795732268243, -0.6868620683472132]
[-0.6868620683472132, 0.4179255983958064, -0.9419100256698957]
[-0.25504795732268243, -0.6868620683472132, 0.4179255983958064]
```

### Maticová exponenciála

Pro výpočet maticové exponenciály z čtvercové matice lze použít funkci <strong>expm</strong>. Uživatel může specifikovat počet iterací.

```python 
print(expm(Matrix([[1, 2, 0], [2, 2, 1], [0, 2, 0]]), 25))
```
Výstup:

```python 
[15.315254263497797, 20.825554710978267, 4.977932111577207]
[20.825554710978267, 30.70596373056413, 7.923811299700528]
[9.955864223154414, 15.847622599401056, 4.902476908008666]
```

### Rychlá diskrétní Fourierova transformace vektoru

Pro výpočet diskrétní Fourierovy transformace vektoru v, vektor musí mít počet prvků rovný nějaké mocnině dvou, lze použít funkci <strong>fft</strong>.

```python 
print(fft(Matrix([[1], [2], [3], [4]])))
```
Výstup:

```python 
[(10+0j)]
[(-2+2j)]
[(-2+0j)]
[(-1.9999999999999998-2j)]
```

### Rychlá inverzní diskrétní Fourierova transformace vektoru

Pro výpočet inverzní diskrétní Fourierovy transformace vektoru v, vektor musí mít počet prvků rovný nějaké mocnině dvou, lze použít funkci <strong>ifft</strong>.

```python 
print(ifft(Matrix([[10], [-2 + 2j], [-2], [-2 - 2j]])))
```
Výstup:

```python 
[(1+0j)]
[(2+6.123233995736766e-17j)]
[(3+0j)]
[(4-6.123233995736766e-17j)]
```

### Rychlá konvoluce vektorů

Pro výpočet konvoluce dvou vektorů o stejném počtu prvků lze použít funkci <strong>conv</strong>. 

```python 
print(conv(Matrix([[1], [2], [4]]), Matrix([[1], [0], [1]])))
```
Výstup:

```python 
[1.0]
[2.0]
[5.0]
[2.0]
[4.0]
```

### Násobení řídké matice vektorem

Pro vynásobení řídké matice (<strong>Sparse</strong>) vektorem (<strong>Matrix</strong>) lze použít funkci <strong>spmv</strong>.

```python 
print(spmv(Sparse([(0, 0, 1), (0, 1, 2), (1, 1, -2), (1, 2, 3), (2, 2, 4)], (3, 3)), Matrix([[1], [-1], [2]])))
```
Výstup:

```python 
[-1]
[8]
[8]
```

### Řádkově odstupňovaný tvar řídké matice

Pro výpočet řádkově odstupňovaného tvaru řídké matice lze použít funkci <strong>sREF</strong>.

```python 
print(sREF(Sparse([[0, 1, 1], [0, 2, 2], [1, 0, 1], [2, 1, 3], [2, 2, 6]], (3, 4))))
```
Výstup:

```python 
[1, 0, 0, 0]
[0, 1, 2, 0]
[0, 0, 0, 0]
```

## 3. Programátorská část

<par>První část programu obsahuje třídu Matrix, pomocí které uživatel zadá matici. Tato třída má atributy <strong>data</strong> a <strong>dim</strong>. V <strong>data</strong> jsou uloženy prvky matice v dvourozměrném poli, prvky mohou být typu <strong>int</strong> a <strong>float</strong>. <strong>dim</strong> je tuple <strong>(m, n)</strong>, kde <strong>m</strong> je počet řádků a <strong>n</strong> je počet sloupců. V třídě Matrix je naprogramováno několik metod pro vypisování, sčítání, odečítání, násobení, porovnávání a transponování matice.</par>

<par>Druhá část programu obsahuje funkce, které mají jako vstup buď číslo <strong>int</strong>, nebo matici <strong>Matrix</strong>.</par>

<par>Hlavní použité algoritmy jsou:
- <em>Gaussova eliminace</em>, např. ve funkci <strong>REF</strong>
- <em>Strassenův algoritmus</em>, ve funkci <strong>fast_mul</strong>
- <em>Gramova–Schmidtova ortogonalizace</em>, ve funkci <strong>qr</strong>
- <em>QR algoritmus</em>, ve funkci <strong>eig</strong></par>

<par>Až na funkce <strong>qr</strong> a  <strong>eig</strong>, vužívají všechny funkce numericky stabilní algoritmy. Funkce <strong>eig</strong> používá <em>QR algoritmus</em>, který je numerický a u kterého není zaručena konvergence k vlastním číslům pro všechny vstupní matice.</par>
