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

<par>První část programu obsahuje třídy Matrix a Sparse, pomocí kterých uživatel zadá matici. Třída Matrix má atributy <strong>data</strong> a <strong>dim</strong>. V <strong>data</strong> jsou uloženy prvky matice v dvourozměrném poli, prvky mohou být typu <strong>int</strong>, <strong>float</strong> a <strong>complex</strong>. <strong>dim</strong> je tuple <strong>(m, n)</strong>, kde <strong>m</strong> je počet řádků a <strong>n</strong> je počet sloupců. V třídě Matrix je naprogramováno několik metod pro vypisování, sčítání, odečítání, násobení, porovnávání a transponování matice.</par>

<par>Třída Sparse má atributy <strong>data</strong> a <strong>dim</strong>. V <strong>data</strong> jsou uloženy prvky matice a jejich indexy řádku a sloupce v poli, prvky mohou být typu <strong>int</strong>, <strong>float</strong> a <strong>complex</strong>. <strong>dim</strong> je tuple <strong>(m, n)</strong>, kde <strong>m</strong> je počet řádků a <strong>n</strong> je počet sloupců, narozdíl od třídy Matrix musí rozměry matice uživatel specifikovat. V třídě Sparse je naprogramováno několik metod pro vypisování, sčítání, odečítání, násobení, porovnávání a transponování matice.</par>

<par>Program využívá knihovny ze standardních knihoven Pythonu a to: <strong>copy</strong>, <strong>math</strong> a <strong>cmath</strong>.</par>

<par>Druhá část programu obsahuje funkce, které mají jako vstup buď číslo <strong>int</strong>, nebo matici <strong>Matrix</strong>/<strong>Sparse</strong>.</par>

### Hlavní použité algoritmy

#### Gaussova eliminace

Gaussova eliminace převede matici libovolného typu do tzv. řádkově odstupňovaného tvaru, za pomocí tzv. elementárních řádkových úprav (prohazování řádků, násobení řádku nenulovým skalárem a příčítání násobku jednoho řádku k druhému)[^fn1]. Využita např. ve funkcích <strong>REF</strong> a <strong>sREF</strong>.

[^fn1]: BARTO, Libor a TŮMA, Jiří. Lineární algebra. Online. Dostupné z: https://www.mff.cuni.cz/data/web/obsah/department_math/ka/skripta_la7.pdf. [cit. 2025-08-19].

#### Strassenův algoritmus

Strassenův algoritmus umožňuje rychlé vynásobení dvou matic. Matice se nejprve vyplní nulami, tak aby byly čtvercové a řádu mocniny dvou. Následně se rozdělí na 4 bloky a pomocí předem vypočítaných vzorců se bloky sčítají a násobí, přičemž násobení se provede rekurzivně, tak aby vyšel součin dvou matic[^fn2]. Algoritmus je využit ve funkci <strong>fast_mul</strong>.

[^fn2]: STANOVSKÝ, David a BARTO, Libor. Počítačová algebra. Druhé, upravené vydání. Praha: Matfyzpress, 2017. ISBN 978-80-7378-340-2.

#### Gramova–Schmidtova ortogonalizace

Gramova-Schmidtova ortogonalizace dostane jako vstup lineárně nezávislou posloupnost vektorů a vrátí ortnonormální posloupnost vektorů se stejným lineárním obalem[^fn1]. Využita ve funkci <strong>qr</strong>.

#### QR algoritmus

QR algoritmus dostane jako vstup matici a postupným děláním QR rozkladu vstupní matice a násobením maticí Q a Q transponovanou za určitých podmínek konverguje matice k horní trojúhelníkové matici s vlastními čísli na diagonále[^fn3]. Využit ve funkci <strong>eig</strong>.

[^fn3]: QR algorithm. Online. In: Wikipedia: the free encyclopedia. San Francisco (CA): Wikimedia Foundation, 2001-. Dostupné z: https://en.wikipedia.org/wiki/QR_algorithm. [cit. 2025-08-19].

#### Rychlá inverze čtvercové matice

Algoritmus dostane jako vstup regulární matici řádu mocniny dvou, kterou rozdělí na 4 bloky a pomocí předem vypočítaných vzorců se bloky sčítají a násobí, tak aby vyšla inverze matice. Při násobení matic je využit Strassenův algoritmus[^fn4]. Využita ve funkci <strong>fast_LUP</strong>.

[^fn4]: Blockwise inversion. Online. In: Wikipedia: the free encyclopedia. San Francisco (CA): Wikimedia Foundation, 2001-. Dostupné z: https://en.wikipedia.org/wiki/Invertible_matrix#Blockwise_inversion. [cit. 2025-08-19].

#### Rychlý LUP rozklad matice

Algoritmus dostane jako vstup matici typu <strong>m</strong> x <strong>n</strong>, kde <strong>m</strong> je mocnina dvou a vrátí LUP rozklad matice právě tehdy, když je vstupní matice regulární. Matici podélně rozdělí na 2 bloky a následně z nich rekurzivně spočítá LUP rozklad a pomocí předem vypočítaných vzorců a správného umístění bloků zredujuje problém na vynásobí dvou matic, tady je použit Strassenův algoritmus[^fn5]. Funkce <strong>fast_LUP</strong>.

[^fn5]: KUČERA, Luděk a NEŠETŘIL, Jaroslav. Algebraické metody diskrétní matematiky. Praha, 1989.

#### Cooley-Turkeyův algoritmus

Cooley-Turkeyův algoritmus dostane jako vstup vektor s počtem členů rovný mocnině dvou a vrátí jeho diskrétní Fourierovu transformaci. Vektor se podélně rozdělí na 2 vektory s polovičním počtem prvků, následně se rekurzivně určí jejich diskrétní Fourierova transformace a pomocí jedné vlastnosti primitivního kořenu jednotky se z těchto transformací určí diskrétní Fourierova transformace původního vektoru[^fn5]. Využit ve funkci <strong>conv</strong>.

#### Rychlá konvoluce vektorů

Vstupem jsou dva vektory stejné délky, výstup je konvoluce dvou vektorů. Nejprve algoritmus vyplní oba vektory nulami, tak aby byl počet jejich prvků mocnina dvou, následně provede diskrétní Fourierovu transformaci obou vektorů, zde je použit Cooley-Turkeyův algoritmus. Naposledy se využívá věty, že diskrétní Fourierova transformace konvoluce dvou vektorů je hadamardův součin diskrétních Fourierových transformací těchto vektorů[^fn5]. Funkce <strong>conv</strong>.
