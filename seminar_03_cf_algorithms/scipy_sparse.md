# Scipy sparse

There are two formats in `scipy`: **CSR** (Compressed Sparse Row) and **CSC** (Compressed Sparse Row).
Let's look at this using the example of the following matrix:

$$
\begin{pmatrix}
0 & 0 & 1 \\
4 & 0 & 0 \\
0 & 0 & 3
\end{pmatrix}
$$

## CSR

### 1. Identify non-zero elements row by row

Row 0:
- $(0,2) = 1$

Row 1:
- $(1,0) = 4$

Row 2:
- $(2,2) = 3$

---

### 2. `data` array

All non-zero values stored in row order:

```
data = [1, 4, 3]
```

---

### 3. `indices` array

Column index of each value in `data`:

```
indices = [2, 0, 2]
```

Meaning:

| value | column |
|------|------|
| 1 | 2 |
| 4 | 0 |
| 3 | 2 |

---

### 4. `indptr` array

`indptr` shows where each row starts in the `data` array.

```
indptr = [0, 1, 2, 3]
```

Explanation:

- Row 0 starts at index `0` in `data`
- Row 1 starts at index `1`
- Row 2 starts at index `2`
- Final value `3` = total number of stored values

---

### 5. Reading rows using `indptr`

Rule:

```
row i → data[indptr[i] : indptr[i+1]]
```

#### Row 0

```
data[0:1] = [1]
indices[0:1] = [2]
```

Row:

$$
[0,0,1]
$$

#### Row 1

```
data[1:2] = [4]
indices[1:2] = [0]
```

Row:

$$
[4,0,0]
$$

#### Row 2

```
data[2:3] = [3]
indices[2:3] = [2]
```

Row:

$$
[0,0,3]
$$

---

### Final CSR representation

```
data    = [1, 4, 3]
indices = [2, 0, 2]
indptr  = [0, 1, 2, 3]
```
---

## CSC

### 1. Identify non-zero elements column by column

Column 0:
- $(1,0) = 4$

Column 1:
- no non-zero values

Column 2:
- $(0,2) = 1$
- $(2,2) = 3$

---

### 2. `data` array

All non-zero values stored in column order:

```
data = [4, 1, 3]
```

---

### 3. `indices` array

Row index of each value in `data`:

```
indices = [1, 0, 2]
```

Meaning:

| value | row |
|------|------|
| 4 | 1 |
| 1 | 0 |
| 3 | 2 |

---

### 4. `indptr` array

`indptr` shows where each column starts in the `data` array.

```
indptr = [0, 1, 1, 3]
```

Explanation:

- Column 0 starts at index `0`
- Column 1 starts at index `1`
- Column 2 starts at index `1`
- Final value `3` = total number of stored values

Note: Column 1 has no non-zero elements, so its start and end indices are the same.

---

### 5. Reading columns using `indptr`

Rule:

```
column j → data[indptr[j] : indptr[j+1]]
```

#### Column 0

```
data[0:1] = [4]
indices[0:1] = [1]
```

Column:

$$
\begin{pmatrix}
0 \\
4 \\
0
\end{pmatrix}
$$

---

#### Column 1

```
data[1:1] = []
indices[1:1] = []
```

Column:

$$
\begin{pmatrix}
0 \\
0 \\
0
\end{pmatrix}
$$

---

#### Column 2

```
data[1:3] = [1, 3]
indices[1:3] = [0, 2]
```

Column:

$$
\begin{pmatrix}
1 \\
0 \\
3
\end{pmatrix}
$$

---

### Final CSC representation

```
data    = [4, 1, 3]
indices = [1, 0, 2]
indptr  = [0, 1, 1, 3]
```