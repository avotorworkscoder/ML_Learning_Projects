import numpy as np 




print(np.zeros((2, 2,2)))
print(np.ones((3, 3)))
print(np.arange(5,20,2))
print(np.linspace(0,1,5))
print(np.random.rand(3,1,2))
print(np.random.randint(0,10,(2,2)))
print(np.eye(4))
print(np.random.seed(42))
print(np.random.rand(2,2))
 
b= np.array([])
for i in range(2):
    c=int(input("Enter a number: "))
    b = np.append(b,c).astype(int)
print("Array b:", b)

input_num = int(input("Enter the number of elements: "))
b=np.array([int(input("Enter element {}: ".format(i+1))) for i in range(input_num)])
print("Array b:", b)

assert all(isinstance(x, (int, np.integer)) for x in b), "All elements must be integers"

print(b[0])

b=np.array([int(x) for x in input("Enter elements separated by spaces: ").split()])
print("Array b:", b)

b=np.array(list(map(int, input("Enter elements separated by spaces: ").split())))
print("Array b:", b)

b=np.fromstring(input("Enter elements separated by spaces: "), dtype=int, sep=' ')
print("Array b:", b)

# Get dimensions from the user
rows = int(input("Enter number of rows: "))
cols = int(input("Enter number of columns: "))

# One-liner to build the 2D array
b = np.array([[int(input(f"Enter element [{r}][{c}]: ")) for c in range(cols)] for r in range(rows)])
print("2D Array b:\n", b)


# Get dimensions from the user
rows = int(input("Enter number of rows: "))
cols = int(input("Enter number of columns: "))

# One-liner to build the 2D array
b=np.array([int(input(f"Enter element {i+1}: ")) for i in range(rows*cols)]).reshape(rows, cols)
print("2D Array b:\n", b)



a = np.array([1, 2, 3,4,5])
print(a)

print(a[2:4])
print(a[:3])
print(a[3:])
print(a[::2])
print(a[-3:])

b=np.array([[1,2,3],[4,5,6],[7,8,9]])
mask=b>5
print(b[mask])

print(b.ndim)

c=a[1:3]
c[0]=99
print(c)
print(a)  # Original array changed

d=a[1:3].copy()
d[0]=9 
print(d)
print(a)  # Original array unchanged

e=a[[1,3]]
print(e)  # Fancy indexing
print(a)  # Original array unchanged

print(b.T)