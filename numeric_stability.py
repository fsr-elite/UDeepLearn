x = 1.0e9
y = 1.0e-6

sumx = x

for i in range(0, 1000000):
	sumx = sumx + y

sumx_sub = sumx - x

print(sumx_sub)