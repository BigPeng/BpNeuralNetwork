import sys
import random
data = []
for line in sys.stdin:
	data.append(line.strip())
random.shuffle(data)
for line in data:
	print line
	
