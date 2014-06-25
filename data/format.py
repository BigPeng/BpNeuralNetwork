
from sys import stdin
def format(x):
	if x >= avg:
		return 1
	return 0
avg = 0
for line in stdin:
	attrs,lable = line.strip().rsplit(',',1)
	pixels = [int(attr) for attr in attrs.split(',')]
	avg = sum(pixels)*1.0/len(pixels)
	avgPixels = map(format,pixels)
	s = ''
	for each in avgPixels:
		s += str(each)+','
	s += lable
	print s