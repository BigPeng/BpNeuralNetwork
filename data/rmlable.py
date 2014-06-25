import sys
for line in sys.stdin:
	attrs,lable = line.strip().rsplit(',',1)
	print attrs
