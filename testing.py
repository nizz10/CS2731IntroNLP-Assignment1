############# enumerate
dict = {'a':1, 'b':2, 'c':3}
list = ['a', 'b', 'c']
for index, key in enumerate(dict.keys()):
	print(index, key)
for index, key in enumerate(list):
	print(index, key)

# ############# Using dictionary
# dict = {'a':1, 'b':2, 'c':3}
# print(dict['a'])
# for key in dict:
# 	print(key);
# for key in dict.keys():
# 	if key == 'a':
# 		del dict[key]
# 		dict["ok"] = 1
# print("")
# for key in dict:
# 	print(key);

# a = ['a1', 'a2', 'a3']
# b = ['b1', 'b2']

# print "List:"
# for x, y in [(x, y) for x in a for y in b]:
# 	print x, y
#
# ############## Using pickle
# import pickle
#
# data = {'a':1, 'b':12, 'c':4}
# file = open("tttt.pkl", "wb")
# pickle.dump(data, file)
# file.close()
#
# with open("tttt.pkl", "rb")as f:
# 	b = pickle.load(f)
# 	print(data==b)
#
#
# ############# File writing and reading
# with open("testing1.txt", "w") as f:
# 	f.writelines("Hello\n")
# 	f.writelines("World")
#
# with open("testing2.txt", "w") as f:
# 	f.write("Hello\n")
# 	f.write("World")
#
# with open("testing2.txt", "r") as f:
# 	str = f.readline().rstrip("\n")
# 	print(str)
