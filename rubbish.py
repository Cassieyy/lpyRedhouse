import bisect
import sys
import bisect
import numpy as np

# a = np.array([[1,2,3],[4,5,6]])
# print(a.sum(axis=1))
a = np.array([[1,0],[0,1]])
print(np.nanmean(a))



# a = [1,2,3]
# a.pop(0)
# print(a)



# task = {}
# lines = sys.stdin.readlines()
# n, m = map(int, lines[0].strip().split())
# for line in lines[1:-1]:
#     if not line.strip().split():
#         continue
#     a, b = map(int, line.strip().split())
#     task[a] = max(task.get(a, 0), b)
# arr = sorted(task.keys())
# for i in range(1, len(arr)):
#     if task[arr[i]] < task[arr[i -1]]:
#         task[arr[i]] = task[arr[i -1]]
# skills = map(int, lines[-1].strip().split())
# for skill in skills:
#     if skill in task:
#         print(task[skill])
#     else:
#         ind = bisect.bisect(arr, skill)
#         if ind == 0:
#             print(0)
#         else:
#             print(task[arr[ind -1]])

lines = sys.stdin.readlines()
hashmap = {}
n,m = lines[0].strip().split()
for line in lines[1:-1]:
    a,b = map(int,line.strip().split())
    hashmap[a] = max(hashmap.get(a,0),b)
arr = sorted(hashmap.keys())
for i in range(1,len(arr)):
    if hashmap[arr[i]]<hashmap[arr[i-1]]:
        hashmap[arr[i]] = hashmap[arr[i-1]]
print(hashmap)
skills = map(int,lines[-1].strip().split())
for skill in skills:
    if skill in hashmap:
        print(hashmap[skill])
    else:
        index = bisect.bisect(arr,skill)
        if index ==0:
            print(0)
        else:
            print(hashmap[arr[index-1]])







# import torch
# from torchvision import ops



# import sys
# n = int(sys.stdin.readline().strip())
# point = []
# for i in range(n):
#     point.append(list(map(int, sys.stdin.readline().strip().split())))
# point.sort(key=lambda k:k[1],reverse=True)
 
# res = []
# res.append(point[0])
# for i in  range(1,len(point)):
#     if point[i][0] > res[-1][0]:
#         res.append(point[i])
#     else:
#         continue
# res.sort(key=lambda k:k[0])
# for i in res:
#     print(i[0],i[1])


# if __name__ == "__main__":

    # a = torch.randn((3,4))
    # print(a)
    # print(a.mean())








# import numpy as np 


# print(123)
# a = 1+2
# print(a)

# a = np.array([
#     [1,2,3,4],
#     [2,3,4,5],
#     [3,4,5,6]
# ])

# b = np.array([
#     [0,0,0,0],
#     [0,1,1,0],
#     [0,0,0,0]
# ])
# c = a[b==1]
# print(c)


# a = torch.tensor([
#     [1,2,3,4],
#     [2,3,4,5],
#     [3,4,5,6]
# ])
# b = torch.tensor([
#     [0,0,0,0],
#     [0,1,1,0],
#     [0,0,0,0]
# ])
# c = torch.where(b==1,b,a)
# print(c)

#torch.index_select
# torch.masked_select


# def simple_decorator(f):
#     def wrapper():
#         print("func enter")
#         f()
#         print("func exit")
#     return wrapper

# # @simple_decorator
# # def hello():
# #     print("Hello World!")

# @simple_decorator
# def hello1():
#     print("Hello World!")
# hello1()


