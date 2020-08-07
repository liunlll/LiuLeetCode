# a = [3,4,2,2,3]
# b = [1,2,3,4,5]
#hash_dict = dict.fromkeys([x for x in range(5)])
#print(hash_dict)
# hash_dict = {x:y for x,y in zip(b,a)}
# print(hash_dict)
# hash_dict['a'] = hash_dict.get('a',0) + 1
# print(hash_dict)

# hash_map = dict.fromkeys(a)
# res = []
# for i in b:
#     if i not in hash_map.keys():
#         res.append(i)
# print(res)
# nums = b
# hash_map = {}
# target = 4
#
# hash_map = {id(x):index for x,index in enumerate(b)}
#
# for index,num in enumerate(nums):
#     j = hash_map.get(id(target - num))
#     if j is not None and j != index:
#         print([index,j])


##################两数之和###################
def twoSum(nums,target):
    hashmap = {x:i for i,x in enumerate(nums)}
    print(hashmap)
    for i,num in enumerate(nums):
        another_num = target - num
        j = hashmap.get(another_num)
        print(i)
        print(j)
        if j is not None and i != j:
            return [i,j]

##################两数之和二#################
def twoSum_v2(nums,target):
    hashmap = {}
    for i,num in enumerate(nums):
        if (target - num) in hashmap:
            return [i,hashmap[target - num]]
        hashmap[num] = i

# nums = [2,3,3,1]
# target = 6
# print(twoSum_v2(nums,target))

##################三数之和######################
def threeSum(nums):
    if len(nums) < 3:
        return []
    res = set()
    nums.sort()
    for i,num1 in enumerate(nums[:-2]):
        if i >= 1 and nums[i - 1] == num1:
            continue
        hashmap = {}
        for num2 in nums[i+1:]:
            num3 = 0 - num1 - num2
            if num3 in hashmap:
                res.add((num1,num2,num3))
            else:
                hashmap[num2] = 1
    return list(map(list,res))

nums = [2,3,3,4,-6]
print(threeSum(nums))





