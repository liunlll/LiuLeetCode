# 题目描述
# 在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。
# 请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
# -*- coding:utf-8 -*-
class Solution1:
    # array 二维列表
    def Find(self, target, array):
        if array is None:
            return False
        i ,j = 0,len(array[0]) - 1
        while(i < len(array) and j >= 0):
            if target < array[i][j]:
                j = j - 1
            elif target > array[i][j]:
                i = i + 1
            else:
                return True
        return False
# array = [[1,2,8,9],[2,4,8,12],[4,7,10,13],[6,8,11,15]]
# target = 14
# ss = Solution1()
# print(ss.Find(target,array))

#################面试题6，从头到尾打印列表###############################
#-*- coding:utf-8 -*-
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution2:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        # write code here
        res = []
        pre,cur = None,listNode
        while cur:
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp
        while pre:
            res.append(pre.val)
            pre = pre.next
        return res


# head = ListNode(-1000)
# p = head
# for i in range(5):
#     p_new = ListNode(i)
#     p.next = p_new
#     p = p_new
# head = head.next
#
# ss = Solution2()
# print(ss.printListFromTailToHead(head))
#
# def print_list(head):
#     list_arr = []
#     p = head
#     while p is not None:
#         list_arr.append(p.val)
#         p = p.next
#     print("-->".join([str(x) for x in list_arr]))
# print_list(head)


# 面试题57
# 小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。但是他并不满足于此,
# 他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。
# 现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列? Good Luck!
#输出所有和为S的连续正数序列。序列内按照从小至大的顺序，序列间按照开始数字从小到大的顺序
# -*- coding:utf-8 -*-
class Solution57:
    def FindContinuousSequence(self, tsum):
        res = []
        if tsum < 3:
            return res
        small,big,middle = 1,2,(1+tsum)//2
        while small < middle:
            curSum = (small + big)*(big - small + 1) // 2
            if curSum == tsum:
                res.append([x for x in range(small,big+1)])
                small += 1
            elif curSum < tsum:
                big += 1
            else:
                small += 1
        return res

s = Solution57()
print(s.FindContinuousSequence(15))

##############################
# 题目描述
# 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵：
# 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.
# -*- coding:utf-8 -*-
class Solution:
    # matrix类型为二维列表，需要返回列表
    def printMatrix(self, matrix):
        # write code here
        if not matrix:
            return None
        rows = len(matrix)
        cols = len(matrix[0])
        start = 0
        result = []
        while rows > 2 * start and cols > 2 * start:
            endx = rows - 1 - start
            endy = cols - 1 - start
            for i in range(start, endy + 1):
                result.append(matrix[start][i])
            if start < endx:
                for i in range(start + 1, endx + 1):
                    result.append(matrix[i][endy])
            if start < endx and start < endy:
                for i in range(endy - 1, start - 1, -1):
                    result.append(matrix[endx][i])
            if start < endx - 1 and start < endy:
                for i in range(endx - 1, start, -1):
                    result.append(matrix[i][start])
            start += 1
        return result


