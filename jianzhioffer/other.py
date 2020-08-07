##############面试题16：数值的整数次方###################################
#给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
# -*- coding:utf-8 -*-
class Solution16(object):
    def Power(self, base, exponent):
        '''
        :param base:
        :param exponent: int
        :return:
        '''
        if exponent == 1:
            return base
        if exponent == 0:
            return 1
        if exponent < 0:
            return 1 / self.Power(base,-exponent)

        res = self.Power(base,exponent >> 1)
        if exponent % 2 == 0:
            res *= res
        else:
            res = res * res * base
        return res

###############21：调整数组使得奇数位于偶数前面#######################################
# 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，
# 并保证奇数和奇数，偶数和偶数之间的相对位置不变。
# 输入：[1,2,3,4,5,6]
# 输出：[1,3,5,2,4,6]
from collections import deque
class Solution21(object):
    def reOrderArray(self, array):
        new_arr = deque()
        n = len(array)
        for i in range(n):
            if array[n-i-1] % 2 == 1:
                new_arr.appendleft(array[n-i-1])
            if array[i] % 2 == 0:
                new_arr.append(array[i])
        return list(new_arr)


############41：数据流中的中位数################################################
# 中位数是有序列表中间的数。如果列表长度是偶数，中位数则是中间两个数的平均值。
# 例如，
# [2,3,4] 的中位数是 3
# [2,3] 的中位数是 (2 + 3) / 2 = 2.5
# 设计一个支持以下两种操作的数据结构：
# void addNum(int num) - 从数据流中添加一个整数到数据结构中。
# double findMedian() - 返回目前所有元素的中位数。
# 示例：
# addNum(1)
# addNum(2)
# findMedian() -> 1.5
# addNum(3)
# findMedian() -> 2
# 进阶:
# 如果数据流中所有整数都在 0 到 100 范围内，你将如何优化你的算法？
# 如果数据流中 99% 的整数都在 0 到 100 范围内，你将如何优化你的算法？

import heapq


class MedianFinder(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.min_heap = []
        self.max_heap = []

    def addNum(self, num):
        """
        :type num: int
        :rtype: None
        """

        if len(self.min_heap) < len(self.max_heap):
            heapq.heappush(self.min_heap,num)
        else:
            heapq.heappush(self.max_heap, -num)

    def findMedian(self):
        """
        :rtype: float
        """
        if (len(self.min_heap) + len(self.max_heap)) % 2 == 0:
        #     return (heapq.heappop(self.min_heap) - heapq.heappop(self.max_heap)) / 2
        # return - heapq.heappop(self.max_heap)
            return (self.min_heap[0] - self.max_heap[0]) / 2
        return -self.max_heap[0]

#丑数
def getUglyNumber(n):
    cnt = 0
    number = 0
    map = set()
    map.add(1)
    def isUgly(num):
        while num % 2 == 0:
            num = num / 2
            if num in map:
                return True
        while num % 3 == 0:
            num = num / 3
            if num in map:
                return True
        while num % 5 == 0:
            num = num / 5
            if num in map:
                return True
        return False
    while(cnt < n):
        number += 1
        if isUgly(number):
            map.add(number)
            cnt += 1
    return number

print(getUglyNumber(150))


