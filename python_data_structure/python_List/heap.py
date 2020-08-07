import heapq
a = [15,1,2,4,3]
#线性时间使用floyd算法建立堆，无返回值，直接在堆上操作
heapq.heapify(a)
print(a)
#往堆中加入一个新的值
heapq.heappush(a,0)
print(a)
heapq.heappush(a,-1)
print(a)
heapq.heappush(a,5)
print(a)
#从堆中弹出最小值
print(heapq.heappop(a))
print(heapq.heappop(a))
print(heapq.heappop(a))
print(a)
#弹出最小值，并且元素插入到堆中
print(heapq.heapreplace(a,7))
print(a)
#将元素压入堆中，然后弹出对顶元素
print(heapq.heappushpop(a,0))

# ##############215. 数组中的第K个最大元素############################
# 在未排序的数组中找到第 k 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
# 示例 1:
# 输入: [3,2,1,5,6,4] 和 k = 2
# 输出: 5
# 示例 2:
# 输入: [3,2,3,1,2,4,5,5,6] 和 k = 4
# 输出: 4
# 说明:
# 你可以假设 k 总是有效的，且 1 ≤ k ≤ 数组的长度。
class Solution215(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        min_heap = nums[:k]
        heapq.heapify(min_heap)
        for x in nums[k:]:
            if min_heap[0] < x:
                heapq.heappushpop(min_heap, x)
        return min_heap[0]
    def findKthLargest2(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        return heapq.nlargest(k,nums)[-1]

    def findKthLargest3(self, nums, k):
        self.nums = nums
        lo = 0
        hi = len(self.nums) - 1
        while 1:
            mi = self.partition(lo,hi)
            if mi == len(nums) - k:
                return self.nums[-k]
            elif mi > len(nums) - k:
                hi = mi - 1
            else:
                lo = mi + 1

    def partition(self,lo,hi):
        r = (lo + hi) // 2
        self.nums[lo],self.nums[r] = self.nums[r],self.nums[lo]
        p = self.nums[lo]
        while lo < hi:
            while lo < hi and p <= self.nums[hi]:
                hi -= 1
            self.nums[lo] = self.nums[hi]
            while lo < hi and p >= self.nums[lo]:
                lo += 1
            self.nums[hi] = self.nums[lo]
        self.nums[lo] = p
        return lo

#
# nums = [3,2,3,1,2,4,5,5,6]
# k = 4
# s215 = Solution215()
# print(s215.findKthLargest3(nums,k))
