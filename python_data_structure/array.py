##########################################################################################
# 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
# 示例:
# 输入: [0,1,0,3,12]
# 输出: [1,3,12,0,0]
# 说明:
# 必须在原数组上操作，不能拷贝额外的数组。
# 尽量减少操作次数。
def moveZeroes(nums):
    """
    :type nums: List[int]
    :rtype: None Do not return anything, modify nums in-place instead.
    """
    pl,f = 0,0
    for pl in range(len(nums)):
        if nums[pl] == 0:
            if f == 0:
                pr = pl
                f =1
            while(nums[pr] == 0):
                pr = pr+1
                if pr >= len(nums):
                    return nums
            nums[pl],nums[pr] = nums[pr],nums[pl]
    return nums


####################################3. 无重复字符的最长子串######################
# 给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。
# 示例 1:
# 输入: "abcabcbb"
# 输出: 3
# 解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
# 示例 2:
# 输入: "bbbbb"
# 输出: 1
# 解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
# 示例 3:
# 输入: "pwwkew"
# 输出: 3
# 解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
#      请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
class Solution3(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        lp,rp = 0,0
        res = ""
        max_len = 0
        window = set()
        while rp < len(s):
            while rp < len(s) and s[rp] not in window:
                window.add(s[rp])
                rp +=1
            if rp - lp > max_len:
                max_len = rp - lp
                res = s[lp:rp]
            while rp < len(s) and s[rp] in window:
                window.remove(s[lp])
                lp += 1
        return max_len


#########################239. 滑动窗口最大值#############################
# 给定一个数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。
# 滑动窗口每次只向右移动一位。
# 返回滑动窗口中的最大值。
# 示例:
# 输入: nums = [1,3,-1,-3,5,3,6,7], 和 k = 3
# 输出: [3,3,5,5,6,7]
# 解释:
#   滑动窗口的位置                最大值
# ---------------               -----
# [1  3  -1] -3  5  3  6  7       3
#  1 [3  -1  -3] 5  3  6  7       3
#  1  3 [-1  -3  5] 3  6  7       5
#  1  3  -1 [-3  5  3] 6  7       5
#  1  3  -1  -3 [5  3  6] 7       6
#  1  3  -1  -3  5 [3  6  7]      7

class Solution239(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        if not nums:
            return []
        res,window = [],[]
        for i,num in enumerate(nums):
            if i >= k and window[0] <= i-k:
                window.pop(0) #防止队列元素个数大于三

            while window and nums[window[-1]] < num:
                window.pop()   #维护队列
            window.append(i)
            if i >= k-1:
                res.append(nums[window[0]]) #最大值永远是队列第一个值
        return res

# s239 = Solution239()
# print(s239.maxSlidingWindow([1,3,-1,-3,5,3,6,7],3))

#################128. 最长连续序列
# 给定一个未排序的整数数组，找出最长连续序列的长度。
# 要求算法的时间复杂度为 O(n)。
# 示例:
# 输入: [100, 4, 200, 1, 3, 2]
# 输出: 4
# 解释: 最长连续序列是 [1, 2, 3, 4]。它的长度为 4。

class Solution128(object):
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums_set = set(nums)
        longest = 0
        for x in nums_set:
            if x - 1 not in nums_set:
                cur_x = x
                cur_len = 1
                while cur_x + 1 in nums_set:
                    cur_x += 1
                    cur_len += 1
                longest = max(cur_len,longest)
        return longest

#############################################268. 缺失数字###########################################
# 给定一个包含 0, 1, 2, ..., n 中 n 个数的序列，找出 0 .. n 中没有出现在序列中的那个数。
# 示例 1:
# 输入: [3,0,1]
# 输出: 2
# 示例 2:
# 输入: [9,6,4,2,3,5,7,0,1]
# 输出: 8
# 说明:
# 你的算法应具有线性时间复杂度。你能否仅使用额外常数空间来实现?
class Solution268(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        sum_ = n
        for i,x in enumerate(nums):
            sum_ = sum_ + i - x
        return sum_

# s268 = Solution268()
# print(s268.missingNumber([9,6,4,2,3,5,7,0,1]))
