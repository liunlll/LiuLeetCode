###############################################################################################
# 给出 n 代表生成括号的对数，请你写出一个函数，使其能够生成所有可能的并且有效的括号组合。
# 例如，给出 n = 3，生成结果为：
# [
#   "((()))",
#   "(()())",
#   "(())()",
#   "()(())",
#   "()()()"
# ]
# def generateParenthesis(n):
#     if n == 1:
#         return ["()"]
#     else:
#         return list(set(["(){}".format(x) for x in generateParenthesis(n-1)]
#                         + ["{}()".format(x) for x in generateParenthesis(n-1)]
#                         + ["({})".format(x) for x in generateParenthesis(n-1)]))
# aa = generateParenthesis(4)
ll = []
# def gen(left,right,n,result):
#     if left == n and right == n:
#         ll.append(result)
#         print(result)
#         print(ll)
#         return
#     if left < n:
#         gen(left + 1,right,n,result + "(")
#     if right < n:
#         gen(left,right+1,n,result + ")")
def gen(left,right,n,result):
    if left == n and right == n:
        ll.append(result)
        print(result)
        print(ll)
        return
    if left < n:
        gen(left + 1,right,n,result + "(")
    if right < n and right < left:
        gen(left,right + 1,n,result + ")")

def lengthOfLongestSubstring(s):
    if not s:return 0
    left = 0
    lookup = set()
    n = len(s)
    max_len = 0
    cur_len = 0
    for i in range(n):
        cur_len += 1
        while s[i] in lookup:
            lookup.remove(s[left])
            left += 1
            cur_len -= 1
        if cur_len > max_len:max_len = cur_len
        lookup.add(s[i])
    return max_len

####################################################################################################
# 给你一个字符串 S、一个字符串 T，请在字符串 S 里面找出：包含 T 所有字母的最小子串。
# 示例：
# 输入: S = "ADOBECODEBANC", T = "ABC"
# 输出: "BANC"
# 说明：
# 如果 S 中不存这样的子串，则返回空字符串 ""。
# 如果 S 中存在这样的子串，我们保证它是唯一的答案。
def minWindow(s, t):
    """
    :type s: str
    :type t: str
    :rtype: str
    """

    pl,pr,match,start,minL= 0,0,0,0,float('inf')
    needs = {}
    for c in t:
        needs[c] = needs.get(c,0) + 1
    window = {}

    while pr < len(s):
        if s[pr] in needs:
            window[s[pr]] = window.get(s[pr],0) + 1
            if window[s[pr]] == needs[s[pr]]:
                match += 1
        pr += 1

        while match == len(needs):
            if pr - pl < minL:
                start = pl
                minL = pr - pl
            if s[pl] in needs:
                window[s[pl]] -= 1
                if window[s[pl]] < needs[s[pl]]:
                    match -= 1
            pl += 1
    return s[start:minL + start] if minL != float('inf') else ""

##################################################################################################################
# 给定一个整数数组，你需要寻找一个连续的子数组，如果对这个子数组进行升序排序，那么整个数组都会变为升序排序。
# 你找到的子数组应是最短的，请输出它的长度。
# 示例 1:
# 输入: [2, 6, 4, 8, 10, 9, 15]
# 输出: 5
# 解释: 你只需要对 [6, 4, 8, 10, 9] 进行升序排序，那么整个表都会变为升序排序。
# 说明 :
# 输入的数组长度范围在 [1, 10,000]。
# 输入的数组可能包含重复元素 ，所以升序的意思是<=。
def findUnsortedSubarray(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    diff = [i for i,(x,y) in enumerate(zip(nums,sorted(nums))) if x != y]
    return len(nums) and diff[-1] - diff[0] + 1


def subsets(nums):
    res = []
    n = len(nums)

    def helper(i, tmp):
        res.append(tmp)
        for j in range(i, n):
            tmp2 = tmp + [nums[j]]
            helper(j + 1, tmp2)

    helper(0, [])
    return res


#########################################121. 买卖股票的最佳时机#########################
# 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
# 如果你最多只允许完成一笔交易（即买入和卖出一支股票），设计一个算法来计算你所能获取的最大利润。
# 注意你不能在买入股票前卖出股票。
# 示例 1:
# 输入: [7,1,5,3,6,4]
# 输出: 5
# 解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
#      注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格。
# 示例 2:
# 输入: [7,6,4,3,1]
# 输出: 0
# 解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
class Solution121(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        res = 0
        for i in range(len(prices)-1):
            for j in range(i+1,len(prices)):
                if prices[j] - prices[i] > res:
                    res = prices[j] - prices[i]
        return res

    def maxProfit2(self, prices):
        if not prices:
            return 0
        minprice = prices[0]
        maxprofit = 0
        for i in range(1,len(prices)):
            if prices[i] - minprice > maxprofit:
                maxprofit = prices[i] - minprice
            if prices[i] < minprice:
                minprice = prices[i]
        return maxprofit

############################# 56. 合并区间################################
# 给出一个区间的集合，请合并所有重叠的区间。
# 示例 1:
# 输入: [[1,3],[2,6],[8,10],[15,18]]
# 输出: [[1,6],[8,10],[15,18]]
# 解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
# 示例 2:
# 输入: [[1,4],[4,5]]
# 输出: [[1,5]]
# 解释: 区间 [1,4] 和 [4,5] 可被视为重叠区间。
class Solution56(object):
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        res = []
        intervals = sorted(intervals,key=lambda x : x[0])
        for i in intervals:
            if not res or res[-1][-1] < i[0]:
                res.append(i)
            else:
                res[-1][-1] = max(res[-1][-1],i[-1])
        return res
s56 = Solution56()
ss = [[1,3],[2,6],[8,10],[15,18]]
print(s56.merge(ss))










