#################斐波那契数列#####################
def fib1(n):
    return n if n <= 1 else fib1(n-1) + fib1(n-2)

#递推加上记忆化得到递推
def fib2(n,hash = {}):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    elif n not in hash:
        hash[n] = fib2(n-1) + fib2(n-2)
    return hash[n]


def fib3(n):
    x,y = 1,1
    for _ in range(1,n):
        x,y = y,x+y
    return y
    # a = 0
    # b = 1
    # if n <= 1:
    #     return n
    # if n >= 2:
    #     for i in range(n-1):
    #         b,a = a+b ,b
    # return b

def fib4(n):
    f = [0 for _ in range(n+1)]
    f[0] = 0
    f[1] = 1
    if n >= 2:
        for i in range(2,n+1):
            f[i] = f[i - 1] + f[i - 2]
    return f[n]

##########################走迷宫？#################################
# 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start”）。
# 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。
# 问总共有多少条不同的路径？
# 例如，上图是一个7 x 3 的网格。有多少可能的路径？
# 说明：m 和 n 的值均不超过 100。
# 示例 1:
# 输入: m = 3, n = 2
# 输出: 3
# 解释:
# 从左上角开始，总共有 3 条路径可以到达右下角。
# 1. 向右 -> 向右 -> 向下
# 2. 向右 -> 向下 -> 向右
# 3. 向下 -> 向右 -> 向右
# 示例 2:
# 输入: m = 7, n = 3
# 输出: 28
class Solution62(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        self.m = m
        self.n = n
        self._map = [[-1 for _ in range(self.m)] for _ in range(self.n)]
        return self.count_path(0,0)

    def count_path(self,m,n):
        if m == self.m -1 or n == self.n - 1:
            return 1
        if self._map[n][m] < 0:
            self._map[n][m] = self.count_path(m+1,n) + self.count_path(m,n+1)
        return self._map[n][m]
    def dp(self,m,n):
        dp_map = [[1 for _ in range(m)] for _ in range(n)]
        dp_map[0][0] = 0
        dp_map[0][1] = 1
        dp_map[1][0] = 1
        for i in range(1,m):
            for j in range(1,n):
                dp_map[j][i] = dp_map[j - 1][i] + dp_map[j][i - 1]
        return dp_map[n-1][m-1]

class Solution63(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        m = len(obstacleGrid[0])
        n = len(obstacleGrid)

        if obstacleGrid[0][0] == 1:
            return 0
        obstacleGrid[0][0] = 1
        for i in range(1,m):
            obstacleGrid[0][i] = int(obstacleGrid[0][i - 1] == 1 and obstacleGrid[0][i] == 0)
        for j in range(1,n):
            obstacleGrid[j][0] = int(obstacleGrid[j-1][0] == 1 and obstacleGrid[j][0] == 0)

        for j in range(1,n):
            for i in range(1,m):
                if obstacleGrid[j][i] == 1:
                    obstacleGrid[j][i] = 0
                else:
                    obstacleGrid[j][i] = obstacleGrid[j-1][i] + obstacleGrid[j][i-1]

        print(obstacleGrid)

        return obstacleGrid[n-1][m-1]


###################################################120.三角形最小路径和#################
# 给定一个三角形，找出自顶向下的最小路径和。每一步只能移动到下一行中相邻的结点上。
# 例如，给定三角形：
# [
#      [2],
#     [3,4],
#    [6,5,7],
#   [4,1,8,3]
# ]
# 自顶向下的最小路径和为 11（即，2 + 3 + 5 + 1 = 11）。
# 说明：
# 如果你可以只使用 O(n) 的额外空间（n 为三角形的总行数）来解决这个问题，那么你的算法会很加分。
class Solution120(object):
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        for i in range(1,len(triangle)):
            for j in range(i + 1):
                if j == 0:
                    triangle[i][j] = triangle[i-1][j] + triangle[i][j]
                elif j == i:
                    triangle[i][j] = triangle[i-1][j-1] + triangle[i][j]
                else:
                    triangle[i][j] = min(triangle[i-1][j-1] , triangle[i-1][j]) + triangle[i][j]
        print(triangle)
        return min(triangle[-1])
#注：
# 动态规划
# 状态定义[i,j]这一步，最小值是多少
# 状态方程：      if j == 0:
#                     triangle[i][j] = triangle[i-1][j] + triangle[i][j]
#                 elif j == i:
#                     triangle[i][j] = triangle[i-1][j-1] + triangle[i][j]
#                 else:
#                     triangle[i][j] = min(triangle[i-1][j-1] , triangle[i-1][j]) + triangle[i][j]
# 边界条件：第一个值不用算，就是边界条件


##############################################152题######################################
# 给定一个整数数组 nums ，找出一个序列中乘积最大的连续子序列（该序列至少包含一个数）。
# 示例 1:
# 输入: [2,3,-2,4]
# 输出: 6
# 解释: 子数组 [2,3] 有最大乘积 6。
# 示例 2:
# 输入: [-2,0,-1]
# 输出: 0
# 解释: 结果不能为 2, 因为 [-2,-1] 不是子数组。
class Solution152(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        opt = [[0,0] for _ in range(len(nums))]
        opt[0][0],opt[0][1] = nums[0],nums[0]
        for i in range(1,len(nums)):
            opt[i][0] = max(opt[i-1][0]*nums[i],opt[i-1][1]*nums[i],nums[i])
            opt[i][1] = min(opt[i-1][0]*nums[i],opt[i-1][1]*nums[i],nums[i])
        return max([x[0] for x in opt])

#############################53. 最大子序和#########################################
# 给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
# 示例:
# 输入: [-2,1,-3,4,-1,2,1,-5,4],
# 输出: 6
# 解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
# 进阶:
# 如果你已经实现复杂度为 O(n) 的解法，尝试使用更为精妙的分治法求解。
class Solution53(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0

        for i in range(1,len(nums)):
            nums[i] = max(nums[i],nums[i-1]+nums[i])
        print(nums)
        return max(nums)


#############################123. 买卖股票的最佳时机 III########################
# 给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。
# 设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。
# 注意: 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
# 示例 1:
# 输入: [3,3,5,0,0,3,1,4]
# 输出: 6
# 解释: 在第 4 天（股票价格 = 0）的时候买入，在第 6 天（股票价格 = 3）的时候卖出，这笔交易所能获得利润 = 3-0 = 3 。
#      随后，在第 7 天（股票价格 = 1）的时候买入，在第 8 天 （股票价格 = 4）的时候卖出，这笔交易所能获得利润 = 4-1 = 3 。
# 示例 2:
# 输入: [1,2,3,4,5]
# 输出: 4
# 解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。  
#      注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。  
#      因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。
# 示例 3:
# 输入: [7,6,4,3,1]
# 输出: 0
# 解释: 在这个情况下, 没有交易完成, 所以最大利润为 0。
class Solution123(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        dp = [[[0,0],[0,0]] for _ in range(len(prices))]
        dp[0][0][0] = 0
        dp[0][0][1] = -prices[0]
        dp[0][1][0] = 0
        dp[0][1][1] = -prices[0]
        for i in range(1,len(prices)):
            dp[i][0][0] = max(dp[i-1][0][0],dp[i-1][0][1] + prices[i]) #最多交易一次，且手里没有商品，本来就没有（没有交易过），交易一次（卖掉）
            dp[i][0][1] = max(dp[i-1][0][1],-prices[i]) ##最多交易一次，且手里有商品，交易一次（买了没卖），没有交易过（现在买一个），
            dp[i][1][0] = max(dp[i-1][1][0],dp[i-1][1][1] + prices[i]) #
            dp[i][1][1] = max(dp[i-1][1][1],dp[i-1][0][0] - prices[i]) #
        print(dp)
        return dp[len(prices)-1][1][0]



#####################################300最长上升子序列###################
# 给定一个无序的整数数组，找到其中最长上升子序列的长度。
# 示例:
# 输入: [10,9,2,5,3,7,101,18]
# 输出: 4
# 解释: 最长的上升子序列是 [2,3,7,101]，它的长度是 4。
# 说明:
# 可能会有多种最长上升子序列的组合，你只需要输出对应的长度即可。
# 你算法的时间复杂度应该为 O(n2) 。
class Solution300(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        dp = [0 for _ in range(len(nums))]
        dp[0] = 1
        for i in range(1,len(nums)):
            max_lis = 0
            for j in range(i):
                if nums[i] > nums[j] and dp[j] > max_lis:
                    max_lis = dp[j]
            dp[i] = max_lis + 1
        print(dp)
        return max(dp)
    def lengthOfLIS2(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        dp = [1 for _ in range(len(nums))]
        dp[0] = 1
        for i in range(1,len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i],dp[j] + 1)
        print(dp)
        return max(dp)





###########################322. 零钱兑换#################################
# 给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。
# # 示例 1:
# # 输入: coins = [1, 2, 5], amount = 11
# # 输出: 3
# # 解释: 11 = 5 + 5 + 1
# # 示例 2:
# # 输入: coins = [2], amount = 3
# # 输出: -1
class Solution322(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        dp = [amount + 1 for _ in range(amount + 1)]
        dp[0] = 0
        for i in range(1,amount+1):
            for j in coins:
                if j <= i:
                    dp[i] = min(dp[i],dp[i-j] + 1)
        if dp[amount] > amount:
            return -1
        return dp[amount]


##################################72. 编辑距离##################################################
# 给定两个单词 word1 和 word2，计算出将 word1 转换成 word2 所使用的最少操作数。
# 你可以对一个单词进行如下三种操作：
# 插入一个字符
# 删除一个字符
# 替换一个字符
# 示例 1:
# 输入: word1 = "horse", word2 = "ros"
# 输出: 3
# 解释:
# horse -> rorse (将 'h' 替换为 'r')
# rorse -> rose (删除 'r')
# rose -> ros (删除 'e')
# 示例 2:
# 输入: word1 = "intention", word2 = "execution"
# 输出: 5
# 解释:
# intention -> inention (删除 't')
# inention -> enention (将 'i' 替换为 'e')
# enention -> exention (将 'n' 替换为 'x')
# exention -> exection (将 'n' 替换为 'c')
# exection -> execution (插入 'u')
class Solution72(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        n = len(word1)
        m = len(word2)
        dp = [[0 for _ in range(m+1)] for _ in range(n+1)]

        for i in range(n+1):
            dp[i][0] = i
        for j in range(m+1):
            dp[0][j] = j

        for i in range(1,n+1):
            for j in range(1,m+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j],dp[i-1][j-1],dp[i][j-1]) + 1

        return dp[n][m]

#################################剪绳子################################
# 问题:现有一根长度为N的绳子，需要你剪成M段，使M段的乘积最大。（其中M、N都为整数，剪成的每段长度也为整数，N已知，M未知）
# 例如 绳子长度N=8 剪成M=3，数值为别为2,3,3，则乘积最大为 2*3*3=18。
class Solutionj14:
    def maxProductAfterCutting_solution(self, number):
        if number < 2:
            return 0
        if number == 2:
            return 1
        if number == 3:
            return 2
        dp = [0 for _ in range(number+1)]
        dp[1] = 1
        dp[2] = 2
        dp[3] = 3

        for i in range(4,number+1):
            max = 0
            for j in range(1,i//2+1):
                if max < dp[j]*dp[i-j]:
                    max = dp[j]*dp[i-j]
            dp[i] = max

        print(dp)
        return dp[number]


# #礼物最大值
# 在一个二维矩阵中，每一个数字代表这个位置的礼物的价值，从左上角每次往右或者往下一格，
# 到右下角，请问经过的礼物的最大值是多少
# 例:
# grid = [[1,10,3,8],
#         [12,2,9,6],
#         [5,7,4,11],
#         [3,7,16,5]]
# 路线(1,12,5,7,7,16,5)为53
def getMaxValue(grid):
    n = len(grid)
    m = len(grid[0])
    if m == 0:
        return
    if n == 1 and m == 1:
        return grid[0][0]
    for i in range(1,n):
        grid[i][0] = grid[i-1][0] + grid[i][0]
    for j in range(1,m):
        grid[0][j] = grid[0][j-1] + grid[0][j]
    for i in range(1,n):
        for j in range(1,m):
            grid[i][j] = max(grid[i-1][j],grid[i][j-1]) + grid[i][j]
    return grid[-1][-1]

############最长公共子序列（LCS）###################################
#求两个字符串的最长公共字序列的长度
#例子：
# 输入 str1 = "abcde" , str2 = "ace"
# 输出是 3
def lcs(str1,str2):
    n,m = len(str1),len(str2)
    dp = [[0 for _ in range(m + 1)] for _ in range(n+1)]
    for i in range(n):
        dp[i][0] = 0
    for j in range(m):
        dp[0][j] = 0
    for i in range(n):
        for j in range(m):
            if str1[i] == str2[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i+1][j],dp[i][j+1])
    return dp


# ###################################1155. 掷骰子的N种方法################################
# 这里有 d 个一样的骰子，每个骰子上都有 f 个面，分别标号为 1, 2, ...f。
# 我们约定：掷骰子的得到总点数为各骰子面朝上的数字的总和。
# 如果需要掷出的总点数为 target，请你计算出有多少种不同的组合情况（所有的组合情况总共有 f^d 种），模 10^9 + 7 后返回。
# 示例 1：
# 输入：d = 1, f = 6, target = 3
# 输出：1
# 示例 2：
# 输入：d = 2, f = 6, target = 7
# 输出：6
# 示例 3：
# 输入：d = 2, f = 5, target = 10
# 输出：1
# 示例 4：
# 输入：d = 1, f = 2, target = 3
# 输出：0
# 示例 5：
# 输入：d = 30, f = 30, target = 500
# 输出：222616187
# 提示：
# 1 <= d, f <= 30
# 1 <= target <= 1000
class Solution1155(object):
    def numRollsToTarget(self, d, f, target):
        """
        :type d: int
        :type f: int
        :type target: int
        :rtype: int
        """
        dp = [[0 for j in range(target)] for i in range(d)]
        for j in range(f):
            if j < target:
                dp[0][j] = 1
        for i in range(1,d):
            for j in range(target):
                for k in range(1,f+1):
                    if j - k >= 0:
                        dp[i][j] += dp[i-1][j-k]
        return dp[-1][-1]


#########################121. 买卖股票的最佳时机#################################
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
        dp = [[0,0] for _ in range(len(prices))]
        dp[0][0] = 0
        dp[0][1] = -prices[0]
        for i in range(1,len(prices)):
            dp[i][0] = max(dp[i-1][0],dp[i-1][1]+prices[i])
            dp[i][1] = max(dp[i-1][1],0-prices[i]) #原来没有交易过，现在交易一次
        print(dp)

########################96. 不同的二叉搜索树###########################
# 给定一个整数 n，求以 1 ... n 为节点组成的二叉搜索树有多少种？
# 示例:
# 输入: 3
# 输出: 5
# 解释:
# 给定 n = 3, 一共有 5 种不同结构的二叉搜索树:
#    1         3     3      2      1
#     \       /     /      / \      \
#      3     2     1      1   3      2
#     /     /       \                 \
#    2     1         2                 3

class Solution96(object):
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp = [0 for _ in range(n+1)]
        dp[0],dp[1] = 1,1
        for j in range(2,n+1):
            for i in range(1,j+1):
                dp[j] = dp[j] + dp[i-1]*dp[j-i]
        return dp[-1]