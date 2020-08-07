#########回溯算法###############################
##########################模板
# def backtrack(self, choiceList, track, answer):
#     '''
#     :param choiceList: 可选择的列表
#     :param track: 已经选择了的路径，决策路径，已经做出的一些列的选择
#     :param answer:用来存储符合条件的结果
#     :return:
#     '''
#     if track is Ok:
#         answer.add(track)
#     for choice in choiceList:
#         #choose过程
#         #把choice 加入到track
#         #把choice从choiceList中移出来
#         backtrack(choiceList,track,answer)
#         #unchoose过程
#         #把choice 移出 track
#         #把choice加入到choiceList中
#########################################46. 全排列#################
# 给定一个没有重复数字的序列，返回其所有可能的全排列。
# 示例:
# 输入: [1,2,3]
# 输出:
# [
#   [1,2,3],
#   [1,3,2],
#   [2,1,3],
#   [2,3,1],
#   [3,1,2],
#   [3,2,1]
# ]
class Solution46(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        answer = []
        self.visit = {}
        self.backtrack(nums,[],answer)
        print(answer)

    def backtrack(self,choiceList,track,answer):
        '''
        :param choiceList: 可选择的列表
        :param track: 已经选择了的路径，决策路径，已经做出的一些列的选择
        :param answer:用来存储符合条件的结果
        :return:
        '''
        if len(track) >= 3:
            answer.append(track[:])
            print(track)
            return

        for i,choice in enumerate(choiceList):
            if self.visit.get(choice,0):
                continue
            self.visit[choice] = True
            track.append(choice)
            self.backtrack(choiceList,track,answer)
            track.pop()
            self.visit[choice] = False


#############################784. 字母大小写全排列###################################
# 给定一个字符串S，通过将字符串S中的每个字母转变大小写，我们可以获得一个新的字符串。返回所有可能得到的字符串集合。
# 示例:
# 输入: S = "a1b2"
# 输出: ["a1b2", "a1B2", "A1b2", "A1B2"]
# 输入: S = "3z4"
# 输出: ["3z4", "3Z4"]
# 输入: S = "12345"
# 输出: ["12345"]
# 注意：
# S 的长度不超过12。
# S 仅由数字和字母组成。
class Solution784(object):
    def letterCasePermutation(self, S):
        """
        :type S: str
        :rtype: List[str]
        """
        self.res = []
        self._dfs(0,S,[])
        return self.res
    def _dfs(self,index,s,path):
        if index == len(s):
            self.res.append(''.join(path))
            return
        if '0' <= s[index] <= '9':
            path.append(s[index])
            self._dfs(index + 1,s,path)
            path.pop()
        else:
            for i in [s[index].lower(),s[index].upper()]:
                path.append(i)
                self._dfs(index + 1,s,path)
                path.pop()

######################################22. 括号生成##################################
# 给出 n 代表生成括号的对数，请你写出一个函数，使其能够生成所有可能的并且有效的括号组合。
# 例如，给出 n = 3，生成结果为：
# [
#   "((()))",
#   "(()())",
#   "(())()",
#   "()(())",
#   "()()()"
# ]
class Solution22(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        self.res = []
        self._backtrack(n,0,0,"")
        return self.res

    def _backtrack(self,n,left,right,path):
        if left == n and right == n:
            self.res.append(path)
            return
        if left < n:
            path = path + '('
            self._backtrack(n, left + 1,right,path)
            path = path[:-1]
        if right < n and right < left:
            path = path + ')'
            self._backtrack(n, left,right+1,path)
            path = path[:-1]
    # def _backtrack(self,n,layer,path):
    #     if layer >= 2*n:
    #         self.res.append(path)
    #         return
        # for i in ["(",")"]:
        #     path = path + i
        #     self._backtrack(n,layer+1,path)
        #     path = path[:-1]



#############################################78. 子集######################################
# 给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。
# 说明：解集不能包含重复的子集。
# 示例:
# 输入: nums = [1,2,3]
# 输出:
# [
#   [3],
#   [1],
#   [2],
#   [1,2,3],
#   [1,3],
#   [2,3],
#   [1,2],
#   []
# ]
class Solution78(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        self.res = []
        self.visit = {}
        self._backtrack(nums,0,[])
        return self.res
    def _backtrack(self,nums,layer,path):
        self.res.append(path[:])
        if layer >= len(nums):
        #    self.res.append(path[:])
            return
        for i in range(layer,len(nums)):
            #path.append(nums[i])
            self._backtrack(nums,i+1,path + [nums[i]])
            #path.pop()

#

#########################################39. 组合总和#############################################
# 给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
# candidates 中的数字可以无限制重复被选取。
# 说明：
# 所有数字（包括 target）都是正整数。
# 解集不能包含重复的组合。 
# 示例 1:
# 输入: candidates = [2,3,6,7], target = 7,
# 所求解集为:
# [
#   [7],
#   [2,2,3]
# ]
# 示例 2:
# 输入: candidates = [2,3,5], target = 8,
# 所求解集为:
# [
#   [2,2,2,2],
#   [2,3,3],
#   [3,5]
# ]
class Solution39(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        self.res = []
        self._trackback(candidates,target,0,[],0)
        return self.res
    def _trackback(self,candidates,target,index,path,sum_):
        if sum_ == target:
            self.res.append(path[:])
            return
        if sum_ > target:
            return
        for i in range(index,len(candidates)):
            self._trackback(candidates,target,i,path + [candidates[i]],sum_ + candidates[i])

################################ 40. 组合总和 II#################################
# 给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
# candidates 中的每个数字在每个组合中只能使用一次。
# 说明：
# 所有数字（包括目标数）都是正整数。
# 解集不能包含重复的组合。 
# 示例 1:
# 输入: candidates = [10,1,2,7,6,1,5], target = 8,
# 所求解集为:
# [
#   [1, 7],
#   [1, 2, 5],
#   [2, 6],
#   [1, 1, 6]
# ]
# 示例 2:
# 输入: candidates = [2,5,2,1,2], target = 5,
# 所求解集为:
# [
#   [1,2,2],
#   [5]
# ]
class Solution40(object):
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        self.res = []
        candidates.sort()
        def backtrack(candidates,target,index,path,sum_):
            # if sum_ > target:
            #     return
            if sum_ == target:
                self.res.append(path[:])
                return
            for i in range(index+1,len(candidates)):
                if sum_+candidates[i] > target:
                    break
                if i > index + 1 and candidates[i] == candidates[i-1]:
                    continue
                backtrack(candidates,target,i,path + [candidates[i]],sum_ + candidates[i])
        backtrack(candidates,target,-1,[],0)
        return self.res

candidates = [10,1,2,7,6,1,5]
target = 8
s40 = Solution40()
print(s40.combinationSum2(candidates,target))

#################17. 电话号码的字母组合##########################
# 给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。
# 给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。
# phone = {'2': ['a', 'b', 'c'],
#          '3': ['d', 'e', 'f'],
#          '4': ['g', 'h', 'i'],
#          '5': ['j', 'k', 'l'],
#          '6': ['m', 'n', 'o'],
#          '7': ['p', 'q', 'r', 's'],
#          '8': ['t', 'u', 'v'],
#          '9': ['w', 'x', 'y', 'z']}
# 示例:
#
# 输入："23"
# 输出：["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
# 说明:
# 尽管上面的答案是按字典序排列的，但是你可以任意选择答案输出的顺序。
class Solution17(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        self.phone = {'2': ['a', 'b', 'c'],
         '3': ['d', 'e', 'f'],
         '4': ['g', 'h', 'i'],
         '5': ['j', 'k', 'l'],
         '6': ['m', 'n', 'o'],
         '7': ['p', 'q', 'r', 's'],
         '8': ['t', 'u', 'v'],
         '9': ['w', 'x', 'y', 'z']}
        self.res = []
        self._backtrack(digits,0,'')
        return self.res

    def _backtrack(self,digits,index,path):
        if index >= len(digits):
            self.res.append(path)
            return
        for c in self.phone[digits[index]]:
            self._backtrack(digits,index+1,path + c)


# #################################79. 单词搜索#############################
# 给定一个二维网格和一个单词，找出该单词是否存在于网格中。
# 单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。
# 示例:
# board =
# [
#   ['A','B','C','E'],
#   ['S','F','C','S'],
#   ['A','D','E','E']
# ]
# 给定 word = "ABCCED", 返回 true.
# 给定 word = "SEE", 返回 true.
# 给定 word = "ABCB", 返回 false.
class Solution79(object):
    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        if len(word) == 0 or len(board[0]) == 0:
            return False
        self.res = []
        self.visit = set()
        self.n = len(board)
        self.m = len(board[0])
        self.f = False
        for i in range(self.n):
            for j in range(self.m):
                if board[i][j] == word[0]:
                    self.visit.add((i,j))
                    self._trackback(board,word,i,j,1,board[i][j])
                    self.visit.remove((i,j))
        return self.f

    def _trackback(self,board,word,i,j,index,path):
        if self.f:
            return
        if len(path) >= len(word):
            self.res.append(path)
            self.f = True
            return
        for off_i,off_j in [(-1,0),(0,-1),(1,0),(0,1)]:
            tmp_i, tmp_j = i + off_i, j + off_j
            if 0<= tmp_i < self.n and 0<= tmp_j <self.m:
                if (tmp_i,tmp_j) not in self.visit:
                    if board[tmp_i][tmp_j] == word[index]:
                        self.visit.add((tmp_i,tmp_j))
                        self._trackback(board,word,tmp_i,tmp_j,index+1,path+board[tmp_i][tmp_j])
                        self.visit.remove((tmp_i, tmp_j))

###########################################51. N皇后###########################
# n 皇后问题研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。
# 给定一个整数 n，返回所有不同的 n 皇后问题的解决方案。
# 每一种解法包含一个明确的 n 皇后问题的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。
# 示例:
# 输入: 4
# 输出: [
#  [".Q..",  // 解法 1
#   "...Q",
#   "Q...",
#   "..Q."],
#  ["..Q.",  // 解法 2
#   "Q...",
#   "...Q",
#   ".Q.."]
# ]
# 解释: 4 皇后问题存在两个不同的解法。
class Solution51(object):
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        self.res = []
        self.visit = {"col":[0 for _ in range(n)],"add":[0 for _ in range(2*n)],"dif":[0 for _ in range(2*n)]}
        self._trackback(n,0,0,[])
        return [["."*i + "Q" + "."*(n-i-1) for i in Q] for Q in self.res]
    def _trackback(self,n,i,j,path):
        if len(path) >= n:
            self.res.append(path[:])
            return
        for j in range(n):
            if self.visit["col"][j] or self.visit["add"][i+j] or self.visit["dif"][i-j+n]:
                continue
            self.visit["col"][j] = 1
            self.visit["add"][i+j] = 1
            self.visit["dif"][i-j+n] = 1
            self._trackback(n,i+1,j,path + [j])
            self.visit["col"][j] = 0
            self.visit["add"][i+j] = 0
            self.visit["dif"][i-j+n] = 0

############################329. 矩阵中的最长递增路径##################################
# 给定一个整数矩阵，找出最长递增路径的长度。
# 对于每个单元格，你可以往上，下，左，右四个方向移动。 你不能在对角线方向上移动或移动到边界外（即不允许环绕）。
# 示例 1:
# 输入: nums =
# [
#   [9,9,4],
#   [6,6,8],
#   [2,1,1]
# ]
# 输出: 4
# 解释: 最长递增路径为 [1, 2, 6, 9]。
# 示例 2:
# 输入: nums =
# [
#   [3,4,5],
#   [3,2,6],
#   [2,2,1]
# ]
# 输出: 4
# 解释: 最长递增路径是 [3, 4, 5, 6]。注意不允许在对角线方向上移动。
class Solution329(object):
    def longestIncreasingPath(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: int
        """
        if not matrix:
            return 0
        res = 0
        self.n = len(matrix)
        self.m = len(matrix[0])
        self.res_mem = [[0 for _ in range(self.m)] for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.m):
                res = max(res, self._trackback(matrix,i,j,[(i,j)]))
        return res
    def _trackback(self,matrix,i,j,path):
        if self.res_mem[i][j] > 0:
            return self.res_mem[i][j]
        max_ = 0
        for off_i, off_j in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
            if 0<= i+off_i<self.n and 0<=j+off_j<self.m and matrix[i+off_i][j+off_j] > matrix[i][j]:
                max_ =  max(max_,self._trackback(matrix,i+off_i,j+off_j,path + [(i+off_i,j+off_j)]))
        self.res_mem[i][j] =  max_ + 1
        return self.res_mem[i][j]

    def dp(self,matrix):
        if not matrix or not matrix[0]:
            return 0
        n = len(matrix)
        m = len(matrix[0])
        matrix_list = [(matrix[i][j],i,j) for i in range(n) for j in range(m)]
        dp = [[1 for _ in range(m)] for _ in range(n)]
        matrix_list.sort()
        for index in range(1,len(matrix_list)):
            num,i,j = matrix_list[index]
            for off_i,off_j in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                if 0<= off_i + i <n and 0<= off_j + j <m and matrix[off_i+i][off_j+j] < matrix[i][j]:
                    dp[i][j] = max(dp[i][j],dp[off_i+i][off_j+j] + 1)
        return max([dp[i][j] for i in range(n) for j in range(m)])



# ###########################36. 有效的数独 ##########################
# 判断一个 9x9 的数独是否有效。只需要根据以下规则，验证已经填入的数字是否有效即可。
# 数字 1-9 在每一行只能出现一次。
# 数字 1-9 在每一列只能出现一次。
# 数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。
# 数独部分空格内已填入了数字，空白格用 '.' 表示。
# 示例 1:
# 输入:
# [
#   ["5","3",".",".","7",".",".",".","."],
#   ["6",".",".","1","9","5",".",".","."],
#   [".","9","8",".",".",".",".","6","."],
#   ["8",".",".",".","6",".",".",".","3"],
#   ["4",".",".","8",".","3",".",".","1"],
#   ["7",".",".",".","2",".",".",".","6"],
#   [".","6",".",".",".",".","2","8","."],
#   [".",".",".","4","1","9",".",".","5"],
#   [".",".",".",".","8",".",".","7","9"]
# ]
# 输出: true
# 示例 2:
# 输入:
# [
#   ["8","3",".",".","7",".",".",".","."],
#   ["6",".",".","1","9","5",".",".","."],
#   [".","9","8",".",".",".",".","6","."],
#   ["8",".",".",".","6",".",".",".","3"],
#   ["4",".",".","8",".","3",".",".","1"],
#   ["7",".",".",".","2",".",".",".","6"],
#   [".","6",".",".",".",".","2","8","."],
#   [".",".",".","4","1","9",".",".","5"],
#   [".",".",".",".","8",".",".","7","9"]
# ]
# 输出: false
# 解释: 除了第一行的第一个数字从 5 改为 8 以外，空格内其他数字均与 示例1 相同。
#      但由于位于左上角的 3x3 宫内有两个 8 存在, 因此这个数独是无效的。
# 注：
# 一个有效的数独（部分已被填充）不一定是可解的。
# 只需要根据以上规则，验证已经填入的数字是否有效即可。
# 给定数独序列只包含数字 1-9 和字符 '.' 。
# 给定数独永远是 9x9 形式的。
class Solution36(object):
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        N = len(board)
        row_map = [set() for _ in range(N)]
        col_map = [set() for _ in range(N)]
        x3_map = [[set() for _ in range(3)] for _ in range(3)]
        for i in range(N):
            for j in range(N):
                if board[i][j] != '.':
                    if board[i][j] not in row_map[i]:
                        row_map[i].add(board[i][j])
                    else:
                        return False
                    if board[i][j] not in col_map[j]:
                        col_map[j].add(board[i][j])
                    else:
                        return False
                    if board[i][j] not in x3_map[i//3][j//3]:
                        x3_map[i // 3][j // 3].add(board[i][j])
                    else:
                        return False
        return True

    def _isValid(self,board):
        for i in range(9):
            for j in range(9):
                num = board[i][j]
                if num != '.':
                    board[i][j] = '.'
                    for index in range(9):
                        if board[i][index] == num or board[index][j] == num or board[i//3 * 3 + index // 3][j//3 * 3 + index % 3] == num:
                            return False
                    board[i][j] = num
        return True

#
# board =[["8","3",".",".","7",".",".",".","."],
#         ["6",".",".","1","9","5",".",".","."],
#         [".","9","8",".",".",".",".","6","."],
#         ["8",".",".",".","6",".",".",".","3"],
#         ["4",".",".","8",".","3",".",".","1"],
#         ["7",".",".",".","2",".",".",".","6"],
#         [".","6",".",".",".",".","2","8","."],
#         [".",".",".","4","1","9",".",".","5"],
#         [".",".",".",".","8",".",".","7","9"]]
# s36 = Solution36()
# print(s36._isValid(board))


################################37. 解数独###########################
# 编写一个程序，通过已填充的空格来解决数独问题。
# 一个数独的解法需遵循如下规则：
# 数字 1-9 在每一行只能出现一次。
# 数字 1-9 在每一列只能出现一次。
# 数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。
# 空白格用 '.' 表示。
class Solution37(object):
    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        if not board:
            return False
        self._trackback(board)

    def _trackback(self,board):
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == '.':
                    for num in [str(x) for x in range(1,10)]:
                        if self._isValid(board,i,j,num):
                            board[i][j] = num
                            if self._trackback(board):
                                return True
                            else:
                                board[i][j] = '.'
                    return False
        return True

    def _isValid(self,board,i,j,num):
        for index in range(9):
            if board[i][index] == num or board[index][j] == num or board[i//3 * 3 + index // 3][j//3 * 3 + index % 3] == num:
                return False
        return True

# s37 = Solution37()
# s37.solveSudoku(board)
# print(board)



