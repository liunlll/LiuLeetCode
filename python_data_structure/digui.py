def knap_rec(weight,wlist,n):
    if weight == 0:
        return True
    if weight < 0 or (weight > 0 and n < 1):
        return False
    if knap_rec(weight - wlist[n-1],wlist,n-1):
        print('2')
        print("{},{},{}".format(weight - wlist[n-1],wlist,n-1))
        print("Item" + str(n) + ":",wlist[n-1])
        return True
    if knap_rec(weight,wlist,n-1):
        print('3')
        print("{},{},{}".format(weight,wlist,n-1))
        return True
    else:return False


###############################################################################################
# 给定一个二叉树和一个目标和，判断该树中是否存在根节点到叶子节点的路径，这条路径上所有节点值相加等于目标和。
# 说明: 叶子节点是指没有子节点的节点。
# 示例: 
# 给定如下二叉树，以及目标和 sum = 22，
#               5
#              / \
#             4   8
#            /   / \
#           11  13  4
#          /  \      \
#         7    2      1
# 返回 true, 因为存在目标和为 22 的根节点到叶子节点的路径 5->4->11->2
class TreeNode(object):
    def __init__(self,val,left=None,right=None):
        self.val = val
        self.left = left
        self.right = right

####################定义一棵树####################
#           1
#          / \
#         2   3
#        / \  /
#       4  5 7
#         /   \
#        6     8
tree = TreeNode(1,TreeNode(2,TreeNode(4),TreeNode(5,TreeNode(6))),TreeNode(3,TreeNode(7,right=TreeNode(8))))

def hasPathSum(root):
    x = root
    def dfs_tree(x,path = []):
        if not x:
            return
        path.append(x.val)
        if not x.left and not x.right:
            print(sum(path))
        dfs_tree(x.left,path)
        dfs_tree(x.right,path)
        path.pop()
    dfs_tree(x)


################################################################################################
# 给出 n 代表生成括号的对数，请你写出一个函数，使其能够生成所有可能的并且有效的括号组合。
# 例如，给出 n = 3，生成结果为：
# [
#   "((()))",
#   "(()())",
#   "(())()",
#   "()(())",
#   "()()()"
# ]
class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        self.res = []
        self._gen(0,0,n,"")
        return self.res
    def _gen(self,left,right,n,result):
        if left+right >= 2*n:
            self.res.append(result)
            return
        self._gen(left + 1,right,n,result + '(')
        self._gen(left,right+1,n,result + ")")
# a = Solution()
# print(a.generateParenthesis(2))

###################################全排列#####################
class Solution2(object):
    def quanpailie(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        res = []
        result = []
        def _pl(l,n):
            if l >= n:
                res.append(result)
                return
            # result.append(1)
            # self._pl(l + 1, n, result)
            # result.append(2)
            # self._pl(l + 1, n, result)
            # result.append(3)
            # self._pl(l + 1, n, result)
            for i in range(2):
                result.append(i)
                _pl(l+1,n)
                result.pop()
        _pl(0,n)
        return res


###########################################690########################################################
# 给定一个保存员工信息的数据结构，它包含了员工唯一的id，重要度 和 直系下属的id。
# 比如，员工1是员工2的领导，员工2是员工3的领导。他们相应的重要度为15, 10, 5。那么员工1的数据结构是[1, 15, [2]]，员工2的数据
# 结构是[2, 10, [3]]，员工3的数据结构是[3, 5, []]。注意虽然员工3也是员工1的一个下属，但是由于并不是直系下属，因此没有体现在员工1的数据结构中。
# 现在输入一个公司的所有员工信息，以及单个员工id，返回这个员工和他所有下属的重要度之和。
# 示例 1:
# 输入: [[1, 5, [2, 3]], [2, 3, []], [3, 3, []]], 1
# 输出: 11
# 解释:
# 员工1自身的重要度是5，他有两个直系下属2和3，而且2和3的重要度均为3。因此员工1的总重要度是 5 + 3 + 3 = 11。
# 注意:
# 一个员工最多有一个直系领导，但是可以有多个直系下属
# 员工数量不超过2000。

# Employee info
class Employee(object):
    def __init__(self, id, importance, subordinates):
        # It's the unique id of each node.
        # unique id of this employee
        self.id = id
        # the importance value of this employee
        self.importance = importance
        # the id of direct subordinates
        self.subordinates = subordinates

class Solution690(object):
    def getImportance(self, employees, id):
        """
        :type employees: Employee
        :type id: int
        :rtype: int
        """
        self.res = 0
        hashmap = {}
        for employee in employees:
            hashmap[employee.id] = [employee.importance, employee.subordinates]

        def dfs(subs):
            if subs is None:
                return
            for sub in subs:
                self.res += hashmap[sub][0]
                dfs(hashmap[sub][1])

        dfs([id])
        return self.res


#############################################559
# 给定一个 N 叉树，找到其最大深度。
# 最大深度是指从根节点到最远叶子节点的最长路径上的节点总数。
# 例如，给定一个 3叉树 :
#         1
#  3      2       4
# 5   6
# 我们应返回其最大深度，3。
# 说明:
# 树的深度不会超过 1000。
# 树的节点总不会超过 5000。

# Definition for a Node.
class Node(object):
    def __init__(self, val, children):
        self.val = val
        self.children = children
class Solution559(object):
    def maxDepth(self, root):
        """
        :type root: Node
        :rtype: int
        """
        if not root:
            return 0
        self.res = 0
        def dfs(layer,root):
            if not root.children:
                if layer > self.res:
                    self.res = layer
                return
            for child in root.children:
                dfs(layer + 1,child)
        dfs(1,root)
        return self.res


root = Node(1,[Node(3,[Node(5,[]),Node(6,[])]),Node(2,[]),Node(4,[])])

ss = Solution559()
print(ss.maxDepth(root))