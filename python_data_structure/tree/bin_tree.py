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
tree2 = TreeNode(3,TreeNode(5,TreeNode(4)),TreeNode(3,TreeNode(7,right=TreeNode(8))))
#################递归版本的遍历######################
def trav_pre_r(x,res = []):
    if x is None:
        return
    res.append(x.val)
    print(x.val)
    trav_pre_r(x.left)
    trav_pre_r(x.right)
    return res

def trav_in_r(x,res = []):
    if x is None:
        return
    trav_in_r(x.left)
    res.append(x.val)
    trav_in_r(x.right)
    return res
def trav_post_r(x,res = []):
    if x is None:
        return
    trav_post_r(x.left)
    trav_post_r(x.right)
    res.append(x.val)
    return res
###################二叉树的深度优先搜所#############################
#重点是我知道我在哪一层上面
def tree_dfs(node,level,res = []):
    if not node:
        return
    # if len(res) < level + 1:
    #     res.append([])
    res.append((level,node.val))
    tree_dfs(node.left,level + 1,res)
    tree_dfs(node.right, level + 1, res)
    return res
#print(tree_dfs(tree,0))
####################迭代版本的遍历###########################
def trav_pre_i(x,res = []):
    stack = []
    while True:
        while(x):
            res.append(x.val)
            if x.right:
                stack.append(x.right)
            x = x.left
        if stack:
            x = stack.pop()
        else:
            break
    return res


def trav_in_i(x,res = []):
    stack = []
    while True:
        while(x):
            stack.append(x)
            x = x.left
        if not stack:
            break
        x = stack.pop()
        res.append(x.val)
        x = x.right
    return res
################层次遍历#############################
def trav_level(x,res = []):
    queue = []
    queue.append(x)
    while queue:
        x = queue.pop(0)
        res.append(x.val)
        if x.left:
            queue.append(x.left)
        if x.right:
            queue.append(x.right)
    return res



def trav_level2(x,res = []):
    queue = []
    queue.append(x)
    while queue:
        level = []
        for i in range(len(queue)):
            x = queue.pop(0)
            level.append(x.val)
            if x.left:
                queue.append(x.left)
            if x.right:
                queue.append(x.right)
        res.extend(level)
    return res


#################层次遍历知道遍历到第几层####################

def trav_num_level(x,res = []):
    queue = []
    queue.append(x)
    queue.append(None)
    leval = 1
    while queue:
        # q = []
        # for i in queue:
        #     if i != None:
        #         q.append(i.val)
        #     else:q.append(None)
        # print(q)
        x = queue.pop(0)
        if x == None:
            leval += 1
            if not queue:
                break
            queue.append(None)
            continue
        #print("第 {} 层结点 {}".format(leval,x.val))
        res.append((leval,x.val))
        if x.left:
            queue.append(x.left)
        if x.right:
            queue.append(x.right)
    return res

#########################leetcode515#######################
# 515. 在每个树行中找最大值
# 输入:
#
#           1
#          / \
#         3   2
#        / \   \
#       5   3   9
#
# 输出: [1, 3, 9]
# # Definition for a binary tree node.
# # class TreeNode(object):
# #     def __init__(self, x):
# #         self.val = x
# #         self.left = None
# #         self.right = None
def lagestValue(x):
    queue = []
    queue.append(x)
    queue.append(None)
    leval = 1
    res = []
    tmp = []
    while queue:
        x = queue.pop(0)
        if x == None:
            res.append(max(tmp))
            tmp.clear()
            if not queue:
                break
            leval = leval + 1
            queue.append(None)
            continue
        tmp.append(x.val)
        if x.left:
            queue.append(x.left)
        if x.right:
            queue.append(x.right)
    return res


from collections import deque

def largestValues2(root):
    if not root:
        return []
    s = deque()
    s.append(root)
    res = []
    while s:
        tmp_list = []
        print(len(s))
        for i in range(len(s)):
            tmp = s.popleft()
            tmp_list.append(tmp.val)
            if tmp.left:
                s.append(tmp.left)
            if tmp.right:
                s.append(tmp.right)
        res.append(max(tmp_list))
    return res

##########################################################################################
# 给定一个二叉树，根节点为第1层，深度为 1。在其第 d 层追加一行值为 v 的节点。
# 添加规则：给定一个深度值 d （正整数），针对深度为 d-1 层的每一非空节点 N，为 N 创建两个值为 v 的左子树和右子树。
# 将 N 原先的左子树，连接为新节点 v 的左子树；将 N 原先的右子树，连接为新节点 v 的右子树。
# 如果 d 的值为 1，深度 d - 1 不存在，则创建一个新的根节点 v，原先的整棵树将作为 v 的左子树。
# 示例 1:
# 输入:
# 二叉树如下所示:
#        4
#      /   \
#     2     6
#    / \   /
#   3   1 5
# v = 1
# d = 2
# 输出:
#        4
#       / \
#      1   1
#     /     \
#    2       6
#   / \     /
#  3   1   5
#
# 示例 2:
# 输入:
# 二叉树如下所示:
#       4
#      /
#     2
#    / \
#   3   1
# v = 1
# d = 3
# 输出:
#       4
#      /
#     2
#    / \
#   1   1
#  /     \
# 3       1


# ############################二叉树里面的递归的一些题目#######################
# 给定两个二叉树，想象当你将它们中的一个覆盖到另一个上时，两个二叉树的一些节点便会重叠。
# 你需要将他们合并为一个新的二叉树。合并的规则是如果两个节点重叠，那么将他们的值相加作为节点合并后的新值，否则不为 NULL 的节点将直接作为新二叉树的节点。
# 示例 1:
# 输入:
# 	Tree 1                     Tree 2
#           1                         2
#          / \                       / \
#         3   2                     1   3
#        /                           \   \
#       5                             4   7
# 输出:
# 合并后的树:
# 	     3
# 	    / \
# 	   4   5
# 	  / \   \
# 	 5   4   7
#找到边界条件
# 处理根节点
# 处理左孩子，处理右孩子
def mergeTrees(t1,t2):
    if not t1:
        return t2
    elif not t2:
        return t1
    else:
        t1.val = t1.val + t2.val
        t1.left = mergeTrees(t1.left,t2.left)
        t1.right = mergeTrees(t1.right,t2.right)
    return t1
# #####################################################################
# 给定一个二叉树，找出其最小深度。
# 最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
# 说明: 叶子节点是指没有子节点的节点。
# 示例:
# 给定二叉树 [3,9,20,null,null,15,7],
#     3
#    / \
#   9  20
#     /  \
#    15   7
# 返回它的最小深度  2
#####dfs
def minDepthR(root):
    if not root:
        return 0
    else:
        if root.left and root.right:
            return 1 + min(minDepthR(root.left),minDepthR(root.right))
        elif root.left:
            return minDepthR(root.left) + 1
        elif root.right:
            return minDepthR(root.right) + 1
        else:
            return 1 + min(minDepthR(root.left),minDepthR(root.right))


def minDepthDfs(root,level = 0,res = []):
    if not root:
        return
    if root.left is None and root.right is None:
        res.append(level)
    minDepthDfs(root.left, level + 1)
    minDepthDfs(root.right, level + 1)
    return (min(res) + 1,max(res) + 1)

# ##############################################################################################
# 翻转一棵二叉树。
# 示例：
# 输入：
#      4
#    /   \
#   2     7
#  / \   / \
# 1   3 6   9
# 输出：
#      4
#    /   \
#   7     2
#  / \   / \
# 9   6 3   1
def invertTree(root):
    """
    :type root: TreeNode
    :rtype: TreeNode
    """
    queue = []
    queue.append(root)
    while queue:
        x = queue.pop(0)
        x.left,x.right = x.right,x.left
        if x.left:
            queue.append(x.left)
        if x.right:
            queue.append(x.right)
    return root

########################################################################################
# 给定一个二叉树，它的每个结点都存放着一个整数值。
# 找出路径和等于给定数值的路径总数。
# 路径不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。
# 二叉树不超过1000个节点，且节点数值范围是 [-1000000,1000000] 的整数。
# 示例：
# root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8
#
#       10
#      /  \
#     5   -3
#    / \    \
#   3   2   11
#  / \   \
# 3  -2   1
#
# 返回 3。和等于 8 的路径有:
#
# 1.  5 -> 3
# 2.  5 -> 2 -> 1
# 3.  -3 -> 11
def pathSum(root, sum):
    """
    :type root: TreeNode
    :type sum: int
    :rtype: int
    """
    if not root:
        return 0
    res = 0
    stack = []
    stack.append((root,[root.val]))
    while stack:
        x,sum_list = stack.pop(0)
        for i in sum_list:
            if i == sum:
                res += 1
        if x.left:
            stack.append((x.left,[x.left.val] + [x.left.val + s for s in sum_list]))
        if x.right:
            stack.append((x.right,[x.right.val] + [x.right.val + s for s in sum_list]))
    return res



# t = invertTree(tree)
# print(trav_level2(t))
# print(trav_in_i(t))

# print(trav_level2(tree))
# print(trav_level2(tree2))
#
# print(trav_level2(mergeTrees(tree,tree2)))


def find_path(x,layer):
    if x is None:
        return
    print(layer,x.val)
    layer += 1
    find_path(x.left,layer)
    find_path(x.right,layer)
    layer -= 1


def find_path2(x,path = []):
    if x is None:
        return
    path.append(x.val)
    print(x.val)
    print(path)
    find_path2(x.left,path)
    find_path2(x.right,path)
    path.pop()

def find_path3(x,path):
    if x is None:
        return
    path.append(x.val)
    print(path)
    print(x.val)
    find_path3(x.left, path)
    find_path3(x.right, path)

path=[]
visit = {}
def quanpailie(index):
    if index >= 3:
        print(path)
        return
    for i in range(3):
        if visit.get(i,0):
            continue
        path.append(i)
        visit[i] = True
        quanpailie(index+1)
        path.pop()
        visit[i] = False


###############################257. 二叉树的所有路径#######################
# 给定一个二叉树，返回所有从根节点到叶子节点的路径。
# 说明: 叶子节点是指没有子节点的节点。
# 示例:
# 输入:
#    1
#  /   \
# 2     3
#  \
#   5
#
# 输出: ["1->2->5", "1->3"]
# 解释: 所有根节点到叶子节点的路径为: 1->2->5, 1->3

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution257(object):
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        self.res = []
        self._dfs(root,[])
        return self.res
    def _dfs(self,root,path):
        if not root:
            return
        path.append(root.val)
        if not root.right and not root.left:
            self.res.append(path[:])
            return
        if root.left:
            self._dfs(root.left,path)
            path.pop()
        if root.right:
            self._dfs(root.right,path)
            path.pop()

######################################572. 另一个树的子树###########################################
# 给定两个非空二叉树 s 和 t，检验 s 中是否包含和 t 具有相同结构和节点值的子树。s 的一个子树包括 s 的一个节点和这个节点的
# 所有子孙。s 也可以看做它自身的一棵子树。
# 示例 1:
# 给定的树 s:
#      3
#     / \
#    4   5
#   / \
#  1   2
# 给定的树 t：
#    4
#   / \
#  1   2
# 返回 true，因为 t 与 s 的一个子树拥有相同的结构和节点值。
# 示例 2:
# 给定的树 s：
#      3
#     / \
#    4   5
#   / \
#  1   2
#     /
#    0
# 给定的树 t：
#    4
#   / \
#  1   2
# 返回 false。
class Solution572(object):
    def isSubtree(self, s, t):
        """
        :type s: TreeNode
        :type t: TreeNode
        :rtype: bool
        """
        if not t:
            return True
        if not s:
            return False
        return self.isSame(s,t) or self.isSubtree(s.left,t) or self.isSubtree(s.right,t)

    def isSame(self, p, q):
        if not p and not q:
            return True
        if not p or not q:
            return False
        return p.val == q.val and self.isSame(p.left, q.left) and self.isSame(p.right, q.right)


################################101. 对称二叉树##############################
# 给定一个二叉树，检查它是否是镜像对称的。
# 例如，二叉树 [1,2,2,3,4,4,3] 是对称的。
#     1
#    / \
#   2   2
#  / \ / \
# 3  4 4  3
# 但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:
#     1
#    / \
#   2   2
#    \   \
#    3    3
class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True
        return self._isMirror(root.left,root.right)
    def _isMirror(self,r1,r2):
        if not r1 and not r2:
            return True
        if not r1 or not r2:
            return False
        if r1.val != r2.val:
            return False
        return self._isMirror(r1.left,r2.right) and self._isMirror(r1.right,r2.left)

##############################112. 路径总和########################################
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
# 返回 true, 因为存在目标和为 22 的根节点到叶子节点的路径 5->4->11->2。
class Solution112(object):
    def hasPathSum(self, root, sum_):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        if not root:
            if sum_ == 0:
                return True
            else:
                return False
        sum_ = sum_ - root.val
        if not root.left and not root.right:
            if sum_ == 0:
                return True
            else:
                return False
        return self.hasPathSum(root.left,sum_) or self.hasPathSum(root.right,sum_)


    #     if not root:
    #         return False
    #     self.res = False
    #     self.res_path = []
    #     self._trackback(root,sum_,[root.val])
    #     print(self.res_path)
    #     return self.res
    #
    # def _trackback(self,x,sum_,path):
    #     # if not x:
    #     #     return
    #     if not x.left and not x.right:
    #         if sum(path) == sum_:
    #             self.res = True
    #             self.res_path.append(path[:])
    #     if x.left:
    #         self._trackback(x.left,sum_,path+[x.left.val])
    #     if x.right:
    #         self._trackback( x.right, sum_, path+[x.right.val])
#

# #########################543. 二叉树的直径####################################
# 给定一棵二叉树，你需要计算它的直径长度。一棵二叉树的直径长度是任意两个结点路径长度中的最大值。这条路径可能穿过根结点。
# 示例 :
# 给定二叉树
#           1
#          / \
#         2   3
#        / \
#       4   5
# 返回 3, 它的长度是路径 [4,2,1,3] 或者 [5,2,1,3]。
# 注意：两结点之间的路径长度是以它们之间边的数目表示。
class Solution543(object):
    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        self.max_d = 0
        def get_depth(root):
            if not root:
                return 0
            left = get_depth(root.left)
            right = get_depth(root.right)
            self.max_d = max(self.max_d,left+right+1)
            return max(left,right) + 1
        get_depth(root)
        return self.max_d - 1

###################################538. 把二叉搜索树转换为累加树#################################
# 给定一个二叉搜索树（Binary Search Tree），把它转换成为累加树（Greater Tree)，
# 使得每个节点的值是原来的节点值加上所有大于它的节点值之和。
# 例如：
# 输入: 二叉搜索树:
#               5
#             /   \
#            2     13
# 输出: 转换为累加树:
#              18
#             /   \
#           20     13


class Solution538(object):
    def convertBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return root
        self.pre = 0
        def helper(root):
            if not root:
                return
            helper(root.right)
            root.val = root.val + self.pre
            self.pre = root.val
            helper(root.left)
        helper(root)
        return root


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

################################236. 二叉树的最近公共祖先######################################
# 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
# 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，
# 满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”
# 例如，给定如下二叉树:  root = [3,5,1,6,2,0,8,null,null,7,4]
# 示例 1:
# 输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
# 输出: 3
# 解释: 节点 5 和节点 1 的最近公共祖先是节点 3。
# 示例 2:
# 输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
# 输出: 5
# 解释: 节点 5 和节点 4 的最近公共祖先是节点 5。因为根据定义最近公共祖先节点可以为节点本身。
class Solution236(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """

        # self.all_path = []
        # def dfs_tree(root,path):
        #     if not root.left and not root.right:
        #         self.all_path.append(path[:])
        #         return
        #     if root.left:
        #         path.append(root.left.val)
        #         dfs_tree(root.left,path)
        #         path.pop()
        #     if root.right:
        #         path.append(root.right.val)
        #         dfs_tree(root.right,path)
        #         path.pop()
        # dfs_tree(root,[root.val])
        # return self.all_path
        self.res = root
        def dfs_tree(root):
            if not root:
                return False
            left = dfs_tree(root.left)
            right = dfs_tree(root.right)
            mid = root == q or root == p
            if left + right + mid >= 2:
                self.res = root
            return left or right or mid
        dfs_tree(root)
        return self.res


# ##################################235. 二叉搜索树的最近公共祖先#####################################
# 给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。
# 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且
# x 的深度尽可能大（一个节点也可以是它自己的祖先）。”
# 例如，给定如下二叉搜索树:  root = [6,2,8,0,4,7,9,null,null,3,5]
# 示例 1:
# 输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
# 输出: 6
# 解释: 节点 2 和节点 8 的最近公共祖先是 6。
# 示例 2:
# 输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
# 输出: 2
# 解释: 节点 2 和节点 4 的最近公共祖先是 2, 因为根据定义最近公共祖先节点可以为节点本身。
# 说明:
# 所有节点的值都是唯一的。
# p、q 为不同节点且均存在于给定的二叉搜索树中。
class Solution235(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        root_val = root.val
        p_val = p.val
        q_val = q.val
        if p_val > root_val and q_val > root_val:
            return self.lowestCommonAncestor(root.right,p,q)
        if p_val < root_val and q_val < root_val:
            return self.lowestCommonAncestor(root.left,p,q)
        else:
            return root


####################### 124. 二叉树中的最大路径和########################################
# 给定一个非空二叉树，返回其最大路径和。
# 本题中，路径被定义为一条从树中任意节点出发，达到任意节点的序列。该路径至少包含一个节点，且不一定经过根节点。
# 示例 1:
# 输入: [1,2,3]
#        1
#       / \
#      2   3
# 输出: 6
# 示例 2:
# 输入: [-10,9,20,null,null,15,7]
#    -10
#    / \
#   9  20
#     /  \
#    15   7
# 输出: 42
class Solution124(object):
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        res = float('-inf')
        def max_gain(root):
            nonlocal res
            if not root:
                return 0
            left = max(max_gain(root.left),0)
            right = max(max_gain(root.right),0)
            max_ =  left + right + root.val
            res = max(res,max_)
            return root.val + max(left,right)
        max_gain(root)
        return res
    # def max_gain(self,root):
    #     if not root:
    #         return 0
    #     left = max(self.max_gain(root.left),0)
    #     right = max(self.max_gain(root.right),0)
    #     max_ =  left + right + root.val
    #     self.res = max(self.res,max_)
    #     return root.val + max(left,right)

s124 = Solution124()
print(s124.maxPathSum(tree))