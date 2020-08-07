# -*- coding:utf-8 -*-
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
#########输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
# 例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
class Solution1:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        if not pre or not tin:
            return None         #列表为空就是假
        return self.ConstructCore(pre,tin)
    def ConstructCore(self,pre,tin):
        root = TreeNode(pre[0])     #构造根节点

        root_index_in_tin = tin.index(root.val) #在中序遍历中找到根节点所在的位置
        if root_index_in_tin > 0:   #存在左子树
            root.left = self.ConstructCore(pre[1:root_index_in_tin + 1],tin[0:root_index_in_tin])
        if len(tin) - 1 - root_index_in_tin > 0: #存在右子树
            root.right = self.ConstructCore(pre[root_index_in_tin + 1:],tin[root_index_in_tin + 1:])
        return root

pre = [1,2,4,7,3,5,6,8]
tin = [4,7,2,1,5,3,8,6]
s1 = Solution1()
root = s1.reConstructBinaryTree(pre,tin)

def trav_level2(x,res = []):
    if not x:
        return
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


# ############################33.二叉搜索树的后序遍历序列##########################
# 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。
class Solution33(object):
    def VerifySquenceOfBST(self, sequence):
        if not sequence:
            return False
        return self.VerifySquenceOfBST_helper(sequence)
    def VerifySquenceOfBST_helper(self, sequence):
        if len(sequence) <=  3:
            return True
        i = 0
        while i < len(sequence) - 1:
            if sequence[-1] < sequence[i]:
                break
            i += 1
        if sequence[i:-1] and (sequence[-1] > min(sequence[i:-1])):
            return False
        return self.VerifySquenceOfBST_helper(sequence[:i]) and self.VerifySquenceOfBST_helper(sequence[i:-1])


#55#############################
# 给定一个二叉树，判断它是否是高度平衡的二叉树。
# 本题中，一棵高度平衡二叉树定义为：
# 一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过1。
# 示例 1:
# 给定二叉树 [3,9,20,null,null,15,7]
#     3
#    / \
#   9  20
#     /  \
#    15   7
# 返回 true 。
# 示例 2:
# 给定二叉树 [1,2,2,3,3,null,null,4,4]
#        1
#       / \
#      2   2
#     / \
#    3   3
#   / \
#  4   4
# 返回 false 。

class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True
        self.res = True
        def dfs_tree(root):
            if not root:
                return 0
            left = dfs_tree(root.left) + 1
            right = dfs_tree(root.right) + 1
            if abs(left - right) > 1:
                self.res = False
            return max(left,right)
        dfs_tree(root)
        return self.res

