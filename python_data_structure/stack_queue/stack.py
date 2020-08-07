def isValid(s):
    stack = []
    paren_map = {'}':'{',']':'[',')':'('}
    for i in s:
        if i not in paren_map.keys():
            stack.append(i)
        else:
            if not stack or paren_map[i] != stack.pop():
                return False
    return not stack

################################剑指offer9#############################################
# 用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。
class Solutionj9(object):
    def __init__(self):
        self.stack1 = []
        self.stack2 = []
    def push(self, node):
        # write code here
        self.stack1.append(node)
    def pop(self):
        if not self.stack2 and not self.stack1:
            return False
        if not self.stack2:
            for i in range(len(self.stack1)):
                self.stack2.append(self.stack1.pop())
        return self.stack2.pop()




# ######################################946. 验证栈序列###################################
# 给定 pushed 和 popped 两个序列，只有当它们可能是在最初空栈上进行的推入 push 和弹出 pop 操作序列的结果时，返回 true；否则，返回 false 。
# 示例 1：
# 输入：pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
# 输出：true
# 解释：我们可以按以下顺序执行：
# push(1), push(2), push(3), push(4), pop() -> 4,
# push(5), pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1
# 示例 2：
# 输入：pushed = [1,2,3,4,5], popped = [4,3,5,1,2]
# 输出：false
# 解释：1 不能在 2 之前弹出。
# 提示：
# 0 <= pushed.length == popped.length <= 1000
# 0 <= pushed[i], popped[i] < 1000
# pushed 是 popped 的排列。
class Solution946(object):
    def validateStackSequences(self, pushed, popped):
        """
        :type pushed: List[int]
        :type popped: List[int]
        :rtype: bool
        """
        queue = []
        i,j = 0,0
        while i < len(pushed) and j < len(popped):
            if pushed[i] == popped[j]:
                i,j = i+1,j+1
            elif queue and queue[-1] == popped[j]:
                queue.pop()
                j = j+1
            else:
                queue.append(pushed[i])
                i = i+1
        return queue[::-1] == popped[j:]


