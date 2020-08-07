#########贪心算法###################################
# 假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。对每个孩子 i ，都有一个胃口值 gi ，
# 这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 j ，都有一个尺寸 sj 。如果 sj >= gi ，我们可以将这个饼干 j 分配给孩子 i ，
# 这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。
# 注意：
# 你可以假设胃口值为正。
# 一个小朋友最多只能拥有一块饼干。
# 示例 1:
# 输入: [1,2,3], [1,1]
# 输出: 1
# 解释:
# 你有三个孩子和两块小饼干，3个孩子的胃口值分别是：1,2,3。
# 虽然你有两块小饼干，由于他们的尺寸都是1，你只能让胃口值是1的孩子满足。
# 所以你应该输出1。
# 示例 2:
# 输入: [1,2], [1,2,3]
# 输出: 2
# 解释:
# 你有两个孩子和三块小饼干，2个孩子的胃口值分别是1,2。
# 你拥有的饼干数量和尺寸都足以让所有孩子满足。
# 所以你应该输出2.
def findContentChildren(g, s):
    s.sort()
    g.sort()
    len_s = len(s)
    len_g = len(g)
    res = 0
    while len_s > 0:
        si = s[len_s - 1]
        while len_g > 0:
            gj = g[len_g - 1]
            len_g -= 1
            if si >= gj:
                res += 1
                break
        len_s -= 1
    return res

##################################################################################################
# 给定一个非负整数数组，你最初位于数组的第一个位置。
# 数组中的每个元素代表你在该位置可以跳跃的最大长度。
# 判断你是否能够到达最后一个位置。
# 示例 1:
# 输入: [2,3,1,1,4]
# 输出: true
# 解释: 从位置 0 到 1 跳 1 步, 然后跳 3 步到达最后一个位置。
# 示例 2:
# 输入: [3,2,1,0,4]
# 输出: false
# 解释: 无论怎样，你总会到达索引为 3 的位置。但该位置的最大跳跃长度是 0 ， 所以你永远不可能到达最后一个位置。
def canJump(nums):
    """
    :type nums: List[int]
    :rtype: bool
    """
    max_Jump = 0
    for i in range(len(nums)-1):
        if i > max_Jump:
            return False
        max_Jump = max((i + nums[i]),max_Jump)
    if max_Jump >= len(nums) - 1:
        return True
# ###############################################################################################################
# 给定一个用字符数组表示的 CPU 需要执行的任务列表。其中包含使用大写的 A - Z 字母表示的26 种不同种类的任务。任务可以以任意顺序执行，
# 并且每个任务都可以在 1 个单位时间内执行完。CPU 在任何一个单位时间内都可以执行一个任务，或者在待命状态。
# 然而，两个相同种类的任务之间必须有长度为 n 的冷却时间，因此至少有连续 n 个单位时间内 CPU 在执行不同的任务，或者在待命状态。
# 你需要计算完成所有任务所需要的最短时间。
# 示例 1：
# 输入: tasks = ["A","A","A","B","B","B"], n = 2
# 输出: 8
# 执行顺序: A -> B -> (待命) -> A -> B -> (待命) -> A -> B.
# 注：
# 任务的总个数为 [1, 10000]。
# n 的取值范围为 [0, 100]。
#贪心算法，贪的是目前的使用的最少时间
def _max_de(hash_map):
    max_ = max(hash_map,key = hash_map.get)
    hash_map[max_] -= 1
    return max_
task = ["A","A","A","A","B","B","C"]
order_task = []
hash_map = {}
for i in task:
    hash_map[i] = hash_map.get(i,0) + 1
prepre,pre = None,_max_de(hash_map)
cur_cost = 0
for i in range(1,len(task)):
    for t in hash_map:
        cur = t
        if cur == pre:
            cur_cost += 3
        elif cur == prepre:
            cur_cost += 2
        else:cur_cost += 1







# print(prepre)
# print(hash_map)
#
# max(hash_map,key = hash_map.get)
#
# nums = [3,2,1,0,4]
# print(canJump(nums))