class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


head = ListNode(-1000)
p = head
for i in range(4):
    p_new = ListNode(i)
    p.next = p_new
    p = p_new
head = head.next

def print_list(head):
    list_arr = []
    p = head
    while p is not None:
        list_arr.append(p.val)
        p = p.next
    print("-->".join([str(x) for x in list_arr]))


##########################83. 删除排序链表中的重复元素###############################
# 给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。
# 示例 1:
# 输入: 1->1->2
# 输出: 1->2
# 示例 2:
# 输入: 1->1->2->3->3
# 输出: 1->2->3
# ################82. 删除排序链表中的重复元素 II
# 给定一个排序链表，删除所有含有重复数字的节点，只保留原始链表中 没有重复出现 的数字。
# 示例 1:
# 输入: 1->2->3->3->4->4->5
# 输出: 1->2->5
# 示例 2:
# 输入: 1->1->1->2->3
# 输出: 2->3

class Solution83(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        p = head
        while p:
            if p.next and p.val == p.next.val:
                p.next=p.next.next
            p = p.next
        return head
    def deleteDuplicates82(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        pre,pre.next= ListNode(0),head
        pl,pr = pre,head
        while pr:
            if pr.next and pr.next.val == pr.val:
                tmp = pr.val
                while pr and pr.val == tmp:
                    pr = pr.next
            else:
                pl.next = pr
                pl = pl.next
                pr = pr.next
        pl.next = pr    #0-->0-->2-->2的情况下，没做
        return pre.next


#################141. 环形链表##########################
# 给定一个链表，判断链表中是否有环。
# 为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。
# 示例 1：
# 输入：head = [3,2,0,-4], pos = 1
# 输出：true
# 解释：链表中有一个环，其尾部连接到第二个节点。

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        visit = set()
        while head:
            if head not in visit:
                visit.add(head)
            else:
                return True
            head = head.next
        return False
    def hasCycle2(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        pslow = head
        pfast = head
        while pfast and pfast.next:
            pslow = pslow.next
            pfast = pfast.next.next
            if pslow == pfast:
                return True
        return False


print_list(head)
head.next.val = 0
head.next.next.next.val = 2
print_list(head)
s83 = Solution83()
print_list(s83.deleteDuplicates82(head))


