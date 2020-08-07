class ListNode(object):
     def __init__(self,x):
         self.val = x
         self.next = None

head = ListNode(-1000)
p = head
for i in range(5):
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


def revers_list(head):
    p = head
    new_h = None
    while p is not None:
        tmp = p.next
        p.next = new_h
        new_h = p
        p = tmp
    return new_h
def revers_list2(head):
    cur,prev = head,None
    while cur:
        #prev,cur,cur.next = cur,cur.next,prev
        cur.next,prev,cur= prev,cur,cur.next
    return prev

def swapPairs(head):
    pre,pre.next = ListNode(-1000),head
    dummy = pre
    while pre.next and pre.next.next:
        a = pre.next
        b = a.next
        pre.next = b
        a.next = b.next
        b.next = a
        pre = a
    return dummy.next

# def swap_two(head,n,m):
#     dummy = ListNode(-1000)
#     dummy.next = head
#     p = dummy
#     while p and n > 0:
#         p = p.next

#
# print_list(head)
# print_list(swapPairs(head))


print_list(head)


