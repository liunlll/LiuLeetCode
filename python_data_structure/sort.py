import random
def bubbleSort(unSortlist,lo,hi):
    '''
    :param unSortlist: 未排序的向量
    :param lo: 排序的起点的位置,注意python是从0开始的
    :param hi: 要排序的终点的位置+1，注意python是从0开始的
    :return: 排序好的向量
    '''
    for i in range(lo,hi-1):
        for j in range(lo,hi-1-i):
            if unSortlist[j] > unSortlist[j+1]:
                unSortlist[j], unSortlist[j + 1] = unSortlist[j+1], unSortlist[j]
    return unSortlist

def merge(unSortlist,lo,mi,hi):
    print(unSortlist)
    print("asdfasdf ",lo,mi,hi)

    buffer = unSortlist[lo:mi]
    i = 0
    j = 0
    k = 0
    while(j < (mi - lo) or k < (hi - mi)):
        if j < (mi - lo):
            if k >= (hi - mi) or unSortlist[j + mi] <= buffer[k]:
                unSortlist[i + lo] = unSortlist[j + mi]
                i += 1
                j += 1
        if k < (hi - mi):
            if j >= (mi - lo) or buffer[k] < unSortlist[j + mi]:
                unSortlist[i + lo] = buffer[k]
                i += 1
                k += 1
    print(unSortlist)
    return unSortlist



def mergeSort(unSortlist,lo,hi):
    print(lo,hi)
    if (hi - lo) < 2:
        return
    mi = (lo + hi) // 2
    mergeSort(unSortlist,lo,mi)
    mergeSort(unSortlist,mi,hi)
    merge(unSortlist,lo,mi,hi)


def partition(unSortlist,lo,hi):
    r = (lo + hi) // 2
    unSortlist[lo],unSortlist[r] = unSortlist[r],unSortlist[lo]
    pivot = unSortlist[lo]
    while(lo < hi):
        while(lo < hi and pivot <= unSortlist[hi]):
            hi -= 1
        unSortlist[lo] = unSortlist[hi]
        while(lo < hi and unSortlist[lo] <= pivot):
            lo += 1
        unSortlist[hi] = unSortlist[lo]
    unSortlist[lo] = pivot
    return lo

def quickSort(unSortlist,lo=0, hi = None):
    if hi is None:
        hi = len(unSortlist)
    if hi - lo < 2:
        return
    mi = partition(unSortlist,lo,hi-1)
    quickSort(unSortlist,lo,mi)
    quickSort(unSortlist,mi + 1,hi)

def maopao(input_list):
    flag = True
    for j in range(len(input_list) - 1):
        for i in range(0,len(input_list)-j-1):
            if input_list[i] > input_list[i+1]:
                input_list[i],input_list[i+1] = input_list[i+1],input_list[i]
                flag = False
        if flag:
            return input_list
    return input_list


if __name__ == '__main__':
    input_list = [22,4,23,21,34,14,4,14,21,21,12]
    # print(merge([2, 6, 1, 4],0,2,4))
    # mergeSort(input_list,0,len(input_list))
    quickSort(input_list)
    print(input_list)
# import random
# print(random.randint(1,2))