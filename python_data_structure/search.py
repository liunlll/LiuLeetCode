def binarySearch(sortedList,target,lo,hi):
    '''
    :param sortedList:
    :param target:
    :param lo: 第一个元素
    :param hi: 最后一个元素的位置+1
    :return: 成功返回index,失败-1
    '''
    while(lo < hi):
        mi = (lo + hi) // 2
        if target < sortedList[mi]:
            hi = mi
        elif target > sortedList[mi]:
            lo = mi + 1
        else:
            return mi
    return -1

#########################剑指offer11###################################################################
# 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。 输入一个非减排序的数组的一个旋转，输出旋转数组的最小
# 元素。 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。 NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
# -*- coding:utf-8 -*-
class Solutionj18(object):
    def minNumberInRotateArray(self, rotateArray):
        if not rotateArray:
            return
        lo = 0
        hi = len(rotateArray)-1
        if rotateArray[lo] < rotateArray[hi]:
            return rotateArray[lo]
        elif rotateArray[lo] > rotateArray[hi]:
            while hi-lo > 1:
                mi = (hi + lo) // 2
                if rotateArray[lo] <= rotateArray[mi]:
                    lo = mi
                else:
                    hi = mi
            return rotateArray[hi]
        else:
            return min(rotateArray)
sj18 = Solutionj18()
print(sj18.minNumberInRotateArray([3]))

