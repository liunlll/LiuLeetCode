#面试题53题目一：统计一个数字在排序数组中出现的次数
#[1,2,3,3,3,3,4,5],输出3
# -*- coding:utf-8 -*-
class Solution53_1(object):
    def GetNumberOfK(self, data, k):
        if not data:
            return 0
        return self.GetNumberLastOfK(data,k,0,len(data)) - self.GetNumberFirstOfK(data,k,0,len(data)) + 1

    def GetNumberFirstOfK(self,data,k,lo,hi):
        while hi - lo >= 0:
            mi = (lo + hi) // 2
            if k < data[mi]:
                hi = mi
            elif data[mi] < k:
                lo = mi
            else:
                if mi == 0 or data[mi-1] != k:
                    return mi
                else:
                    hi = mi
    def GetNumberLastOfK(self,data,k,lo,hi):
        while hi - lo > 0:
            mi = (lo + hi) // 2
            if k < data[mi]:
                hi = mi
            elif data[mi] < k:
                lo = mi
            else:
                if mi == len(data)-1 or data[mi+1] != k:
                    return mi
                else:
                    lo = mi
s53_1 = Solution53_1()
nums = [3,3,3,3,3,3,3,4]
# print(s53_1.GetNumberFirstOfK(nums,3,0,len(nums)))
# print(s53_1.GetNumberLastOfK(nums,3,0,len(nums)))
print(s53_1.GetNumberOfK([],3))