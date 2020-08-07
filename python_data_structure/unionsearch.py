##############################200. 岛屿数量#####################
# 给定一个由 '1'（陆地）和 '0'（水）组成的的二维网格，计算岛屿的数量。一个岛被水包围，并且它是通过水平方向或垂直方
# 向上相邻的陆地连接而成的。你可以假设网格的四个边均被水包围。
# 示例 1:
# 输入:
# 11110
# 11010
# 11000
# 00000
# 输出: 1
# 示例 2:
# 输入:
# 11000
# 11000
# 00100
# 00011
# 输出: 3
from collections import deque
class Solution200(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """

        count = 0
        n,m = len(grid),len(grid[0])
        visit = set()
        def add_visit(i,j):
            visit.add((i,j))
            for off_x,off_y in [(-1,0),(0,-1),(1,0),(0,1)]:
                if (off_x+i,off_y+j) not in visit and 0 <= off_x+i < n and 0 <= off_y+j < m and grid[i][j] == 1:
                    add_visit(off_x+i,off_y+j)

        def add_visit2(i,j):
            queue = deque()
            queue.append((i,j))
            while queue:
                x, y = queue.popleft()
                visit.add((x, y))
                for off_x, off_y in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                    if (off_x + x, off_y + y) not in visit and 0 <= off_x + x < n and 0 <= off_y + y < m and grid[off_x + x][off_y + y] == 1:
                        queue.append((off_x + x, off_y + y))

        for i in range(n):
            for j in range(m):
                if (i,j) not in visit:
                    if grid[i][j] == 1:
                        count += 1
                        add_visit2(i,j)
        return count


# grid = [[1,1,0,0,0],
#         [1,1,0,0,0],
#         [0,0,1,0,0],
#         [0,0,0,1,1]]
# s200 = Solution200()
# print(s200.numIslands(grid))


#def recursive_lcs(str_a,path):



#print(recursive_lcs('1354', '31564',''))
def backtrack(s,layer,path):
    if layer >= len(s):
        print(path)
        return
    backtrack(s,layer+1,path+s[layer])
    backtrack(s,layer+1,path)
