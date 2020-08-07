##################208. 实现 Trie (前缀树)########################################
# 实现一个 Trie (前缀树)，包含 insert, search, 和 startsWith 这三个操作。
# 示例:
# Trie trie = new Trie();
# trie.insert("apple");
# trie.search("apple");   // 返回 true
# trie.search("app");     // 返回 false
# trie.startsWith("app"); // 返回 true
# trie.insert("app");
# trie.search("app");     // 返回 true
# 说明:
# 你可以假设所有的输入都是由小写字母 a-z 构成的。
# 保证所有输入均为非空字符串。
class Trie(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root ={}
        self.end_of_word = "#"

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: None
        """
        cur_node = self.root
        for c in word:
            if c not in cur_node:
                cur_node[c] = {}
            cur_node = cur_node[c]
        cur_node[self.end_of_word] = self.end_of_word

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        cur_node = self.root
        for c in word:
            if c not in cur_node:
                return False
            cur_node = cur_node[c]
        if self.end_of_word not in cur_node:
            return False
        return True

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        cur_node = self.root
        for c in prefix:
            if c not in cur_node:
                return False
            cur_node = cur_node[c]
        return True

t = Trie()
t.insert("abcd")
t.insert("adcd")
print(t.search("abc"))
print(t.startsWith("abc"))
print(t.root)
