# https://leetcode.cn/problems/balanced-binary-tree/
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        if(root==None):
            return True
        else:
            if( abs(self.getHeight(root.left)-self.getHeight(root.right))>1 ):
                return False
            else:
                return self.isBalanced(root.left) and self.isBalanced(root.right)
    
    def getHeight(self, root: Optional[TreeNode]) -> int:
        if(root==None):
            return 0
        else:
            return max(self.getHeight(root.left), self.getHeight(root.right))+1

