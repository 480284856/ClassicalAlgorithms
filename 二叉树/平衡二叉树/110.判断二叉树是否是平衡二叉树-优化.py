class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        return self.getHeight(root)>=0
        
    def getHeight(self, root: Optional[TreeNode]) -> int:
        if(root!=None):
            h_left = self.getHeight(root.left)
            if(h_left==-1):
                return -1
            h_right = self.getHeight(root.right)
            if(h_right==-1):
                return -1
            if( abs(h_left-h_right)>1):
                return -1
            else:
                return max(h_left,h_right)+1
        else:
            return 0
