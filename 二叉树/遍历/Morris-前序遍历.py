class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        result = []

        while(root):
            node = root
            if(node.left):
                node = node.left
                while(node.right and (node.right != root)):
                    node = node.right
                if( node.right == root ):
                    node.right = None
                    root = root.right
                else:
                    result.append(root.val)##
                    node.right = root
                    root = root.left
            else:
                result.append(root.val)##
                root = root.right

        return result
