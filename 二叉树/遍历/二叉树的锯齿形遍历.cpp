// https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/description/
class Solution {
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        vector<vector<int>> result;

        if(root==nullptr)   return result;
        
        queue<TreeNode *> Q;
        bool isl2r;
        int size;

        isl2r=true;
        Q.push(root);
        while(Q.empty()==false){
            size=Q.size();
            deque<int> oneLayer;
            while(size--){
                root=Q.front();Q.pop();
                if(root->left)  Q.push(root->left);
                if(root->right) Q.push(root->right);
                if(isl2r)   oneLayer.push_back(root->val);
                else    oneLayer.push_front(root->val);
            }
            isl2r = !isl2r;
            if(oneLayer.empty()==false){
                result.emplace_back(vector<int>{oneLayer.begin(), oneLayer.end()});
            }
        }
        return result;
    }
};
