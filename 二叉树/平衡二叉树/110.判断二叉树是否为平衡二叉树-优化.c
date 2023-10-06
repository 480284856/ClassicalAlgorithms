int getHeight(struct TreeNode *root);

bool isBalanced(struct TreeNode* root){
    return getHeight(root)>=0;
}


int getHeight(struct TreeNode *root){
    if(root==NULL){
        return 0;
    }else{
        int h_left,h_right;
        h_left = getHeight(root->left);
        if(h_left==-1){
            return -1;
        }
        h_right = getHeight(root->right);
        if(h_right==-1){
            return -1;
        }
        if(abs(h_left-h_right)>1){
            return -1;
        }
        return (h_left>h_right)? h_left+1 : h_right+1;
    }
}
