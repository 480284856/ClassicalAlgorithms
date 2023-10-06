void postOrder(struct TreeNode *root, int *result, int *returnSize);

int* postorderTraversal(struct TreeNode* root, int* returnSize){
    int *result = (int *)malloc(sizeof(int)*100);
    *returnSize = 0;
    postOrder(root, result, returnSize);
    return result;
}


void postOrder(struct TreeNode *root, int *result, int *returnSize){
    if(!root){
        return;
    }else{
        postOrder(root->left, result, returnSize);
        postOrder(root->right, result, returnSize);
        result[*returnSize] = root->val;
        (*returnSize) ++;
    }
}
