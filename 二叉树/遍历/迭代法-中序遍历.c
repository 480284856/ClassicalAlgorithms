/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     struct TreeNode *left;
 *     struct TreeNode *right;
 * };
 */


/**
 * Note: The returned array must be malloced, assume caller calls free().
 */
int* inorderTraversal(struct TreeNode* root, int* returnSize){
    *returnSize = 0;
    struct TreeNode **stack = (struct TreeNode **)malloc(sizeof(struct TreeNode *)*100);
    int *result = (int *)malloc(sizeof(int)*100);
    int top=0;
    struct TreeNode *node = root;
    while(node || top != 0){
        while( node ){
            stack[top] = node;
            top ++;
            node = node->left;
        }
        node = stack[top-1];
        top --;
        result[(*returnSize)++] = node->val;
        node = node->right;
    }

    return result;
}
