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
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
};

struct TreeNode *CreateTreeNode(int val);
void PreOder(struct TreeNode *root, int *result, int *returnSize);
int* preorderTraversal(struct TreeNode* root, int* returnSize);

int main() {
    struct TreeNode *root = CreateTreeNode(0);
    root->left = CreateTreeNode(1);
    root->right = CreateTreeNode(2);
    root->left->left = CreateTreeNode(3);

    int *result;
    result = NULL;
    int returnSize;

    result = preorderTraversal(root, &returnSize);
    for(int i=0; i<returnSize; i++){
        printf("%d\t", result[i]);
    }
    return 0;
}

int* preorderTraversal(struct TreeNode* root, int* returnSize){
    int * result = (int *)malloc(sizeof(int)*100);
    *returnSize = 0;
    PreOder(root, result, returnSize);
    return result;
}

void PreOder(struct TreeNode *root, int *result, int *returnSize){
    if(!root){
        return;
    }

    result[(*returnSize)++] = root->val;
    PreOder(root->left, result, returnSize);
    PreOder(root->right, result, returnSize);
}


struct TreeNode *CreateTreeNode(int val){
    struct TreeNode *root = (struct  TreeNode *)malloc(sizeof(struct TreeNode));
    if(root == NULL){
        printf("Out of Memory");
        exit(1);
    }
    root->val = val;
    root->left = NULL;
    root->right = NULL;
    return root;
}
