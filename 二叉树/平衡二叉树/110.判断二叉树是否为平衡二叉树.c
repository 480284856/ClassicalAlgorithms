#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
};

struct TreeNode *CreateTreeNode(int val);
int getHeight(struct TreeNode *root);
bool isBalanced(struct TreeNode* root);


int main() {
    int result;

    struct TreeNode *root = CreateTreeNode(1);
    struct TreeNode *p;
    p = root;

    for(int i=2; i<5; i++){
        p->left = CreateTreeNode(i);
        p = p->left;
    }
    p = root;

    for(int i=2; i<5; i++){
        p->right = CreateTreeNode(i);
        p = p->right;
    }

    result = isBalanced(root);
    printf("%d\n", result);
    return 0;
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


bool isBalanced(struct TreeNode* root){
    if(!root){
        return true;
    }else{
        if( abs( getHeight(root->left)-getHeight(root->right) )>1 ){
            return false;
        }else{
            if( isBalanced(root->left) ){
                if( isBalanced(root->right) ){
                    return true;
                }else{
                    return false;
                }
            }else{
                return false;
            }

        }
    }
}


int getHeight(struct TreeNode *root){
    if(root==NULL){
        return 0;
    }else{
        int h_left,h_right;
        h_left = getHeight(root->left);
        h_right = getHeight(root->right);
        return (h_left>h_right)? h_left+1 : h_right+1;
    }
}
