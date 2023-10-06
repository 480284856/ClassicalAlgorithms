void addPath(int *vec, int *vecSize, struct TreeNode *node) {
    int count = 0;
    while (node != NULL) {
        ++count;
        vec[(*vecSize)++] = node->val;
        node = node->right;
    }
    for (int i = (*vecSize) - count, j = (*vecSize) - 1; i < j; ++i, --j) {
        int t = vec[i];
        vec[i] = vec[j];
        vec[j] = t;
    }
}

int *postorderTraversal(struct TreeNode *root, int *returnSize) {
    int *res = malloc(sizeof(int) * 2001);
    *returnSize = 0;
    if (root == NULL) {
        return res;
    }

    struct TreeNode *p1 = root, *p2 = NULL;

    while (p1 != NULL) {
        p2 = p1->left;
        if (p2 != NULL) {
            while (p2->right != NULL && p2->right != p1) {
                p2 = p2->right;
            }
            if (p2->right == NULL) {
                p2->right = p1;
                p1 = p1->left;
                continue;
            } else {//这里是第二次循环，从父节点下来的第二次循环，后续遍历就在这里搜集元素。
                p2->right = NULL;
                addPath(res, returnSize, p1->left);
            }
        }
        p1 = p1->right;
    }
    addPath(res, returnSize, root);
    return res;
}

//作者：力扣官方题解
//链接：https://leetcode.cn/problems/binary-tree-postorder-traversal/
//来源：力扣（LeetCode）
//著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
