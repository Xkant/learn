import numpy as np



class Node():
    def __init__(self):
        self.pre = None
        self.left = None
        self.right = None
        self.split_feature = None
        self.split_point = None
        self.data = None


class Tree():
    def __init__(self, task="classifier"):
        self.task = task
        self.tree = Node()
        self.nodes = 0
        self.leafs = []
        self.leafs_count = 0
        self.deep = 0
        self.task_map = {"classifier": [self._gini, self._gain_gini],
                    "regresion": [self._ls, self._gain_ls]}

    def _gini(self, y):
        """
        计算基尼数
        :param y:
        :return:
        """
        unique_y = set(y)
        p_2 = 0.0
        for i in unique_y:
            mask = (y == i)
            p_2 += ((sum(mask))/y.size)**2
        return 1-p_2


    def _gain_gini(self, y1, y2):
        """
        计算基尼增益
        :param y1:
        :param y2:
        :return:
        """
        n1 = y1.size
        n2 = y2.size
        N = n1 + n2
        gini1 = self._gini(y1)
        gini2 = self._gini(y2)
        return (n1/N)*gini1 + (n2/N)*gini2


    def _ls(self, y):
        """
        计算方差
        :param y:
        :return:
        """
        return np.var(y)


    def _gain_ls(self, y1, y2):
        """
        计算方差增益
        :param y1:
        :param y2:
        :return:
        """
        return self._ls(y1) + Tree._ls(y2)


    def _best_split(self, X, y, f):
        """
        搜索最佳切分属性和最佳切分点
        :param X:
        :param y:
        :param f:
        :return:
        """
        base = None
        best_feature = f[0]
        best_point = 0.0
        for i in range(X.shape[1]):
            unique_x = set(X[i])
            for j in unique_x:
                mask1 = X[:,i] >= j
                mask2 = X[:,i] < j
                cur  = self.task_map[self.task][1](y[mask1], y[mask2])
                if base is None or cur < base:
                    base = cur
                    best_point = j
                    best_feature = i
                    best_feature_mask = f[i]
        return (best_feature, best_point, best_feature_mask)


    def _split_data(self, X, y, best_feature, best_point, f):
        """
        切分数据集
        :param X:
        :param y:
        :param best_feature:
        :param best_point:
        :return:
        """
        mask_col = (np.array([i for i in range(X.shape[1])]) != best_feature)
        mask1 = X[:,best_feature] < best_point
        mask2 = X[:,best_feature] >= best_point
        X1 = X[mask1][:, mask_col]
        y1 = y[mask1]
        f = f[mask_col]
        X2 = X[mask2][:, mask_col]
        y2 = y[mask2]
        return (X1, y1, X2, y2, f)


    def _get_leafs_(self, node):
        """
        获取树的叶子节点
        :param node:
        :return:
        """
        if node is None:
            return self.leafs
        if node.left is None and node.right is None:
            self.leafs.append(node)
            return self.leafs
        self._get_leafs_(node.left)
        self._get_leafs_(node.right)
        return self.leafs


    def _get_leafs(self):
        self.leafs = []
        self.leafs = self._get_leafs_(self.tree)
        return self.leafs



    def _get_leafs_count(self):
        """
        获取叶子节点数
        :return:
        """
        self._get_leafs()
        return len(self.leafs)


    def _get_deep(self):
        """
        获取数的深度
        :return:
        """
        self._get_leafs()
        deep = []
        for node in self.leafs:
            cur = node
            count = 0
            while cur.pre is not None:
                cur = cur.pre
                count += 1
            deep.append(count)
        return max(deep)


    def _stop(self, X, y):
        """
        if (y.size <= self.rows_threshold
                or X.size <= self.cols_threshold
                or self._get_leafs_count() >= self.max_leaf
                or self._get_deep() >= self.max_deep):
            return True
        """
        if X.shape[1] <= 3 or X.shape[0] <= 1:
            return True
        else:
            return False


    def _create_tree(self, X, y, f):
        """
        构建决策树
        :param X:
        :param y:
        :return:
        """
        node = Node()
        if self.tree.pre is None:
            self.tree = node
        if self.task == "classifier":
            node.data = (y == 1).sum()/y.size
        elif self.task == "regression":
            node.data = np.average(y)
        if self._stop(X,y):
            return node
        best_feature, best_point , best_feature_mask= self._best_split(X, y, f)
        X1, y1, X2, y2 ,f = self._split_data(X, y, best_feature, best_point, f)
        left = self._create_tree(X1, y1, f)
        right = self._create_tree(X2, y2, f)
        node.left = left
        node.right = right
        node.split_feature = best_feature_mask
        node.split_point = best_point
        left.pre = node
        right.pre = node
        return node


    def _get_score(self, x, node):
        """
        获取样本落在叶子节点的值
        :param x:
        :param node:
        :return:
        """
        if node.left is None and node.right is None:
            return node.data
        split_feature = node.split_feature
        split_point = node.split_point
        if x[split_feature] <= split_point:
            return self._get_score(x, node.left)
        elif x[split_feature]> split_point:
            return self._get_score(x, node.right)


    def fit(self,X, y, max_leaf=None, max_deep=None, cols_threshold=0, rows_threshold=0, gini_threshold=0):
        self.max_leaf = max_leaf
        self.max_deep = max_deep
        self.cols_threshold = cols_threshold
        self.rows_threshold = rows_threshold
        self.gini_threshold = gini_threshold
        self.cols = X.shape[1]
        self.rows = X.shape[0]
        f = np.array([i for i in range(X.shape[1])])
        self.tree = self._create_tree(X, y, f)
        return self


    def _predict(self, X):
        p = np.zeros_like(X[:, 0])
        for i in range(X.shape[0]):
            p[i] = self._get_score(X[i,:], self.tree)
        return p


    def predict(self, X):
        p = self._predict(X)
        #print(p)
        if self.task == "classifier":
            result = np.zeros_like(p)
            mask = p > 0.5
            result[mask] = 1
            return result
        elif self.task == "regression":
            return p

    def predict_proba(self, X):
        return self._predict(X)



def test1():
    node1 = Node()
    node2= Node()
    node3 = Node()
    node4 = Node()
    node5 = Node()
    node6 = Node()
    node7 = Node()
    node8 = Node()
    node9 = Node()
    node1.left = node2
    node1.right = node3
    node2.pre = node1
    node3.pre = node1
    node2.left = node4
    node2.right = node5
    node4.pre = node2
    node5.pre = node2
    node3.left = node6
    node3.right = node7
    node6.pre = node3
    node7.pre = node3
    node6.left = node8
    node6.right = node9
    node8.pre = node6
    node9.pre = node6
    tree = Tree()
    tree.tree = node1
    print(tree._get_leafs_count())

def test2():
    from sklearn.datasets import load_iris
    data = load_iris()
    mask = data.target < 2
    X = data.data[mask]
    y = data.target[mask]
    tree = Tree()
    tree.fit(X,y)
    print(tree._get_deep())
    print(tree._get_leafs())
    print(tree._get_leafs_count())
    print(y)
    print(tree.predict(X))


if __name__ == "__main__":
    test2()
    
