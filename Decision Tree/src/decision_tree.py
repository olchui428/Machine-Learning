import numpy as np
import random

class Node():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        """
        This class implements a tree structure with multiple branches at each node.
        If self.branches is an empty list, this is a leaf node and what is contained in
        self.value is the predicted class.

        The defaults for this are for a root node in the tree.

        Arguments:
            branches (list): List of Node classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for left and
                right branches) or 0 (at a leaf node).
            attribute_name (str): Contains name of attribute that the tree splits the data
                on. Used for visualization (see `DecisionTree.visualize`).
            attribute_index (float): Contains the  index of the feature vector for the
                given attribute. Should match with self.attribute_name.
            value (number): Contains the value that data should be compared to along the
                given attribute.
        """
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value

class DecisionTree():
    def __init__(self, attribute_names):
        """
        TODO: Implement this class.

        This class implements a binary decision tree learner for examples with
        categorical attributes. Use the ID3 algorithm for implementing the Decision
        Tree: https://en.wikipedia.org/wiki/ID3_algorithm

        A decision tree is a machine learning model that fits data with a tree
        structure. Each branching point along the tree marks a decision (e.g.
        today is sunny or today is not sunny). Data is filtered by the value of
        each attribute to the next level of the tree. At the next level, the process
        starts again with the remaining attributes, recursing on the filtered data.

        Which attributes to split on at each point in the tree are decided by the
        information gain of a specific attribute.

        Here, you will implement a binary decision tree that uses the ID3 algorithm.
        Your decision tree will be contained in `self.tree`, which consists of
        nested Node classes (see above).

        Args:
            attribute_names (list): list of strings containing the attribute names for
                each feature (e.g. chocolatey, good_grades, etc.)
        
        """
        self.attribute_names = attribute_names
        self.tree = None

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    def fit(self, features, targets):
        """
        Takes in the features as a numpy array and fits a decision tree to the targets.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        Output:
            VOID: It should update self.tree with a built decision tree.
        """
        self._check_input(features)

        # # raise NotImplementedError()

        # Create a root node for the tree
        self.tree = Node()
        all_attributes = np.array(range(len(self.attribute_names)))
        self.add_node(features, targets, all_attributes, self.tree)

    def split(self, features, attribute_index, targets):
        feature = np.array(features[:,attribute_index]).flatten()
        fv0 = features[feature==0]
        fv1 = features[feature==1]
        target_fv0 = targets[feature==0]
        target_fv1 = targets[feature==1]

        return fv0, fv1, target_fv0, target_fv1

    def add_node(self, features, targets, unused_f, current_node):
        num_unused_f = len(unused_f)

        # If number of predicting attributes is empty, then Return the single node tree Root
        if num_unused_f == 0:
            count = [0, 0]
            count[1] = np.count_nonzero(targets)
            count[0] = targets.shape[0] - count[1]
            if targets.shape[0] != 0:
                current_node.value = np.array(count).argmax()
            else:
                current_node.value = random.randint(0, 1)
            return 

        if (np.array(targets) == 0).all():
            current_node.value = 0
            return
        elif (np.array(targets) == 1).all():
            current_node.value = 1
            return
        
        # A ← The Attribute that best classifies examples.
        ig_arr = []
        best_attribute_index = unused_f[0]
        idx = 0
        for i in unused_f:
            ig = information_gain(features, i, targets)
            ig_arr.append(ig)
            
        max_gain_value = max(ig_arr)
        idx = ig_arr.index(max_gain_value)
        best_attribute_index = unused_f[idx]
        best_attribute_name = self.attribute_names[best_attribute_index]

        # Decision Tree attribute for Root = A.
        current_node.attribute_index = best_attribute_index
        current_node.attribute_name = best_attribute_name
        update_unused_f = unused_f[unused_f != best_attribute_index]
        
        # Add a new tree branch below Root, corresponding to the test A = vi.
        neg = Node()
        pos = Node()
        neg.value = random.randint(0, 1)
        pos.value = random.randint(0, 1)
        current_node.branches.append(neg)
        current_node.branches.append(pos)

        # Let Examples(vi) be the subset of examples that have the value v0 and v1 for A
        fv0, fv1, target_fv0, target_fv1 = self.split(features, best_attribute_index, targets)
       
        # If Examples(vi) is not empty
        if (target_fv0.shape[0] != 0): # num of examples with fv0 != 0
            if (np.array(target_fv0) == 0).all():
                neg.value = 0
            elif (np.array(target_fv0) == 1).all():
                neg.value = 1
            else:
                self.add_node(fv0, target_fv0, update_unused_f, neg)
        
        if (target_fv1.shape[0] != 0):
            if (np.array(target_fv1) == 0).all():
                pos.value = 0
            elif (np.array(target_fv1) == 1).all():
                pos.value = 1
            else:
                self.add_node(fv1, target_fv1, update_unused_f, pos)
        
    def predict(self, features):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        Outputs:
            predictions (np.array): numpy array of size N array which has the predicitons 
            for the input data.
        """
        self._check_input(features)

        # raise NotImplementedError()

        predictions = np.array([])
        for i in features:
            node = self.tree
            while len(node.branches) != 0:
                if i[node.attribute_index] == 0:
                    node = node.branches[0]
                else:
                    node = node.branches[1]
            predictions = np.append(predictions, node.value)

        return predictions

    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        val = tree.value if tree.value is not None else 9
        print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))

    def visualize(self, branch=None, level=0):
        """
        Visualization of a decision tree. Implemented for you to check your work and to
        use as an example of how to use the given classes to implement your decision
        tree.
        """
        if not branch:
            branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level+1)

def information_gain(features, attribute_index, targets):
    """
    TODO: Implement me!

    Information gain is how a decision tree makes decisions on how to create
    split points in the tree. Information gain is measured in terms of entropy.
    The goal of a decision tree is to decrease entropy at each split point as much as
    possible. This function should work perfectly or your decision tree will not work
    properly.

    Information gain is a central concept in many machine learning algorithms. In
    decision trees, it captures how effective splitting the tree on a specific attribute
    will be for the goal of classifying the training data correctly. Consider
    data points S and an attribute A. S is split into two data points given binary A:

        S(A == 0) and S(A == 1)

    Together, the two subsets make up S. If A was an attribute perfectly correlated with
    the class of each data point in S, then all points in a given subset will have the
    same class. Clearly, in this case, we want something that captures that A is a good
    attribute to use in the decision tree. This something is information gain. Formally:

        IG(S,A) = H(S) - H(S|A)

    where H is information entropy. Recall that entropy captures how orderly or chaotic
    a system is. A system that is very chaotic will evenly distribute probabilities to
    all outcomes (e.g. 50% chance of class 0, 50% chance of class 1). Machine learning
    algorithms work to decrease entropy, as that is the only way to make predictions
    that are accurate on testing data. Formally, H is defined as:

        H(S) = sum_{c in (classes in S)} -p(c) * log_2 p(c)

    To elaborate: for each class in S, you compute its prior probability p(c):

        (# of elements of class c in S) / (total # of elements in S)

    Then you compute the term for this class:

        -p(c) * log_2 p(c)

    Then compute the sum across all classes. The final number is the entropy. To gain
    more intution about entropy, consider the following - what does H(S) = 0 tell you
    about S?

    Information gain is an extension of entropy. The equation for information gain
    involves comparing the entropy of the set and the entropy of the set when conditioned
    on selecting for a single attribute (e.g. S(A == 0)).

    For more details: https://en.wikipedia.org/wiki/ID3_algorithm#The_ID3_metrics

    Args:
        features (np.array): numpy array containing features for each example.
        attribute_index (int): which column of features to take when computing the
            information gain
        targets (np.array): numpy array containing labels corresponding to each example.

    Output:
        information_gain (float): information gain if the features were split on the
            attribute_index.
    """
    targets = np.array(targets).astype(int)
    # occ_arr = np.bincount(targets)
    # p_arr = occ_arr/ targets.shape[0]
    # term_arr = np.array([- i * np.log2(i) if i != 0 else 0 for i in p_arr])
    # H_current = term_arr.sum()

    p1 = np.count_nonzero(targets) / len(targets)
    p0 = 1 - p1
    H_current = -p1 * np.log2(p1) - p0 * np.log2(p0)
      
    # Entropy for particular feature

    feature = features[:, attribute_index]
    feature = np.array(feature).astype(int)

    H_arr = np.zeros(2)
    for j in range(2): # for each feature value
        # for a particular feature value, select those in class i
        targets_split = np.array(targets[feature.flatten() == j])
        if (targets_split.shape[0] != 0):
            occ_arr_for_v = np.bincount(targets_split)
            p_arr_for_v = occ_arr_for_v/ targets_split.shape[0]
            term_arr_for_v = np.array([- i * np.log2(i) if i != 0 else 0 for i in p_arr_for_v])
            H_arr[j] = term_arr_for_v.sum()

    occ_arr_for_feature = np.bincount(feature)
    weight_arr = np.array([0, 0])
    weight_arr = occ_arr_for_feature / feature.shape[0]

    if weight_arr.shape[0] == 1:
        weight_arr = np.append(weight_arr, [0])
    if weight_arr.shape[0] == 0:
        weight_arr = np.append(weight_arr, [0, 0])
   
    weighted_H = weight_arr * H_arr
    info_gain = H_current - weighted_H.sum()    
    
    return info_gain


if __name__ == '__main__':
    # construct a fake tree
    attribute_names = ['larry', 'curly', 'moe']
    decision_tree = DecisionTree(attribute_names=attribute_names)
    while len(attribute_names) > 0:
        attribute_name = attribute_names[0]
        if not decision_tree.tree:
            decision_tree.tree = Node(
                attribute_name=attribute_name,
                attribute_index=decision_tree.attribute_names.index(attribute_name),
                value=0,
                branches=[]
            )
        else:
            decision_tree.tree.branches.append(
                Node(
                    attribute_name=attribute_name,
                    attribute_index=decision_tree.attribute_names.index(attribute_name),
                    value=0,
                    branches=[]
                )
            )
        attribute_names.remove(attribute_name)
    decision_tree.visualize()
