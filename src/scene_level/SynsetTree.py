import csv
from collections import defaultdict

# class TreeNode:
#     def __init__(self, name):
#         self.name = name
#         self.children = []

# class Tree:
#     def __init__(self, csv_file):
#         """
#         初始化树结构,构建节点并根据CSV文件建立父子关系。
#         :param csv_file: CSV文件路径
#         """
#         self.nodes = {}
#         self.roots = []
#         self.build_tree(csv_file)

#     def build_tree(self, csv_file):
#         """
#         从CSV文件构建树结构。
#         :param csv_file: CSV文件路径
#         """
#         parent_child_map = defaultdict(list)

#         # 读取CSV文件并建立父子关系的映射
#         with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
#             reader = csv.DictReader(file)
#             for row in reader:
#                 name = row['Name']
#                 parents = row['Parents'].split(',') if row['Parents'] else []
#                 children = row['Children'].split(',') if row['Children'] else []

#                 if name not in self.nodes:
#                     self.nodes[name] = TreeNode(name)

#                 for child in children:
#                     if child not in self.nodes:
#                         self.nodes[child] = TreeNode(child)

#                     # 将子节点添加到父节点的子节点列表中
#                     self.nodes[name].children.append(self.nodes[child])

#                 # 将父子关系记录下来
#                 for parent in parents:
#                     parent_child_map[parent].append(name)

#         # 构建树的根节点（我们假设根节点是没有父节点的那些节点）
#         self.roots = [node for node in self.nodes.values() if node.name not in parent_child_map]
import csv
from collections import defaultdict

class TreeNode:
    def __init__(self, name, state=None):
        """
        初始化树节点
        :param name: 节点名称
        :param state: 节点的状态
        """
        self.name = name
        self.children = []
        self.state = state  # 新增的state属性

class Tree:
    def __init__(self, csv_file):
        """
        初始化树结构,构建节点并根据CSV文件建立父子关系。
        :param csv_file: CSV文件路径
        """
        self.nodes = {}
        self.roots = []
        self.build_tree(csv_file)

    def build_tree(self, csv_file):
        """
        从CSV文件构建树结构。
        :param csv_file: CSV文件路径
        """
        parent_child_map = defaultdict(list)

        # 读取CSV文件并建立父子关系的映射
        with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                name = row['Name']
                state = row.get('Synset State', None)  # 获取state信息
                parents = row['Parents'].split(',') if row['Parents'] else []
                children = row['Children'].split(',') if row['Children'] else []

                # 如果节点不存在，则创建它，并设置state属性
                if name not in self.nodes:
                    self.nodes[name] = TreeNode(name, state)

                # 更新现有节点的state
                self.nodes[name].state = state

                for child in children:
                    if child not in self.nodes:
                        self.nodes[child] = TreeNode(child)

                    # 将子节点添加到父节点的子节点列表中
                    self.nodes[name].children.append(self.nodes[child])

                # 将父子关系记录下来
                for parent in parents:
                    parent_child_map[parent].append(name)

        # 构建树的根节点（我们假设根节点是没有父节点的那些节点）
        self.roots = [node for node in self.nodes.values() if node.name not in parent_child_map]

    def get_state_by_name(self, name):
        """
        根据节点名称获取节点的状态值。
        :param name: 节点名称
        :return: 节点的状态值，如果节点不存在则返回None
        """
        if name in self.nodes:
            return self.nodes[name].state
        else:
            return None

# 示例使用
# tree = Tree('example.csv')
# state = tree.get_state_by_name('NodeName')
# print(f"State of NodeName: {state}")
    def find_leaves(self, node):
        """
        寻找指定节点所在分支的所有叶子节点。
        :param node: 输入节点
        :return: 叶子节点的名称列表
        """
        if not node.children:  # 如果没有子节点，则是叶子节点
            return [node.name]
        leaves = []
        for child in node.children:
            leaves.extend(self.find_leaves(child))  # 递归查找子节点的叶子节点
        return leaves

    def get_leaves_by_name(self, name):
        """
        根据节点名称获取该节点所在分支的所有叶子节点。
        :param name: 节点名称
        :return: 叶子节点名称列表
        """
        if name not in self.nodes:
            return None  # 如果节点不存在，返回None
        node = self.nodes[name]
        return self.find_leaves(node)

    def print_children_by_name(self, name):
        """
        打印指定节点的所有子节点。
        :param name: 节点名称
        """
        if name not in self.nodes:
            print(f"节点 {name} 不存在.")
            return
        
        node = self.nodes[name]
        if not node.children:
            #print(f"节点 {name} 没有子节点.")
            return

        #print(f"节点 {name} 的子节点:")
        # for child in node.children:
        #     print(child.name)

    def find_terminal_leaves(self, node):
        """
        寻找指定节点所在分支的最末端叶子节点。
        :param node: 输入节点
        :return: 最末端叶子节点名称列表
        """
        # 如果节点没有子节点，则它本身就是叶子节点
        if not node.children:
            return [node.name.strip()]
        
        leaves = []
        for child in node.children:
            # 递归查找子节点的叶子节点
            leaves.extend(self.find_terminal_leaves(child))
        return leaves

    def get_terminal_leaves_by_name(self, name):
        """
        根据节点名称获取该节点所在分支的最末端叶子节点。
        :param name: 节点名称
        :return: 最末端叶子节点名称列表
        """
        if name not in self.nodes:
            return None  # 如果节点不存在，返回None
        node = self.nodes[name]
        return self.find_terminal_leaves(node)
    
# tree = Tree("/Users/asuka/codes/rearrange/DistributeAgent/BEHAVIOR-1KSynsets.csv")
# print("terminal Node",tree.get_terminal_leaves_by_name("vanilla.n.02"))
# print(f"get state of iron.n.04:{tree.get_state_by_name('iron.n.04')}")