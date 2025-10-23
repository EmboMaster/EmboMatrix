# def contains_eai_predicate(condition: list, eai_keywords: set) -> bool:
#     """
#     递归地检查一个条件列表及其所有子列表中是否包含任何EAI关键词。

#     Args:
#         condition (list): 要检查的单个条件列表。
#         eai_keywords (set): EAI关键词的集合，使用集合查询更快。

#     Returns:
#         bool: 如果找到任何一个EAI关键词，则返回 True，否则返回 False。
#     """
#     # 基本情况：如果输入不是列表或为空列表，它不可能包含关键词。
#     if not isinstance(condition, list) or not condition:
#         return False

#     # 检查当前列表的第一个元素（谓词）
#     if condition[0] in eai_keywords:
#         return True

#     # 递归步骤：遍历列表中的其他元素
#     # 如果任何一个元素是列表，就递归地对它进行检查
#     for item in condition[1:]:
#         if contains_eai_predicate(item, eai_keywords):
#             # 只要在任何一个深层分支中找到，就立刻返回True
#             return True

#     # 如果遍历完所有元素及其子分支都没有找到，则返回False
#     return False

# def check_eai_conditions_recursive(goal_conditions: list[list]):
#     """
#     使用递归辅助函数来过滤掉包含EAI条件的子列表。

#     Args:
#         goal_conditions (list[list]): 初始的目标条件列表。

#     Returns:
#         list[list]: 移除了包含EAI条件的子列表后的新列表。
#     """
#     # 将关键词列表转换为集合，可以提高查询效率 (O(1) 平均时间复杂度)
#     eai_keywords = {"sliced", "soaked", "stained", "dusty"}
#     clean_conditions = []

#     for condition in goal_conditions:
#         # 使用递归辅助函数进行深度检查
#         # 如果整个条件分支中都不包含EAI关键词，才将其加入结果列表
#         if not contains_eai_predicate(condition, eai_keywords):
#             clean_conditions.append(condition)
            
#     return clean_conditions

# # --- 测试 ---
# test_input = [['forall', ['?toy_car.n.01', '-', 'toy_car.n.01'], ['and', ['open', '?toy_car.n.01'], ['not', ['dusty', '?toy_car.n.01']]]]]

# # 使用新的函数进行处理
# filtered_output = check_eai_conditions_recursive(test_input)

# print(f"输入: {test_input}")
# print(f"过滤后的输出: {filtered_output}")

# # 另一个测试用例
# test_input_2 = [
#     ['on', 'a', 'b'], 
#     ['not', ['soaked', 'c']], # 这个应该被移除
#     ['inside', 'd', 'e']
# ]
# filtered_output_2 = check_eai_conditions_recursive(test_input_2)
# print(f"\n输入2: {test_input_2}")
# print(f"过滤后的输出2: {filtered_output_2}")


def clean_and_rebuild_condition(condition, eai_keywords: set):
    """
    递归地遍历和重建条件。
    - 如果条件包含EAI关键词，则返回 None (表示移除)。
    - 否则，返回净化后的新条件列表。
    """
    # 1. 基本情况：如果不是列表，直接返回原样
    if not isinstance(condition, list) or not condition:
        return condition

    predicate = condition[0]
    
    predicate_to_check = predicate
    if predicate == 'not' and len(condition) > 1 and isinstance(condition[1], list):
        predicate_to_check = condition[1][0]

    # 2. 检查当前节点：如果谓词本身是禁用的，则整个分支被剪掉
    if predicate_to_check in eai_keywords:
        return None

    # 3. 递归重建
    rebuilt_condition = [predicate]
    for sub_condition in condition[1:]:
        cleaned_sub = clean_and_rebuild_condition(sub_condition, eai_keywords)
        if cleaned_sub is not None:
            rebuilt_condition.append(cleaned_sub)

    # 4. 逻辑简化
    # 4a. 处理 'and' 和 'or'
    if predicate in ('and', 'or'):
        if len(rebuilt_condition) == 2:
            return rebuilt_condition[1]
        if len(rebuilt_condition) == 1:
            return None
            
    # 4b. 【新增】处理量词 (forall, exists, etc.)
    # 一个有效的量词至少需要 `[谓词, 变量, 条件]` 3个部分
    quantifiers = {'forall', 'exists', 'forn'} # 您可以根据需要添加更多量词
    if predicate in quantifiers:
        # 如果净化后条件部分丢失，则移除整个量词表达式
        if len(rebuilt_condition) < 3:
            return None

    return rebuilt_condition


def modify_and_clean_conditions(goal_conditions: list[list]):
    """
    遍历所有顶层目标条件，并进行深度净化和重建。
    """
    eai_keywords = {"sliced", "soaked", "stained", "dusty"}
    final_conditions = []

    for condition in goal_conditions:
        cleaned_condition = clean_and_rebuild_condition(condition, eai_keywords)
        if cleaned_condition is not None:
            final_conditions.append(cleaned_condition)
            
    return final_conditions

# --- 测试 ---
test_input_bug = [['forall', ['?chanterelle.n.01', '-', 'chanterelle.n.01'], ['and', ['sliced', '?chanterelle.n.01'], ['inside', '?chanterelle.n.01', 'cooler.n.01_1']]], ['soaked', '?cooler.n.01_1']]

# 使用修正后的函数进行处理
modified_output_bug = modify_and_clean_conditions(test_input_bug)

print("输入 (之前有问题的例子):")
print(test_input_bug)
print("\n修正后的输出:")
print(modified_output_bug)
print("\n期望的输出:")
print([])

print("-" * 20)

# 再次验证之前的正常情况没有被破坏
test_input_normal = [['forall', ['?toy_car.n.01', '-', 'toy_car.n.01'], ['and', ['open', '?toy_car.n.01'], ['not', ['dusty', '?toy_car.n.01']]]]]
modified_output_normal = modify_and_clean_conditions(test_input_normal)
print("输入 (之前的正常例子):")
print(test_input_normal)
print("\n修正后的输出 (应保持不变):")
print(modified_output_normal)

# --- 测试 ---
test_input = [['forall', ['?toy_car.n.01', '-', 'toy_car.n.01'], ['and', ['open', '?toy_car.n.01'], ['not', ['dusty', '?toy_car.n.01']]]]]

# 使用新的函数进行处理
modified_output = modify_and_clean_conditions(test_input)

print("输入:")
print(test_input)
print("\n修改后的输出:")
print(modified_output)
print("\n期望的输出:")
print([['forall', ['?toy_car.n.01', '-', 'toy_car.n.01'], ['open', '?toy_car.n.01']]])

# 额外测试：如果 'open' 也是禁用词，整个 'and' 会被移除，导致 forall 也为空
print("-" * 20)
test_input_2 = [['forall', ['?toy_car.n.01', '-', 'toy_car.n.01'], ['and', ['soaked', '?toy_car.n.01'], ['not', ['dusty', '?toy_car.n.01']]]]]
modified_output_2 = modify_and_clean_conditions(test_input_2)
print("输入 (两个条件都是禁用词):")
print(test_input_2)
print("\n修改后的输出 (应该为空):")
print(modified_output_2)


test_input_2 = [
    ['on', 'a', 'b'], 
    ['not', ['soaked', 'c']], # 这个应该被移除
    ['inside', 'd', 'e']
]
filtered_output_2 = modify_and_clean_conditions(test_input_2)
print(f"\n输入2: {test_input_2}")
print(f"过滤后的输出2: {filtered_output_2}")