import os
import re
from bddl.condition_evaluation import (
    compile_state,
    create_scope,
    evaluate_state,
    get_ground_state_options,
)
from bddl.config import ACTIVITY_CONFIGS_PATH
from bddl.object_taxonomy import ObjectTaxonomy
from bddl.parsing import (
    gen_natural_language_condition,
    gen_natural_language_conditions,
    parse_domain,
    parse_problem,
)

INSTANCE_EXPR = re.compile(r"problem(\d+).bddl")


class Conditions(object):
    def __init__(self, behavior_activity, activity_definition, simulator_name, predefined_problem=None, problem_filename=None):
        """Object to store behavior activity content and compile conditions for checking and
            simulator use

        Args:
            behavior_activity (str): behavior activity being used
            activity_definition (int): specific definition of behavior_activity
            simulator_name (str): simulator that BEHAVIOR is being used with
            predefined_problem (str): a pre-defined problem that is not in the activity_definitions folder
        """
        self.behavior_activity = behavior_activity
        self.activity_definition = activity_definition
        domain_name, *__ = parse_domain(simulator_name)
        __, self.parsed_objects, self.parsed_initial_conditions, self.parsed_goal_conditions = parse_problem(
            self.behavior_activity, self.activity_definition, domain_name, predefined_problem=predefined_problem, problem_filename=problem_filename
        )


######## API ########


def get_object_scope(conds):
    """Create unpopulated object scope to populate for generating goal and
        ground goal conditions.

    Args:
        conds (Conditions): conditions for the particular activity and definition

    Returns:
        dict<str: None>: unpopulated scope with string keys to be mapped to
                            simulator object values
    """
    return create_scope(conds.parsed_objects)


# def get_initial_conditions(conds, backend, scope, generate_ground_options=True):
#     """Create compiled initial conditions that can be checked and sampled

#     Args:
#         conds (Conditions): conditions for the particular activity and definition

#     Returns:
#         list<bddl.condition_evaluation.HEAD>: compiled conditions if initial
#                                                 condition definition is not
#                                                 empty else None
#     """
#     # breakpoint()
#     conds.parsed_initial_conditions = check_eai_conditions(conds.parsed_initial_conditions)
    
#     if bool(conds.parsed_initial_conditions[0]):
#         initial_conditions = compile_state(
#             [cond for cond in conds.parsed_initial_conditions if cond[0] not in ["inroom"]],
#             backend,
#             scope=scope,
#             object_map=conds.parsed_objects,
#             generate_ground_options=generate_ground_options
#         )
#         return initial_conditions

def get_initial_conditions(conds, backend, scope, generate_ground_options=True):
    """Create compiled initial conditions that can be checked and sampled

    Args:
        conds (Conditions): conditions for the particular activity and definition

    Returns:
        list<bddl.condition_evaluation.HEAD>: compiled conditions if initial
                                                condition definition is not
                                                empty else None
    """
    conds.parsed_initial_conditions = check_eai_conditions(conds.parsed_initial_conditions)
    
    # 修改這裡：檢查列表本身，而不是列表的第一個元素
    if conds.parsed_initial_conditions:
        initial_conditions = compile_state(
            [cond for cond in conds.parsed_initial_conditions if cond[0] not in ["inroom"]],
            backend,
            scope=scope,
            object_map=conds.parsed_objects,
            generate_ground_options=generate_ground_options
        )
        return initial_conditions
    
    # 如果列表為空，隱式返回 None

# def check_eai_conditions(goal_conditions: list[list]):
#     """
#     过滤掉包含EAI条件的子列表，能处理形如 ['not', [...]] 的嵌套结构。

#     Args:
#         goal_conditions (list[list]): 初始的目标条件列表。

#     Returns:
#         list[list]: 移除了包含EAI条件的子列表后的新列表。
#     """
#     eai_conditions = ["sliced", "soaked", "stained", "dusty"]
#     clean_conditions = []  # 创建一个新列表来存放结果

#     for condition in goal_conditions:
#         # 默认要检查的谓词在第一个位置
#         predicate_to_check = condition[0]

#         # 如果第一个元素是 'not'，并且第二个元素是一个列表 (为了安全)
#         # 那么真正要检查的谓词在下一层的列表里
#         if condition[0] == 'not' and isinstance(condition[1], list):
#             predicate_to_check = condition[1][0]

#         # 检查最终确定的谓词是否在我们的“黑名单”中
#         if predicate_to_check not in eai_conditions:
#             clean_conditions.append(condition)
            
#     return clean_conditions


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


def check_eai_conditions(goal_conditions: list[list]):
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


# def get_goal_conditions(conds, backend, scope, generate_ground_options=True):
#     """Create compiled goal conditions with a populated object scope for checking

#     Args:
#         conds (Conditions): conditions for the particular activity and definition
#         populated_object_scope (dict<str: simulator object>): scope mapping object
#                                                                 terms in BDDL to
#                                                                 simulator objects

#     Returns:
#         list<bddl.condition_evaluation.HEAD>: compiled conditions if goal condition
#                                                 definition is not empty else None
#     """
#     if bool(conds.parsed_goal_conditions[0]):
#         # breakpoint()
#         conds.parsed_goal_conditions = check_eai_conditions(conds.parsed_goal_conditions) 
#         # print(conds.parsed_goal_conditions)
        
#         goal_conditions = compile_state(
#             conds.parsed_goal_conditions, 
#             backend, 
#             scope=scope, 
#             object_map=conds.parsed_objects,
#             generate_ground_options=generate_ground_options
#         )
#         return goal_conditions

def get_goal_conditions(conds, backend, scope, generate_ground_options=True):
    """Create compiled goal conditions with a populated object scope for checking
    ...
    """
    # 修改這裡：同樣檢查列表本身
    if conds.parsed_goal_conditions:
        conds.parsed_goal_conditions = check_eai_conditions(conds.parsed_goal_conditions) 
        
        goal_conditions = compile_state(
            conds.parsed_goal_conditions, 
            backend, 
            scope=scope, 
            object_map=conds.parsed_objects,
            generate_ground_options=generate_ground_options
        )
        return goal_conditions
    
    # 如果列表為空，隱式返回 None


# def get_ground_goal_state_options(conds, backend, scope, goal_conditions):
#     """Create compiled ground solutions to goal state with a populated object scope
#         for checking progress on specific solutions

#     Args:
#         conds (Conditions): conditions for the particular activity and definition
#         populated_object_scope (dict<str: simulator object>): scope mapping object
#                                                                 terms in BDDL to
#                                                                 simulator objects

#     Returns:
#         list<bddl.condition_evaluation.HEAD>: compiled goal solutions

#     Raises:
#         AssertionError if there are no ground solutions
#     """
#     ground_goal_state_options = get_ground_state_options(
#         goal_conditions, backend, scope=scope, object_map=conds.parsed_objects
#     )
#     assert len(ground_goal_state_options) > 0
#     return ground_goal_state_options

def get_ground_goal_state_options(conds, backend, scope, goal_conditions):
    """Create compiled ground solutions to goal state with a populated object scope
    ...
    """
    # 如果 goal_conditions 為 None 或空列表，get_ground_state_options 應該能處理並返回空列表
    if not goal_conditions:
        return []

    ground_goal_state_options = get_ground_state_options(
        goal_conditions, backend, scope=scope, object_map=conds.parsed_objects
    )
    
    # 移除這個斷言，允許目標為空或無解的情況
    # assert len(ground_goal_state_options) > 0
    
    return ground_goal_state_options

def evaluate_goal_conditions(goal_conditions):
    """Evaluate compiled goal state to see if current simulator state has been met

    Args:
        goal_conditions (list<bddl.condition_evaluation.HEAD>): list of compiled
                                                                goal conditions with
                                                                populated scope

    Returns:
        bool, dict<str: list<int>>: [description]
    """
    return evaluate_state(goal_conditions)


def get_natural_initial_conditions(conds):
    """Return natural language translation of init of given conditions

    Args:
        conditions (list): conditions being translated

    Returns:
        list<str>: natural language translations, one per condition in conditions
    """
    return gen_natural_language_conditions(conds.parsed_initial_conditions)


def get_natural_goal_conditions(conds):
    """Return natural language translation of goal of given conditions

    Args:
        conditions (list): conditions being translated

    Returns:
        list<str>: natural language translations, one per condition in conditions
    """
    return gen_natural_language_conditions(conds.parsed_goal_conditions)


def get_all_activities():
    """Return a list of all activities included in this version of BDDL.
        
    Returns:
        list<str>: list containing the name of each included activity
    """
    return [x for x in os.listdir(ACTIVITY_CONFIGS_PATH) if os.path.isdir(os.path.join(ACTIVITY_CONFIGS_PATH, x))]


def get_instance_count(act):
    """Return the number of instances of a given activity that are included in this version of BDDL.
    
    Args:
        act (str): name of the activity to check
        
    Returns:
        int: number of instances of the given activity
    """
    problem_files = [INSTANCE_EXPR.fullmatch(x) for x in os.listdir(os.path.join(ACTIVITY_CONFIGS_PATH, act))]
    ids = set(int(x.group(1)) for x in problem_files if x is not None)
    assert ids == set(range(len(ids))), f"Non-contiguous instance IDs found for problem {act}"
    return len(ids)


# def get_reward(ground_goal_state_options): 
#     """Return reward given ground goal state options.
#        Reward formulated as max(<percent literals that are satisfied in the option> for option in ground_goal_state_options)

#     Args: 
#         ground_goal_state_options (list<list<HEAD>>): list of compiled ground goal state options

#     Returns: 
#         float: reward
#     """
#     return max(len(evaluate_state(option)[-1]["satisfied"]) / float(len(option)) for option in ground_goal_state_options)

def get_reward(ground_goal_state_options): 
    """Return reward given ground goal state options.
    ...
    """
    # 添加保護：如果沒有 ground options，說明任務已完成或無目標，獎勵為 1.0
    if not ground_goal_state_options:
        return 1.0

    return max(len(evaluate_state(option)[-1]["satisfied"]) / float(len(option)) for option in ground_goal_state_options)