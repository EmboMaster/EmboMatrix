try:
    from omnigibson.tasks.behavior_task import BehaviorTask
except Exception as e:
    print(f"An error occurred: {e}")
from omnigibson.tasks.dummy_task import DummyTask
from omnigibson.tasks.grasp_task import GraspTask
from omnigibson.tasks.grasp_behavior_task import GraspBehaviorTask
from omnigibson.tasks.point_navigation_task import PointNavigationTask
from omnigibson.tasks.point_reaching_task import PointReachingTask
from omnigibson.tasks.task_base import REGISTERED_TASKS
