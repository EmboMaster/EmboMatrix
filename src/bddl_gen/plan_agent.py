import  bddl_modify_object
import bddl_subtask_gen
import bddl_subtask_check

class PlanAgent:
    def __init__(self, path='bddl_data'):
        self.path = path
    
    def plan_task(self,task=""):
        bddl_files = bddl_modify_object.generate_bddl(task,self.path)
        subtask_files = bddl_subtask_gen.bddl_subtask_gen(bddl_files,self.path)
        return subtask_files
    
    def fix_task(self,reply):
        filename = f'{self.path}/txt/subtask_bddl.txt'
        with open(filename, 'r', encoding='utf-8') as file:
            subtask_files = file.read()
        new_subtask_files = bddl_subtask_gen.bddl_subtask_fix(subtask_files,reply,self.path)
        return new_subtask_files
        