import bddl_subtask_check
import check_feasibility

class ReviewerAgent:
    def __init__(self, path='bddl_data'):
        self.path = path
    
    def review_bddl(self,subtask_files):
        subtask_files_reply = bddl_subtask_check.bddl_subtask_check(subtask_files,self.path)
        final_reply = check_feasibility.check_feasibility(subtask_files_reply,self.path)
        return final_reply
    
    def save_bddl(self,subtask_files):
        subtask_files_reply = bddl_subtask_check.bddl_subtask_check(subtask_files,self.path)
        p,d = check_feasibility.save(subtask_files,subtask_files_reply,self.path)
        return p,d
    
    