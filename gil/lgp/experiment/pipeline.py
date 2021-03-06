import logging
import sys
import os
from os.path import dirname, realpath, join, expanduser
import pickle
import numpy as np
from datetime import datetime

from gil.lgp.logic.problem import Problem
from gil.lgp.utils.helpers import frozenset_of_tuples
from gil.lgp.core.dynamic import HumoroDynamicLGP

_path_file = dirname(realpath(__file__))
_domain_dir = join(_path_file, '../../../data', 'scenarios')
_dataset_dir = join(_path_file, '../../../datasets', 'mogaze')
_data_dir = join(_path_file, '../../../data', 'experiments')
_model_dir = join(expanduser("~"), '.qibullet', '1.4.3')
robot_model_file = join(_model_dir, 'pepper.urdf')


class Experiment(object):

    logger = logging.getLogger(__name__)
    def __init__(self, **kwargs):
        self.verbose = kwargs.get('verbose', False)
        self.task = kwargs.get('task', 'set_table')
        domain_file = kwargs.get('domain_file', join(_domain_dir, 'domain_' + self.task + '.pddl'))
        robot_model_file= kwargs.get('robot_model_file', join(_model_dir, 'pepper.urdf'))
        mogaze_dir = kwargs.get('mogaze_dir', _dataset_dir)
        self.data_dir = kwargs.get('data_dir', _data_dir)
        self.data_name = join(self.data_dir, self.task + str(datetime.now()))
        os.makedirs(self.data_dir, exist_ok=True)
        sim_fps = kwargs.get('sim_fps', 120)
        self.prediction = kwargs.get('prediction', False)
        self.engine = HumoroDynamicLGP(domain_file=domain_file, robot_model_file=robot_model_file, path_to_mogaze=mogaze_dir, 
                                       sim_fps=sim_fps, prediction=self.prediction, verbose=self.verbose, enable_viewer=True)
        # experiment params
        self.test_segments = kwargs.get('test_segments', None)  # test segments takes precedent
        self.total_pnp = kwargs.get('total_pnp', [4, 5, 6, 7])
        self.taskid = kwargs.get('taskid', [2, 3])  # set table for 2, 3 people
        self.human_carry = kwargs.get('human_carry', 3)
        self.trigger_period = kwargs.get('trigger_period', 10)
        self.start_agent_symbols = frozenset([('agent-avoid-human',), ('agent-free',)])
        self.end_agent_symbols = frozenset([('agent-at', 'table')])
        self.get_segments()
        # experiment storing
        self.segment_data = {}

    def get_segments(self):
        task_overlap = []
        self.segments = {}
        
        domain = self.engine.humoro_lgp.logic_planner.domain
        if self.test_segments is None:
            for i in self.taskid:
                segments = self.engine.hr.get_data_segments(taskid=i)
                for segment in segments:
                    objects = self.engine.hr.get_object_carries(segment, predicting=False)
                    n_carries = len(objects)
                    #print("Objects", objects)
                    if not self.prediction:
                        task_overlap.append(self.human_carry / n_carries)
                    else:
                        human_objects = set(self.engine.hr.get_object_carries(segment, predicting=True))
                        task_objects = set(objects)
                        task_overlap.append(len(human_objects.intersection(task_objects)) / len(human_objects.union(task_objects)))
                    if n_carries in self.total_pnp:
                        self.segments[segment] = Experiment.get_problem_from_segment(self.engine.hr, segment, domain, objects,
                                                                                    self.start_agent_symbols, self.end_agent_symbols)
        else:
            for segment in self.test_segments:
                objects = self.engine.hr.get_object_carries(segment, predicting=False)
                if not self.prediction:
                    task_overlap.append(self.human_carry / n_carries)
                else:
                    human_objects = set(self.engine.hr.get_object_carries(segment, predicting=True))
                    task_objects = set(objects)
                    task_overlap.append(len(human_objects.intersection(task_objects)) / len(human_objects.union(task_objects)))
                self.segments[segment] = Experiment.get_problem_from_segment(self.engine.hr, segment, domain, objects,
                                                                             self.start_agent_symbols, self.end_agent_symbols)
        Experiment.logger.info(f'Task IoU ratio: {np.mean(task_overlap)} +- {np.std(task_overlap)}')
    
    def save_data(self):
        with open(self.data_name, 'wb') as f:
            pickle.dump(self.segment_data, f)
    
    def run(self):

        for segment, problem in self.segments.items():
            # single plan
            print("segment", segment)
            print("problem", problem)
            self.engine.init_planner(segment=segment, problem=problem, 
                                     human_carry=self.human_carry, trigger_period=self.trigger_period,
                                     human_freq='human-at', traj_init='outer')
            single_success = self.engine.run(replan=False, sleep=False)
            # dynamic plan
            self.engine.init_planner(segment=segment, problem=problem, 
                                     human_carry=self.human_carry, trigger_period=self.trigger_period,
                                     human_freq='once', traj_init='nearest')
            dynamic_success = self.engine.run(replan=True, sleep=False)
            data = self.engine.get_experiment_data()
            data['single_success'] = single_success
            data['dynamic_success'] = dynamic_success
            self.segment_data[segment] = data
            self.engine.reset_experiment()

    @staticmethod
    def get_problem_from_segment(hr, segment, domain, objects, start_agent_symbols, end_agent_symbols):
        '''
        Infer problem from dataset, human prediction can differ from the goal here, which the robot has to adapt to solve this problem
        '''
        init_pred = hr.get_object_predicates(segment, 0, predicting=False)
        init_pred = [p for p in init_pred if p[1] in objects]
        # print("init pred", init_pred)
        final_pred = hr.get_object_predicates(segment, segment[2] - segment[1], predicting=False)
        final_pred = [p for p in final_pred if p[1] in objects]
        # print("final pred", final_pred)
        # resolve human-carry predicate
        for p in final_pred:
            if p[0] == 'human-carry':
                final_pred.remove(p)
                final_pred.append(('on', p[1], 'table'))  # assume on table at the end
        for p in init_pred:
            if p[0] == 'human-carry':
                for pr in final_pred:
                    if p[1] == pr[1]:
                        init_pred.remove(p)
                        init_pred.append(pr)
                        break
        # init problem
        problem = Problem()
        problem.name = str(segment)
        problem.domain_name = domain.name
        problem.objects = {'object': objects,  
                           'location': domain.constants['location']}
        problem.state = frozenset_of_tuples(init_pred).union(start_agent_symbols)
        problem.positive_goals = [frozenset_of_tuples(final_pred).union(end_agent_symbols)]
        problem.negative_goals = [frozenset()]
        return problem

        
