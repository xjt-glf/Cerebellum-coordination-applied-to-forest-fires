import copy
import math
import unittest

import numpy as np

from lowlevel.env_wrapper import TaskConditionedEnv
from webapp.personnel_control import PersonnelController


class PersonnelControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = TaskConditionedEnv(num_uavs=1, num_rescuers=2,
                                     n_fire_seeds=1, n_fire_seeds_range=None,
                                     spread_steps=0, replan_every=0, seed=812)
        cls.base.reset()
        e = cls.base.env
        e.grid['burning_state'][:] = 0
        e.rescuers_map_info[:] = 0
        e.team_map_info[:] = 0
        for u in e.uavs:
            u.map_info[:] = 0

    def setUp(self):
        self.w = copy.deepcopy(self.base)
        self.e = self.w.env
        self.c = PersonnelController(self.e)
        self.e.personnel_controller = self.c
        self.p = self.e.rescuers[0]

    def step(self, count):
        for _ in range(count):
            self.w.step([8])

    def test_move_survives_automatic_rebalance_and_holds_after_arrival(self):
        self.e.grid['burning_state'][24:28, 45:50] = 1
        self.e.rescuers_map_info[24:28, 45:50] = 1
        self.e.rebalance_every = 2
        target = [10., 45.]
        self.c.enqueue([{'person_ids': [1], 'task': 'move', 'target': target}], 0, 'codex')
        self.step(35)
        self.assertLess(math.dist((self.p.x, self.p.y), target), .8)
        self.assertEqual(self.c.active[1]['state'], 'completed')
        pos = (self.p.x, self.p.y)
        self.step(8)
        self.assertEqual((self.p.x, self.p.y), pos)

    def test_waypoint_sequence_then_manual_retarget(self):
        self.c.enqueue([{'person_ids': [1], 'task': 'move',
                         'waypoints': [[5, 48], [10, 52]]}], 0)
        self.step(35)
        self.assertLess(math.dist((self.p.x, self.p.y), (10, 52)), .8)
        self.assertTrue(any(x['event'] == 'waypoint_reached' for x in self.c.events))
        self.c.enqueue([{'person_ids': [1], 'task': 'retreat', 'target': [3, 50]}], self.w.t)
        self.step(25)
        self.assertLess(math.dist((self.p.x, self.p.y), (3, 50)), .8)

    def test_offline_new_order_waits_old_order_keeps_executing(self):
        self.c.enqueue([{'person_ids': [1], 'task': 'move', 'target': [10, 50]}], 0)
        self.p.is_covered = True
        self.c.prepare(0)
        original = self.c.active[1]['order_id']
        self.p.is_covered = False
        self.c.enqueue([{'person_ids': [1], 'task': 'retreat', 'target': [2, 48]}], 1)
        self.c.prepare(1)
        self.assertEqual(self.c.active[1]['order_id'], original)
        self.assertIn(1, self.c.pending)
        self.p.is_covered = True
        self.c.prepare(2)
        self.assertEqual(self.c.active[1]['task'], 'retreat')

    def test_atomic_validation_rejects_bad_ids_and_nonfinite_targets(self):
        for orders in [[{'person_ids': [0], 'task': 'move', 'target': [5, 5]}],
                       [{'person_ids': [1], 'task': 'move', 'target': [float('nan'), 5]}],
                       [{'person_ids': [1], 'task': 'hold'}, {'person_ids': [1], 'task': 'hold'}]]:
            with self.assertRaises(ValueError):
                self.c.enqueue(orders, 0)
            self.assertFalse(self.c.pending)

    def test_expired_unreceived_order_does_not_activate(self):
        self.p.is_covered = False
        self.c.enqueue([{'person_ids': [1], 'task': 'move', 'target': [5, 5],
                         'valid_for_ticks': 2}], 0)
        self.c.prepare(2)
        self.assertNotIn(1, self.c.active)
        self.p.is_covered = True
        self.c.prepare(3)
        self.assertNotIn(1, self.c.active)

    def test_release_restores_automatic_control(self):
        self.c.enqueue([{'person_ids': [1], 'task': 'hold'}], 0)
        self.step(2)
        self.assertIn(1, self.c.active)
        self.c.enqueue([{'person_ids': [1], 'task': 'auto'}], self.w.t)
        self.step(1)
        self.assertNotIn(1, self.c.active)

    def test_no_commands_preserves_legacy_evolution(self):
        legacy = copy.deepcopy(self.base)
        for _ in range(12):
            legacy.step([8])
            self.w.step([8])
            np.testing.assert_array_equal(self.e.grid['burning_state'], legacy.env.grid['burning_state'])
            self.assertEqual([(r.x, r.y, r.target_pos) for r in self.e.rescuers],
                             [(r.x, r.y, r.target_pos) for r in legacy.env.rescuers])

    def test_contain_works_at_west_perimeter_and_extinguishes(self):
        self.p.x, self.p.y = 12., 49.
        self.e.uavs[0].x, self.e.uavs[0].y = 10, 49
        self.e.grid['burning_state'][18:24, 46:53] = 1
        self.e.rescuers_map_info[18:24, 46:53] = 1
        self.c.enqueue([{'person_ids': [1], 'task': 'contain', 'target': [21, 49],
                         'approach_sector': 'west'}], 0)
        self.step(25)
        self.assertEqual(self.c.active[1]['task'], 'contain')
        self.assertLess(self.p.x, 21)
        self.assertGreater(np.count_nonzero(self.e.grid['burning_state'] == 3), 0)
        self.assertFalse(self.e.is_blocked_for_people(int(round(self.p.x)), int(round(self.p.y))))


if __name__ == '__main__':
    unittest.main()
