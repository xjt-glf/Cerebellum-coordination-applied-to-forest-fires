"""完整环境输入回归：完整格数、真值/认知区别、真实发送文本。"""
import json
import unittest

import numpy as np

from lowlevel.env_wrapper import TaskConditionedEnv
from lowlevel.mission_demo import sense_state
from webapp.brains import LLMBrain
from webapp.world_snapshot import full_snapshot


class FullInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = TaskConditionedEnv(num_uavs=4, num_rescuers=10, seed=20000,
                                    replan_every=0)
        cls.env.reset()
        cls.full = full_snapshot(cls.env)

    def test_all_cells_and_exact_fire_values(self):
        maps = self.full["maps"]
        for name in ["true_fire", "team_known_fire", "personnel_shared_known_fire",
                     *(f"uav_{i}_known_fire" for i in range(4))]:
            self.assertEqual(len(maps[name]["rows_x"]), 100)
            self.assertTrue(all(len(row) == 100 for row in maps[name]["rows_x"]))
        decoded = np.array([[v for v, count in row for _ in range(count)]
                            for row in maps["true_burning_values"]["rows_x_value_count"]])
        np.testing.assert_array_equal(decoded, self.env.env.grid["burning_state"])
        self.assertNotIn("?", "".join(maps["true_fire"]["rows_x"]))
        self.assertIn("?", "".join(maps["team_known_fire"]["rows_x"]))

    def test_actual_prompt_contains_full_data_and_toggle(self):
        s = sense_state(self.env)
        s["full_environment"] = self.full
        brain = LLMBrain("http://unused", "private-test-key", "test")
        prompts = []
        def capture(prompt):
            prompts.append(prompt)
            return json.dumps({"uavs": [{"task": "explore"} for _ in range(4)]})
        brain._call = capture
        _, _, meta = brain.plan(s)
        self.assertEqual(meta["truth_cells"], 10000)
        for name in ["true_burning_values", "uav_3_known_fire", "terrain_height",
                     "personnel", "recent_decisions", "current_order"]:
            self.assertIn(name, prompts[-1])
        self.assertNotIn("private-test-key", prompts[-1])
        brain.full_state = False
        _, _, meta = brain.plan(s)
        self.assertEqual(meta["input_mode"], "brief")
        self.assertNotIn("true_burning_values", prompts[-1])


if __name__ == "__main__":
    unittest.main()
