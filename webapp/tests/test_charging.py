"""返航回归：真实环境、新 Actor、无动作兜底。"""
import unittest
from pathlib import Path

import torch

from lowlevel.env_wrapper import TaskConditionedEnv
from lowlevel.models import Actor
from lowlevel.obs_builder import OBS_DIM, resolve_order_target
from lowlevel.task_spec import TASK_RECHARGE, UavOrder
from webapp.charge_protocol import ChargeProtocol


class ChargingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.env = TaskConditionedEnv(num_uavs=4, max_steps=100, replan_every=0,
                                      seed=20000, terminate_on_all_depleted=False)
        self.env.reset()

    def test_bad_base_target_charges_at_nearby_real_stations(self):
        env = self.env
        for u, xy in zip(env.env.uavs, [(80, 50), (50, 80), (20, 50), (50, 20)]):
            u.x, u.y, u.e = *xy, 70
        env.set_plan([UavOrder(task=TASK_RECHARGE, target=(0, 50)) for _ in range(4)])
        for i, order in enumerate(env.orders):
            self.assertNotEqual(resolve_order_target(env.env, i, order)[0], (0, 50))
        protocol = ChargeProtocol(env)
        obs, _ = protocol.sync()
        self.assertEqual(protocol.targets,
                         {0: (90., 50.), 1: (50., 90.), 2: (10., 50.), 3: (50., 10.)})
        root = Path(__file__).resolve().parents[2]
        sd = torch.load(root / "models/actor_selected.pt",
                        map_location="cpu", weights_only=True)
        actor = Actor(OBS_DIM, 9, hidden=384)
        actor.load_state_dict(sd)
        actor.eval()
        arrived = set()
        for _ in range(20):
            override, _ = protocol.sync()
            if override is not None:
                obs = override
            with torch.no_grad():
                actions = actor.act(torch.as_tensor(obs, dtype=torch.float32),
                                    deterministic=True)[0].numpy()
            obs, _, _, _ = env.step(actions)
            for i, u in enumerate(env.env.uavs):
                if u.recharged:
                    self.assertEqual(u.e, u.e_max)
                    arrived.add(i)
        self.assertEqual(arrived, set(range(4)))

    def test_emergency_order_precedes_depletion(self):
        env = self.env
        u = env.env.uavs[0]
        u.x, u.y, u.e = 80, 50, 40
        env.set_plan([UavOrder() for _ in range(4)])
        protocol = ChargeProtocol(env)
        protocol.sync()
        self.assertEqual(env.orders[0].task, TASK_RECHARGE)
        self.assertEqual(protocol.targets[0], (90., 50.))

    def test_zero_energy_recovers_only_at_charger(self):
        env = self.env
        u = env.env.uavs[0]
        u.x, u.y, u.e = 90, 50, 0
        env.step([0, 8, 8, 8])
        self.assertEqual((u.x, u.y, u.e), (90, 50, 600))
        u.x, u.y, u.e = 70, 50, 0
        env.step([0, 8, 8, 8])
        self.assertEqual((u.x, u.y, u.e), (70, 50, 0))


if __name__ == "__main__":
    unittest.main()
