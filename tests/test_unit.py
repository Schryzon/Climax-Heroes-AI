import unittest
import sys
import os

# Add parent directory to path to resolve src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.actions import Climax_Action
from src.rewards import Reward_Calculator

class TestActionsAndRewards(unittest.TestCase):
    def test_action_dimensions(self):
        # We expect 31 discrete actions: index 0 to 30.
        self.assertEqual(len(Climax_Action), 31)
        self.assertEqual(Climax_Action.IDLE.value, 0)
        self.assertEqual(Climax_Action.SUPPORT.value, 6)
        self.assertEqual(Climax_Action.SPECIAL.value, 7)
        self.assertEqual(Climax_Action.RIDER_FINALE.value, 8)
        self.assertEqual(Climax_Action.RIDER_KICK.value, 21)
        self.assertEqual(Climax_Action.SPECIAL_DOWN.value, 24)
        self.assertEqual(Climax_Action.SPECIAL_RIGHT.value, 29)
        self.assertEqual(Climax_Action.SPECIAL_LEFT.value, 30)

    def test_reward_calculator(self):
        calc = Reward_Calculator(debug=True)
        # Test default reset values
        self.assertEqual(calc.support_active_steps, 0)
        self.assertFalse(calc.support_hit_detected)
        self.assertEqual(calc.round_support_hits, 0)
        self.assertEqual(calc.round_support_whiffs, 0)
        self.assertEqual(calc.round_kick_hits, 0)
        self.assertEqual(calc.round_kick_whiffs, 0)
        self.assertEqual(calc.round_red_shoes_steps, 0)
        self.assertEqual(calc.round_attack_cancels, 0)
        
        # Calculate a basic step reward with support move
        reward = calc.calculate_reward(
            p1_hp=300.0,
            p2_hp=300.0,
            p1_guard=100.0,
            p2_guard=100.0,
            p1_rider=50.0,
            p2_rider=50.0,
            combo_count=0,
            p1_rounds=0,
            p2_rounds=0,
            is_infinite=True,
            last_action=Climax_Action.SUPPORT,
            prev_action=Climax_Action.IDLE,
            round_steps=10,
            opponent_finisher_connected=False
        )
        # Used support move in normal form: should have a -0.2 penalty + 15.0 rider gain reward (50.0 * 0.3) = 14.8
        self.assertEqual(calc.support_active_steps, 29)
        self.assertAlmostEqual(reward, 14.8)
        
        # Test consecutive support spam penalty
        reward2 = calc.calculate_reward(
            p1_hp=300.0,
            p2_hp=300.0,
            p1_guard=100.0,
            p2_guard=100.0,
            p1_rider=50.0,
            p2_rider=50.0,
            combo_count=0,
            p1_rounds=0,
            p2_rounds=0,
            is_infinite=True,
            last_action=Climax_Action.SUPPORT,
            prev_action=Climax_Action.SUPPORT,
            round_steps=11,
            opponent_finisher_connected=False
        )
        # Support active steps should decrement to 28, and penalty should be -0.2 (normal form) - 0.5 (spam) = -0.7
        self.assertEqual(calc.support_active_steps, 28)
        self.assertAlmostEqual(reward2, -0.7)

if __name__ == "__main__":
    unittest.main()
