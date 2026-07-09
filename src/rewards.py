from src.actions import Climax_Action

class Reward_Calculator:
    def __init__(self, debug=False):
        self.debug = debug
        self.reset()
        
    def reset(self):
        self.prev_p1_hp = 300.0
        self.prev_p2_hp = 300.0
        self.prev_p1_guard = 100.0
        self.prev_p2_guard = 100.0
        self.prev_p1_rider = 0.0
        self.prev_p2_rider = 0.0
        self.prev_combo_count = 0
        self.p1_form_changed = False
        self.p1_finisher_attempted = False
        self.special_active_steps = 0
        self.special_hit_detected = False
        self.kick_active_steps = 0
        self.kick_hit_detected = False
        
    def calculate_reward(self, p1_hp, p2_hp, p1_guard, p2_guard, p1_rider, p2_rider, combo_count, p1_rounds, p2_rounds, is_infinite, last_action, prev_action, round_steps, opponent_finisher_connected=False):
        # 1. HP damage dealt (to P2) vs taken (by P1)
        damage_dealt = max(0.0, self.prev_p2_hp - p2_hp)
        damage_taken = max(0.0, self.prev_p1_hp - p1_hp)
        
        # Supercharge damage dealt rewards
        if damage_dealt > 0:
            if last_action == Climax_Action.RIDER_FINALE:
                damage_dealt_reward = (damage_dealt * 1.5) + 8.0  # Finisher hit payoff (scaled down from 3x + 25)
            elif p2_guard == 0.0:
                damage_dealt_reward = damage_dealt * 1.2  # No Mercy: slight boost for hitting guard-broken state
            else:
                damage_dealt_reward = damage_dealt * 1.0
        else:
            damage_dealt_reward = 0.0
            
        reward = damage_dealt_reward - (damage_taken * 1.2)
        
        # 2. Guard Gauge change (shield management)
        guard_dealt = max(0.0, self.prev_p2_guard - p2_guard)
        reward += guard_dealt * 0.1
        
        # Guard Crush bonus
        if p2_guard == 0.0 and self.prev_p2_guard > 0.0:
            reward += 5.0
        
        # AI's guard gauge reduction
        guard_taken = max(0.0, self.prev_p1_guard - p1_guard)
        if guard_taken > 0:
            if damage_taken > 0:
                reward -= guard_taken * 0.05  # Failed block / hit
            else:
                reward += guard_taken * 0.05  # Successful block
                
        # Guard Crush penalty
        if p1_guard == 0.0 and self.prev_p1_guard > 0.0:
            reward -= 5.0
                
        # 3. Rider Gauge change & Potential
        rider_gained = max(0.0, p1_rider - self.prev_p1_rider)
        reward += rider_gained * 0.30  # Encourage active charging
        
        # Minor dodge action cost to prevent infinite evasion spam
        if last_action in [Climax_Action.EVADE_LEFT, Climax_Action.EVADE_RIGHT]:
            reward -= 0.08

        # Heavy penalty for getting hit while charging (forces charging at a safe distance)
        if last_action == Climax_Action.CHARGE_GAUGE and damage_taken > 0:
            reward -= 15.0
            if self.debug:
                print(f"[Reward] Hit while charging penalty! damage_taken={damage_taken:.1f}. -15.0 penalty.")

        # Cancel/Quick Step logic:
        # - If done during an attack: It's a Rider Cancel (consumes 1 bar of meter, costs -0.5).
        # - If done in neutral: It's a Quick Step (free movement, minor cost -0.08 to prevent spam, same as dodge).
        if last_action in [Climax_Action.CANCEL_RIGHT, Climax_Action.CANCEL_LEFT]:
            if prev_action in [
                Climax_Action.LIGHT, Climax_Action.HEAVY, Climax_Action.SPECIAL, 
                Climax_Action.NORMAL_FINISHER, Climax_Action.RIDER_FINALE,
                Climax_Action.RUNNING_LIGHT_RIGHT, Climax_Action.RUNNING_LIGHT_LEFT,
                Climax_Action.RUNNING_HEAVY_RIGHT, Climax_Action.RUNNING_HEAVY_LEFT,
                Climax_Action.LIGHT_DOWN, Climax_Action.HEAVY_DOWN, Climax_Action.SPECIAL_DOWN, Climax_Action.FINISHER_DOWN,
                Climax_Action.LIGHT_RIGHT, Climax_Action.LIGHT_LEFT,
                Climax_Action.HEAVY_RIGHT, Climax_Action.HEAVY_LEFT,
                Climax_Action.SPECIAL_RIGHT, Climax_Action.SPECIAL_LEFT
            ]:
                reward -= 0.5  # Attack Cancel (Rider Cancel) cost
                if self.debug:
                    print("[Reward] Attack cancel executed. -0.5 cost.")
            else:
                reward -= 0.08  # Quick Step in neutral (free movement)
                if self.debug:
                    print("[Reward] Quick Step in neutral. -0.08 cost.")

        # Track special move (Cross / Xbox A) hit success window (punishes mindless spamming/whiffs)
        if last_action in [Climax_Action.SPECIAL, Climax_Action.SPECIAL_DOWN, Climax_Action.SPECIAL_RIGHT, Climax_Action.SPECIAL_LEFT]:
            if self.special_active_steps == 0:
                self.special_active_steps = 30  # 1.0 second window at 30fps
                self.special_hit_detected = False
                
        if self.special_active_steps > 0:
            self.special_active_steps -= 1
            if damage_dealt > 0:
                self.special_hit_detected = True
            if self.special_active_steps == 0 and not self.special_hit_detected:
                reward -= 2.5
                if self.debug:
                    print("[Reward] Special move failed to hit (whiff/block)! -2.5 penalty.")

        # Track Rider Kick (D-pad Up + Circle / Xbox B) hit success window
        if last_action == Climax_Action.RIDER_KICK:
            if self.kick_active_steps == 0:
                self.kick_active_steps = 45  # 1.5 second window at 30fps
                self.kick_hit_detected = False
                
        if self.kick_active_steps > 0:
            self.kick_active_steps -= 1
            if damage_dealt > 0:
                self.kick_hit_detected = True
            if self.kick_active_steps == 0 and not self.kick_hit_detected:
                reward -= 3.0
                if self.debug:
                    print("[Reward] Rider Kick failed to hit (whiff/block)! -3.0 penalty.")

        # Punish getting hit by opponent's Rider Finale (finisher)
        if opponent_finisher_connected:
            reward -= 30.0
            if self.debug:
                print("[Reward] Smacked by Opponent's Rider Finale! -30.0 penalty.")

        # Form Change (L2) and Finisher (R2) attempt rewards when meter is full
        if self.prev_p1_rider >= 95.0:
            if last_action == Climax_Action.FORM_CHANGE:
                reward += 5.0
                if self.debug:
                    print("[Reward] Form Change triggered with full meter! +5.0 bonus.")
            elif last_action == Climax_Action.RIDER_FINALE:
                reward += 5.0
                if self.debug:
                    print("[Reward] Rider Finale triggered with full meter! +5.0 bonus.")
        
        # 4. Combo bonus
        if combo_count > 0:
            reward += combo_count * 0.1
            
        # Milestone combo bonus for hitting 10+ combo sequence (only rewarded once per sequence)
        if combo_count >= 10 and self.prev_combo_count < 10:
            reward += 5.0
            if self.debug:
                print(f"[Reward] Milestone Combo reached! {combo_count} hits! +5.0 bonus.")
            
        # 5. Red Shoes System (forced berserk combat fail-safe when trailing in HP near round end)
        # Inspired by the Kabuto Zecter's secret berserk program. As Hiyori is half-Worm, 
        # this system turns the agent into a relentless attacker to survive.
        if not is_infinite:
            time_left = max(0.0, 99.0 - (round_steps / 30.0))
            if time_left < 20.0 and p1_hp < p2_hp:
                # Double all damage dealt rewards (berserk mode)
                if damage_dealt > 0:
                    reward += damage_dealt_reward * 1.0
                
                hp_deficit = p2_hp - p1_hp
                deficit_penalty = hp_deficit * 0.05
                
                if p2_rounds > p1_rounds:
                      deficit_penalty *= 2.0
                      
                reward -= deficit_penalty
            
        # Update HP and Gauge memory
        self.prev_p1_hp = p1_hp
        self.prev_p2_hp = p2_hp
        self.prev_p1_guard = p1_guard
        self.prev_p2_guard = p2_guard
        self.prev_p1_rider = p1_rider
        self.prev_p2_rider = p2_rider
        self.prev_combo_count = combo_count
        
        return reward
