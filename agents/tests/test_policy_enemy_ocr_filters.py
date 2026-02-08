from agents.policy_agent import PolicyAgent


def test_enemies_present_ignores_quest_header_compact_text():
    agent = PolicyAgent()
    agent.latest_state = {"text": ["ENEMYATTHE GATE"], "enemies": []}
    assert agent._enemies_present() is False


def test_enemies_present_detects_short_enemy_label():
    agent = PolicyAgent()
    agent.latest_state = {"text": ["ENEMY"], "enemies": []}
    assert agent._enemies_present() is True
