import numpy as np

from .single_cbf_qp import SingleCBF_QP
from .ecbf_qp import ECBF_Collision_QP


class SafetyShield:
    """
    Unified safety shield for RL-controlled agent.

    Modes:
        - "none"
        - "soft"
        - "single_cbf"
        - "ecbf_collision"

    All corrections are applied to the agent acceleration a_agent.
    """

    def __init__(self,
                 mode="none",
                 # ---- single CBF params ----
                 v_max=15.0,
                 v_min=3.0,
                 a_max=6.0,
                 alpha=1.0,
                 # ---- collision ECBF params ----
                 D_safe=5.0,
                 k1=2.0,
                 k0=1.0,
                 slack_weight=1000.0,
                 # ---- soft params ----
                 ttc_threshold=2.0,
                 soft_brake_gain=2.0):

        self.mode = mode.lower()

        # ------------------------------
        # Single-agent CBF
        # ------------------------------
        if self.mode == "single_cbf":
            self.single_cbf = SingleCBF_QP(
                v_max=v_max,
                v_min=v_min,
                a_max=a_max,
                alpha=alpha
            )

        # ------------------------------
        # Collision ECBF
        # ------------------------------
        if self.mode == "ecbf_collision":
            self.ecbf = ECBF_Collision_QP(
                D_safe=D_safe,
                k1=k1,
                k0=k0,
                a_max=a_max,
                slack_weight=slack_weight
            )

        # ------------------------------
        # Soft shield parameters
        # ------------------------------
        self.ttc_threshold = ttc_threshold
        self.soft_brake_gain = soft_brake_gain

    # ==========================================================
    # Public API
    # ==========================================================

    def apply(self,
              agent_pos,
              agent_vel,
              a_agent_des,
              others_pos=None,
              others_vel=None,
              others_acc=None):
        """
        agent_pos: (2,)
        agent_vel: (2,)
        a_agent_des: (2,)
        others_pos: list[(2,)]
        others_vel: list[(2,)]
        """
        if others_pos is None:
            others_pos = []
        if others_vel is None:
            others_vel = []

        # ------------------------------
        # 1) No safety
        # ------------------------------
        if self.mode == "none":
            return a_agent_des

        # ------------------------------
        # 2) Soft heuristic shield
        # ------------------------------
        if self.mode == "soft":
            return self._soft_shield(
                agent_pos,
                agent_vel,
                a_agent_des,
                others_pos,
                others_vel
            )

        # ------------------------------
        # 3) Single-agent dynamics CBF
        # ------------------------------
        if self.mode == "single_cbf":
            return self.single_cbf.solve(
                agent_vel,
                a_agent_des
            )

        # ------------------------------
        # 4) Collision ECBF
        # ------------------------------
        if self.mode == "ecbf_collision":
            return self.ecbf.solve(
                agent_pos,
                agent_vel,
                a_agent_des,
                others_pos,
                others_vel,
                others_acc
            )

        return a_agent_des

    # ==========================================================
    # Soft Shield (TTC-based heuristic)
    # ==========================================================

    def _soft_shield(self,
                     agent_pos,
                     agent_vel,
                     a_agent_des,
                     others_pos,
                     others_vel):

        if len(others_pos) == 0:
            return a_agent_des

        min_ttc = np.inf

        speed = np.linalg.norm(agent_vel) + 1e-6

        for p_o, v_o in zip(others_pos, others_vel):

            r = p_o - agent_pos
            v_rel = v_o - agent_vel

            dist = np.linalg.norm(r)
            closing_speed = -np.dot(r, v_rel) / (dist + 1e-6)

            if closing_speed > 0:
                ttc = dist / closing_speed
                min_ttc = min(min_ttc, ttc)

        if min_ttc >= self.ttc_threshold:
            return a_agent_des

        alpha = 1.0 - (min_ttc / self.ttc_threshold)
        alpha = np.clip(alpha, 0.0, 1.0)

        a_safe = a_agent_des.copy()

        # Assume forward direction aligned with velocity
        if speed > 1e-3:
            forward_dir = agent_vel / speed
        else:
            forward_dir = np.array([1.0, 0.0])

        brake = self.soft_brake_gain * alpha

        a_safe = a_safe - brake * forward_dir

        return a_safe