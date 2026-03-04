import numpy as np
import osqp
import scipy.sparse as sp


class SingleCBF_QP:

    def __init__(self,
                 v_max=15.0,
                 v_min=3.0,
                 a_max=6.0,
                 alpha=1.0):

        self.v_max = v_max
        self.v_min = v_min
        self.a_max = a_max
        self.alpha = alpha

    def solve(self, agent_vel, a_agent_des, v_max=None):

        if v_max is None:
            v_max = self.v_max

        v = np.linalg.norm(agent_vel)
        dim = 2

        P = sp.csc_matrix(np.eye(dim))
        q = np.asarray(-a_agent_des, dtype=np.float64)

        A_list = []
        l_list = []
        u_list = []

        if v > 1e-3:

            v_hat = agent_vel / v

            # Upper speed limit
            A_list.append(v_hat.reshape(1, -1))
            l_list.append(-np.inf)
            u_list.append(self.alpha * (v_max - v))

            # Non-negative speed
            A_list.append(v_hat.reshape(1, -1))
            l_list.append(-self.alpha * v)
            u_list.append(np.inf)

            # Minimum speed
            if v < self.v_min:
                A_list.append(v_hat.reshape(1, -1))
                l_list.append(-self.alpha * (v - self.v_min))
                u_list.append(np.inf)

        # Acceleration box constraint
        A_box = np.eye(2)
        A_list.append(A_box)

        l_list.extend([-self.a_max, -self.a_max])
        u_list.extend([self.a_max, self.a_max])

        A = sp.csc_matrix(np.vstack(A_list))
        l = np.array(l_list)
        u = np.array(u_list)

        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u, verbose=False)
        res = prob.solve()

        if res.info.status != 'solved':
            return a_agent_des

        return res.x