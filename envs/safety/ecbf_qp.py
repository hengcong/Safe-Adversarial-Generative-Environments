import numpy as np
import osqp
import scipy.sparse as sp


class ECBF_Collision_QP:
    """
    Relative-distance ECBF for agent collision avoidance.

    Barrier:
        h = r^T r - D^2

    ECBF condition:
        h_ddot + k1 h_dot + k0 h >= 0

    Final inequality:

        2 r^T a_agent <=
        2 r^T a_other
        + 2 v^T v
        + 2 k1 r^T v
        + k0 (r^T r - D^2)
    """

    def __init__(self,
                 D_safe=5.0,
                 k1=2.0,
                 k0=1.0,
                 a_max=6.0,
                 slack_weight=1000.0,
                 headway_D0=2.0,
                 headway_tau=2.0):

        self.D = D_safe
        self.k1 = k1
        self.k0 = k0
        self.a_max = a_max
        self.rho = slack_weight
        self.headway_D0 = headway_D0
        self.headway_tau = headway_tau

    def solve(self,
              agent_pos,
              agent_vel,
              a_agent_des,
              others_pos,
              others_vel,
              others_acc=None,
              use_headway=True,
              max_neighbors=8,
              debug=False,
              boundaries=None):
        """
        Extended ECBF solve that supports additional linear boundary constraints.

        boundaries: optional list of boundary specs. Each spec is a tuple (b_point, n_vec, b_offset)
            - b_point: (2,) a reference point on the boundary (not strictly required if b_offset given)
            - n_vec: (2,) inward unit normal pointing from boundary into drivable area
            - b_offset: scalar such that h = n^T p - b_offset >= 0 means "inside"
          Alternatively you can pass a list of tuples (n_vec, b_offset).
        """

        if others_acc is None:
            others_acc = [np.zeros(2, dtype=float) for _ in others_pos]

        # prune neighbors
        if len(others_pos) > max_neighbors:
            dists = [np.linalg.norm(np.asarray(p, dtype=float) - np.asarray(agent_pos, dtype=float)) for p in
                     others_pos]
            idx_sorted = np.argsort(dists)[:max_neighbors]
            others_pos = [others_pos[i] for i in idx_sorted]
            others_vel = [others_vel[i] for i in idx_sorted]
            others_acc = [others_acc[i] for i in idx_sorted]

        N = len(others_pos)
        # normalize boundaries input into list of (n_vec, b_offset)
        boundary_list = []
        if boundaries is not None:
            for spec in boundaries:
                # allow (b_point, n_vec), (n_vec, b_offset), or (b_point, n_vec, b_offset)
                if len(spec) == 3:
                    b_point = np.asarray(spec[0], dtype=float)
                    n_vec = np.asarray(spec[1], dtype=float)
                    b_offset = float(spec[2])
                elif len(spec) == 2:
                    # try to infer: if first is normal assume (n_vec, b_offset)
                    maybe0 = np.asarray(spec[0], dtype=float)
                    maybe1 = np.asarray(spec[1], dtype=float)
                    # if maybe0 is unit-like treat as normal
                    if np.linalg.norm(maybe0) > 0 and np.allclose(np.linalg.norm(maybe0), 1.0, atol=1e-1):
                        n_vec = maybe0
                        b_offset = float(maybe1)  # treat as offset
                    else:
                        b_point = maybe0
                        n_vec = maybe1
                        # compute b_offset = n^T b_point
                        b_offset = float(np.dot(n_vec, b_point))
                else:
                    continue
                # ensure n_vec is unit
                n_norm = np.linalg.norm(n_vec)
                if n_norm > 0:
                    n_vec = n_vec / n_norm
                boundary_list.append((n_vec, float(b_offset)))

        # dynamic safe distance
        speed = float(np.linalg.norm(agent_vel))
        if use_headway:
            D_safe = float(self.headway_D0 + self.headway_tau * speed)
        else:
            D_safe = float(self.D)

        # number of boundary slacks
        B = len(boundary_list)
        dim = 2 + N + B  # [a_x, a_y, s_0..s_{N-1}, sb_0..sb_{B-1}]

        # objective
        P = np.zeros((dim, dim), dtype=float)
        P[0:2, 0:2] = np.eye(2)
        if N + B > 0:
            P[2:, 2:] = self.rho * np.eye(N + B)
        P = sp.csc_matrix(P)

        q = np.zeros(dim, dtype=float)
        q[0:2] = -np.asarray(a_agent_des, dtype=float)

        A_rows = []
        l_list = []
        u_list = []

        # vehicle-vehicle ECBF constraints (with slack s_i)
        for i, (p_o, v_o, a_o) in enumerate(zip(others_pos, others_vel, others_acc)):
            r = np.asarray(p_o, dtype=float) - np.asarray(agent_pos, dtype=float)
            v = np.asarray(v_o, dtype=float) - np.asarray(agent_vel, dtype=float)

            r_norm2 = float(r @ r)
            v_norm2 = float(v @ v)
            rv = float(r @ v)

            rhs = (2.0 * float(r @ a_o)
                   + 2.0 * v_norm2
                   + 2.0 * self.k1 * rv
                   + self.k0 * (r_norm2 - (D_safe * D_safe)))

            row = np.zeros(dim, dtype=float)
            row[0:2] = 2.0 * r
            row[2 + i] = -1.0  # slack for this neighbor

            A_rows.append(row)
            l_list.append(-np.inf)
            u_list.append(rhs)

        # boundary ECBF constraints (linear boundaries, each with slack)
        for j, (n_vec, b_offset) in enumerate(boundary_list):
            # compute h = n^T p - b_offset
            h_val = float(np.dot(n_vec, agent_pos) - b_offset)
            h_dot = float(np.dot(n_vec, agent_vel))
            # ECBF inequality: n^T a >= -k1 * n^T v - k0 * h
            # bring to form: n^T a - s_b <= rhs_b
            rhs_b = float(self.k0 * h_val + self.k1 * h_dot)  # note: rearranged to match u_list as upper bound
            row = np.zeros(dim, dtype=float)
            row[0:2] = n_vec  # n^T a_agent
            row[2 + N + j] = -1.0  # slack for this boundary

            A_rows.append(row)
            l_list.append(-np.inf)
            u_list.append(rhs_b)

        # slack non-negativity rows for vehicle slacks
        for i in range(N):
            row = np.zeros(dim, dtype=float)
            row[2 + i] = 1.0
            A_rows.append(row)
            l_list.append(0.0)
            u_list.append(np.inf)

        # slack non-negativity rows for boundary slacks
        for j in range(B):
            row = np.zeros(dim, dtype=float)
            row[2 + N + j] = 1.0
            A_rows.append(row)
            l_list.append(0.0)
            u_list.append(np.inf)

        # acceleration box bounds
        for jj in range(2):
            row = np.zeros(dim, dtype=float)
            row[jj] = 1.0
            A_rows.append(row)
            l_list.append(-self.a_max)
            u_list.append(self.a_max)

        A = sp.csc_matrix(np.vstack(A_rows))
        l = np.array(l_list, dtype=float)
        u = np.array(u_list, dtype=float)

        prob = osqp.OSQP()
        prob.setup(P, q, A, l, u, verbose=False, polish=False, eps_abs=1e-4, eps_rel=1e-4, max_iter=20000)

        res = prob.solve()

        status = res.info.status if hasattr(res, 'info') else None
        if not (status == 'solved' or getattr(res.info, 'status_val', None) == 1):
            if debug:
                print("[ECBF] OSQP failed:", status)
            return a_agent_des

        x = np.asarray(res.x, dtype=float)
        a_corr = x[0:2]

        if debug:
            slacks = x[2:] if len(x) > 2 else np.array([])
            print("[ECBF debug] D_safe =", D_safe, "N_constraints =", N, "B_constraints =", B)
            print("[ECBF debug] slacks (first 12) =", slacks[:12])
            n_relaxed = int(np.sum(slacks > 1e-3)) if slacks.size > 0 else 0
            print("[ECBF debug] #relaxed_constraints =", n_relaxed)

        return a_corr