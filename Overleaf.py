import gurobipy as gp
import numpy as np
from scipy.cluster.vq import kmeans

# ====================== Complete Model (Final Version) ======================
# Strictly following the latest "Complete model" from the paper
# New features:
#   • Continuous variable v_{i,t} = served visitors at node i in period t
#   • Average \bar{v}_i
#   • Objective uses v in passenger cost and temporal balancing penalty
#   • Constraints (C12) and (C13) added

print("=== Complete Model: CNN Design of Micro-Circulation Bus Routes in Scenic Areas (Final Version) ===")
print("Multi-period mixed-integer quadratic program (MIQP)")
print("Introduces served visitor variable v_{i,t} and average \bar{v}_i")
print("All constraints (C1)-(C13) and objective follow the latest paper exactly.")

# ====================== 2. Data Generation ======================
np.random.seed(42)
n = 10          # |J| candidate nodes
q_anchor = 5   # |D| anchor nodes
p = 3           # |T| time periods
r = 10          # |K| potential routes

# Generate node coordinates
coords_I = np.random.rand(50, 2) * 10000
centroids, _ = kmeans(coords_I, q_anchor)
J_coords = np.vstack([centroids, np.random.rand(n - q_anchor, 2) * 10000])

J = list(range(n))
D = list(range(q_anchor))
T = list(range(1, p + 1))
K = list(range(r))
T_off = [2]     # off-peak periods

print(f"J = {len(J)} candidate nodes")
print(f"D = {D} anchor nodes")
print(f"T = {T} time periods (off-peak T^off = {T_off})")
print(f"K = {len(K)} potential routes")

# ====================== 3. Parameters ======================
c_open = {i: np.random.randint(500, 1000) for i in J}                    # c_i^open
c_use = {(i, t): np.random.randint(20, 80) for i in J for t in T}        # c_{i,t}^use
c_route = {(k, t): np.random.randint(200, 500) for k in K for t in T}    # c_{k,t}^route
tau = {(j, l, t): np.random.randint(2, 12) for j in J for l in J for t in T if j != l}

q_demand = {(i, t): np.random.randint(50, 300) for i in J for t in T}    # q_{i,t}
c_pass = {(i, t): np.random.randint(3, 15) for i in J for t in T}        # c_{i,t}^pass

alpha = -0.10
beta = 1.0
lambda_reward = 80.0
gamma = 0.08

N_max = 6
A_min = 6
R_min = 2

# ====================== 4. Model Creation ======================
model = gp.Model("Complete_Micro_Circulation_Bus_Model")
model.Params.TimeLimit = 300
model.Params.MIPGap = 1e-6
model.Params.MIPFocus = 1
model.Params.Heuristics = 0.5
model.Params.NonConvex = 2

# ====================== 5. Decision Variables ======================
x = model.addVars(J, T, vtype=gp.GRB.BINARY, name="x")          # x_{i,t}
z = model.addVars(J, T, vtype=gp.GRB.BINARY, name="z")          # z_{i,t}
y = model.addVars(K, T, vtype=gp.GRB.BINARY, name="y")          # y_{k,t}
u = model.addVars(J, K, T, vtype=gp.GRB.BINARY, name="u")       # u_{i,k,t}
a = model.addVars(J, J, K, T, vtype=gp.GRB.BINARY, name="a")    # a_{j,l,k,t} (j≠l)

v = model.addVars(J, T, lb=0, vtype=gp.GRB.CONTINUOUS, name="v")          # v_{i,t}  (new)
bar_v = model.addVars(J, lb=0, vtype=gp.GRB.CONTINUOUS, name="bar_v")     # \bar{v}_i (new)

# ====================== 6. Objective Function ======================
passenger_term = alpha * gp.quicksum(c_pass[i, t] * v[i, t] for i in J for t in T)

operator_term = beta * (
    gp.quicksum(c_route[k, t] * y[k, t] for k in K for t in T) +
    gp.quicksum(c_open[i] * x[i, max(T)] for i in J) +
    gp.quicksum(c_use[i, t] * z[i, t] for i in J for t in T) +
    gp.quicksum(tau[j, l, t] * a[j, l, k, t] for j in J for l in J for k in K for t in T if j != l)
)

offpeak_reward = -lambda_reward * gp.quicksum(z[i, t] for i in J for t in T_off if i not in D)

variance_term = gamma * gp.quicksum((v[i, t] - bar_v[i]) ** 2 for i in J for t in T)

model.setObjective(passenger_term + operator_term + offpeak_reward + variance_term, gp.GRB.MINIMIZE)
print("✅ Objective function implemented exactly as in the latest paper (uses v_{i,t})")

# ====================== 7. Constraints (C1-C13) ======================
# (C1) Construction persistence
for i in J:
    for tt in range(1, len(T)):
        t = T[tt]
        prev = T[tt-1]
        model.addConstr(x[i, t] >= x[i, prev])

# (C2) Only constructed nodes can be activated
for i in J:
    for t in T:
        model.addConstr(z[i, t] <= x[i, t])

# (C3) Anchor node requirement
for k in K:
    for t in T:
        model.addConstr(gp.quicksum(u[i, k, t] for i in D) == y[k, t])

# (C4) Routes visit only active nodes
for i in J:
    for k in K:
        for t in T:
            model.addConstr(u[i, k, t] <= z[i, t])

# (C5) Active nodes must be served
for i in J:
    for t in T:
        model.addConstr(gp.quicksum(u[i, k, t] for k in K) >= z[i, t])

# (C6) Arc consistency
for j in J:
    for l in J:
        for k in K:
            for t in T:
                if j != l:
                    model.addConstr(a[j, l, k, t] <= y[k, t])
                    model.addConstr(a[j, l, k, t] <= u[j, k, t])
                    model.addConstr(a[j, l, k, t] <= u[l, k, t])
                else:
                    model.addConstr(a[j, j, k, t] == 0)

# (C7) Flow conservation
for i in J:
    for k in K:
        for t in T:
            out_sum = gp.quicksum(a[i, l, k, t] for l in J if l != i)
            in_sum = gp.quicksum(a[l, i, k, t] for l in J if l != i)
            model.addConstr(out_sum == u[i, k, t])
            model.addConstr(in_sum == u[i, k, t])

# (C8) Max nodes per route
for k in K:
    for t in T:
        model.addConstr(gp.quicksum(u[i, k, t] for i in J) <= N_max)

# (C9) Minimum active nodes per period
for t in T:
    model.addConstr(gp.quicksum(z[i, t] for i in J) >= A_min)

# (C10) Minimum routes per period
for t in T:
    model.addConstr(gp.quicksum(y[k, t] for k in K) >= R_min)

# (C11) Unique service for non-anchor nodes
non_anchors = [i for i in J if i not in D]
for i in non_anchors:
    for t in T:
        model.addConstr(gp.quicksum(u[i, k, t] for k in K) <= 1)

# (C12) Served visitors bounds
for i in J:
    for t in T:
        model.addConstr(v[i, t] <= q_demand[i, t] * z[i, t])
        model.addConstr(v[i, t] >= 0)

# (C13) Average served visitors
for i in J:
    model.addConstr(bar_v[i] == (1.0 / len(T)) * gp.quicksum(v[i, t] for t in T))

print("✅ All constraints (C1)-(C13) implemented exactly as in the latest paper")
print("✅ Model ready for optimization.")
# ====================== FIX: Force full service when node is activated ======================
# 强制激活节点时必须服务全部基准需求（避免退化解）
for i in J:
    for t in T:
        model.addConstr(v[i, t] == q_demand[i, t] * z[i, t],
                        name=f"FullService_{i}_{t}")
# ====================== 8. Solve the model ======================
model.optimize()

if model.status == gp.GRB.OPTIMAL or model.status == 2:
    print("\n" + "=" * 100)
    print("✅ OPTIMAL SOLUTION FOUND (MIQP)")
    print(f"Total objective value Z = {model.objVal:.2f}")
    print("=" * 100)
    # ====================== 输出 LP 和 MPS 文件（新增） ======================
    model.write("Complete_Micro_Circulation_Bus_Model.lp")  # 人可读的 LP 格式
    model.write("Complete_Micro_Circulation_Bus_Model.mps")  # 标准 MPS 格式（含求解结果）
    print("✅ 已成功导出模型文件：")
    print("   • Complete_Micro_Circulation_Bus_Model.lp")
    print("   • Complete_Micro_Circulation_Bus_Model.mps")
    # ====================== Objective Function Breakdown ======================
    print("\n=== Objective Function Breakdown ===")
    passenger_cost = passenger_term.getValue()
    operator_cost = operator_term.getValue()
    offpeak_reward_val = offpeak_reward.getValue()
    variance_penalty = variance_term.getValue()

    print(f"  Passenger service cost          : {passenger_cost:12.2f}  (α term)")
    print(f"  Operator cost                   : {operator_cost:12.2f}  (β term)")
    print(f"  Off-peak reward (negative)      : {offpeak_reward_val:12.2f}  (-λ term)")
    print(f"  Temporal balancing penalty      : {variance_penalty:12.2f}  (γ term)")
    print(f"  {'─' * 50}")
    print(f"  Total Z                         : {model.objVal:12.2f}")
    print("=" * 100)

    # ====================== Average served visitors \bar{v}_i ======================
    print("\n=== Average served visitors across periods \bar{v}_i ===")
    for i in sorted(J):
        print(f"  Node {i:2d}: \bar{{v}}_i = {bar_v[i].X:.2f}")

    # ====================== Passenger statistics per period (using v) ======================
    print("\n=== Passenger statistics per period per node ===")
    print("Explanation:")
    print("  • Total potential demand = q_{i,t}")
    print("  • Served visitors        = v_{i,t}")
    print("  • Remaining demand       = q_{i,t} - v_{i,t}")
    print("-" * 90)

    for t in T:
        print(f"\n[Period {t}]")

        active_nodes = [i for i in J if z[i, t].X > 0.5]
        print(f"  Active nodes: {sorted(active_nodes)}")

        opened_routes = [k for k in K if y[k, t].X > 0.5]
        print(f"  Routes opened: {len(opened_routes)} ({opened_routes})")

        print(f"  Passenger statistics (total demand / served / remaining):")
        for i in sorted(J):
            total_d = q_demand[i, t]
            served_v = v[i, t].X
            remaining = total_d - served_v
            print(f"    Node {i:2d}: Total {total_d:4.0f} | "
                  f"Served {served_v:6.1f} | "
                  f"Remaining {remaining:6.1f}  "
                  f"(Active: {int(z[i, t].X)})")

        total_d_t = sum(q_demand[i, t] for i in J)
        total_served_t = sum(v[i, t].X for i in J)
        print(f"  Total demand in period: {total_d_t:.0f}")
        print(f"  Total served in period: {total_served_t:.1f}")

    # ====================== Variance statistics ======================
    print("\n" + "=" * 60)
    print("=== Cross-period variance statistics per node ===")
    total_var = 0
    for i in sorted(J):
        v_values = [v[i, t].X for t in T]
        mean_v = bar_v[i].X
        var_i = sum((val - mean_v)**2 for val in v_values)
        total_var += var_i
        print(f"  Node {i:2d}: Variance = {var_i:8.2f}   (Average \bar{{v}} = {mean_v:6.2f})")
    print(f"\nTotal model variance = {total_var:.2f}")

    # ====================== Detailed Results per Period (routes) ======================
    print("\n=== Detailed Results per Period ===")
    for t in T:
        print(f"\n[Period {t}]")

        active = [i for i in J if z[i, t].X > 0.5]
        built = [i for i in J if x[i, t].X > 0.5]
        print(f"  Constructed nodes: {sorted(built)}")

        if t == 1:
            newly_built = [i for i in active]
        else:
            newly_built = [i for i in built if x[i, t-1].X < 0.5]
        print(f"  Newly constructed nodes: {sorted(newly_built)}")

        print(f"  Active nodes: {sorted(active)}")

        opened_routes = [k for k in K if y[k, t].X > 0.5]
        print(f"  Routes opened: {opened_routes} routes")

        print("  Route details:")
        for k in opened_routes:
            start = None
            for i in D:
                if u[i, k, t].X > 0.5:
                    start = i
                    break
            if start is None:
                print(f"    Route {k:2d}: [No valid anchor]")
                continue

            path = [start]
            current = start
            visited = {start}
            for _ in range(30):
                next_node = None
                for l in J:
                    if l not in visited and a[current, l, k, t].X > 0.5:
                        next_node = l
                        break
                if next_node is None:
                    if a[current, start, k, t].X > 0.5:
                        path.append(start)
                    break
                path.append(next_node)
                visited.add(next_node)
                current = next_node
            print(f"    Route {k:2d}: {path}  ← Cycle")

else:
    print("⚠️  No optimal solution found. Status code:", model.status)
