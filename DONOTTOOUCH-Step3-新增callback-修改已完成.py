import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import os
from datetime import datetime

# 忽略所有警告
#warnings.filterwarnings("ignore")
# ==================== Min-Cut Callback ====================
# ==================== Global counter ====================
incumbent_counter = 0

# ==================== Simple Callback ====================


def min_cut_callback(model, where):
    global incumbent_counter
    if where == GRB.Callback.MIPSOL:
        incumbent_counter += 1
        print(f"\n[Callback] Incumbent #{incumbent_counter:03d} found")

        # ---------- 1. 获取所有变量值 ----------
        y_val = {}
        for (i, j, k, t) in y:
            try:
                y_val[(i, j, k, t)] = model.cbGetSolution(y[i, j, k, t])
            except:
                y_val[(i, j, k, t)] = 0.0

        h_val = {}
        for (i, k, t) in h:
            try:
                h_val[(i, k, t)] = model.cbGetSolution(h[i, k, t])
            except:
                h_val[(i, k, t)] = 0.0

        f_val = {}
        for (i, j) in arcs:
            for t in T:
                try:
                    f_val[(i, j, t)] = model.cbGetSolution(f[i, j, t])
                except:
                    f_val[(i, j, t)] = 0.0

        # ---------- 2. 打开文件 ----------
        os.makedirs("solutions", exist_ok=True)
        os.makedirs("images", exist_ok=True)
        os.makedirs("images/cuts", exist_ok=True)

        sol_file = f"solutions/sol_{incumbent_counter:03d}.txt"

        with open(sol_file, "w", encoding="utf-8") as f_out:
            f_out.write(f"Incumbent #{incumbent_counter:03d}\n")
            f_out.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f_out.write("=" * 60 + "\n\n")

            # 写 y / h / f
            f_out.write("【y 变量 (弧使用情况)】\n")
            for t in T:
                f_out.write(f"\n--- Time t = {t} ---\n")
                for k in K:
                    arcs_k = [(i, j) for (i, j) in arcs if y_val.get((i, j, k, t), 0) > 0.5]
                    if arcs_k:
                        f_out.write(f"  k = {k}: {arcs_k}\n")

            f_out.write("\n【h 变量 (节点使用情况)】\n")
            for t in T:
                f_out.write(f"\n--- Time t = {t} ---\n")
                used = [(i, k) for (i, k, tt) in h_val if tt == t and h_val[(i, k, tt)] > 0.5]
                for i, k in sorted(used):
                    f_out.write(f"  h_{i}_{k}_{t} = 1\n")

            f_out.write("\n【f 变量 (流量)】\n")
            for t in T:
                f_out.write(f"\n--- Time t = {t} ---\n")
                for (i, j) in arcs:
                    val = f_val.get((i, j, t), 0)
                    if val > 1e-4:
                        f_out.write(f"  f_{i}_{j}_{t} = {val:.2f}\n")

            # ============================================================
            # ★★★ Cut Separation + Expanded form + Violation ★★★
            # ============================================================
            f_out.write("\n\n" + "="*70 + "\n")
            f_out.write("【Cut Separation (S contains parking p, S' does not)】\n")
            f_out.write("="*70 + "\n")

            for t in T:
                f_out.write(f"\n{'='*20} Time t = {t} {'='*20}\n")

                for k in K:
                    used_nodes = set()
                    for (i, j) in arcs:
                        if y_val.get((i, j, k, t), 0) > 0.5:
                            used_nodes.add(i)
                            used_nodes.add(j)

                    if len(used_nodes) < 2:
                        f_out.write(f"  k={k}: 节点太少，跳过\n")
                        continue

                    G = nx.DiGraph()
                    for (i, j) in arcs:
                        if y_val.get((i, j, k, t), 0) > 0.5:
                            G.add_edge(i, j)

                    undirected = G.to_undirected()
                    components = list(nx.connected_components(undirected))

                    S = set()
                    S_prime = set()
                    parking_in_route = None

                    for comp in components:
                        if any(p in comp for p in P):
                            S.update(comp)
                            for p in P:
                                if p in comp:
                                    parking_in_route = p
                                    break
                        else:
                            S_prime.update(comp)

                    S = S & used_nodes
                    S_prime = S_prime & used_nodes

                    if not S_prime:
                        f_out.write(f"  k={k}: 没有 S'（所有节点都与停车场连通）→ 无需添加 cut\n")
                        continue

                    f_out.write(f"\n  >>> Route k={k} | Parking used p = {parking_in_route}\n")
                    f_out.write(f"      S  (contains parking) = {sorted(S)}\n")
                    f_out.write(f"      S' (no parking)      = {sorted(S_prime)}\n")

                    print(f"[Cut] t={t} k={k} p={parking_in_route}")
                    print(f"      S  = {sorted(S)}")
                    print(f"      S' = {sorted(S_prime)}")

                    # 对每个 i ∈ S' 添加两条 Lazy 不等式 + 展开式 + violation
                    for i in S_prime:
                        h_i = h_val.get((i, k, t), 0)
                        if h_i < 0.5:
                            continue

                        # ---------- 不等式 (1) ----------
                        terms1 = []
                        expr1_list = []
                        lhs1 = 0.0
                        for j in S:
                            for jp in S_prime:
                                if (j, jp) in arcs:
                                    expr1_list.append(y[j, jp, k, t])
                                    val = y_val.get((j, jp, k, t), 0.0)
                                    lhs1 += val
                                    terms1.append(f"y_{j}_{jp}_{k}_{t}")

                        if expr1_list:
                            model.cbLazy(gp.quicksum(expr1_list) >= h[i, k, t])

                        violation1 = max(0.0, h_i - lhs1)

                        math1_sym = f"      (1)  ∑_{{j∈S, j'∈S'}} y_{{j j',k={k},t={t}}}  ≥  h_{{i={i},k={k},t={t}}}"
                        math1_exp = f"          Expanded : {' + '.join(terms1) if terms1 else '0'}  ≥  {h_i:.0f}"
                        math1_vio = f"          LHS = {lhs1:.2f} , RHS = {h_i:.2f} , Violation = {violation1:.2f}"

                        print(math1_sym)
                        print(math1_exp)
                        print(math1_vio)
                        f_out.write(math1_sym + "\n")
                        f_out.write(math1_exp + "\n")
                        f_out.write(math1_vio + "\n")

                        # ---------- 不等式 (2) ----------
                        terms2 = []
                        expr2_list = []
                        lhs2 = 0.0
                        for j in S:
                            for jp in S_prime:
                                if (jp, j) in arcs:
                                    expr2_list.append(y[jp, j, k, t])
                                    val = y_val.get((jp, j, k, t), 0.0)
                                    lhs2 += val
                                    terms2.append(f"y_{jp}_{j}_{k}_{t}")

                        if expr2_list:
                            model.cbLazy(gp.quicksum(expr2_list) >= h[i, k, t])

                        violation2 = max(0.0, h_i - lhs2)

                        math2_sym = f"      (2)  ∑_{{j∈S, j'∈S'}} y_{{j' j,k={k},t={t}}}  ≥  h_{{i={i},k={k},t={t}}}"
                        math2_exp = f"          Expanded : {' + '.join(terms2) if terms2 else '0'}  ≥  {h_i:.0f}"
                        math2_vio = f"          LHS = {lhs2:.2f} , RHS = {h_i:.2f} , Violation = {violation2:.2f}"

                        print(math2_sym)
                        print(math2_exp)
                        print(math2_vio)
                        f_out.write(math2_sym + "\n")
                        f_out.write(math2_exp + "\n")
                        f_out.write(math2_vio + "\n")

                    # 画高亮图
                    node_colors = []
                    for node in G.nodes():
                        if node in P:
                            node_colors.append('lightgreen')
                        elif node in S:
                            node_colors.append('orange')
                        elif node in S_prime:
                            node_colors.append('tomato')
                        else:
                            node_colors.append('lightgray')

                    cut_edges = [(u, v) for u, v in G.edges()
                                 if (u in S and v in S_prime) or (u in S_prime and v in S)]
                    normal_edges = [e for e in G.edges() if e not in cut_edges]

                    plt.figure(figsize=(11, 8))
                    pos = nx.spring_layout(G, seed=42)
                    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800)
                    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold')
                    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color='gray',
                                           width=1.5, arrows=True, arrowsize=15)
                    if cut_edges:
                        nx.draw_networkx_edges(G, pos, edgelist=cut_edges, edge_color='red',
                                               width=3.0, arrows=True, arrowsize=18)

                    plt.title(f"Inc #{incumbent_counter:03d} | t={t} k={k} | p={parking_in_route}\n"
                              f"S (orange)={sorted(S)}\nS' (red)={sorted(S_prime)}", fontsize=11)
                    plt.axis('off')
                    plt.tight_layout()

                    cut_img = f"images/cuts/cut_inc{incumbent_counter:03d}_t{t}_k{k}.png"
                    plt.savefig(cut_img, dpi=160, bbox_inches='tight')
                    plt.close()
                    f_out.write(f"      高亮图: {cut_img}\n")
                    print(f"         → 高亮 Cut 图: {cut_img}")

        print(f"         → 变量值 + Cut Separation 结果已保存: {sol_file}")

        # ---------- 3. 普通 Incumbent 图 ----------
        for t in T:
            route_arcs = {k: [] for k in K}
            for k in K:
                for (i, j) in arcs:
                    if y_val.get((i, j, k, t), 0) > 0.5:
                        route_arcs[k].append((i, j))

            G = nx.DiGraph()
            G.add_nodes_from(J)
            edges_by_k = {k: [] for k in K}
            for k in K:
                for (i, j) in route_arcs[k]:
                    G.add_edge(i, j)
                    edges_by_k[k].append((i, j))

            if not G.edges():
                continue

            node_colors = ['lightgreen' if v in P else 'lightblue' for v in J]
            plt.figure(figsize=(12, 9))
            pos = nx.spring_layout(G, seed=42, k=2.2)

            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000)
            nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')

            color_map = {0: '#e74c3c', 1: '#3498db', 2: '#2ecc71', 3: '#9b59b6'}
            style_map = {0: 'solid', 1: 'dashed', 2: 'dotted', 3: 'dashdot'}

            for k in K:
                if edges_by_k[k]:
                    nx.draw_networkx_edges(G, pos, edgelist=edges_by_k[k],
                                           edge_color=color_map.get(k, 'gray'),
                                           style=style_map.get(k, 'solid'),
                                           width=2.8, arrows=True, arrowsize=22,
                                           connectionstyle='arc3,rad=0.1')

            edge_labels = {}
            for k in K:
                for (i, j) in edges_by_k[k]:
                    edge_labels[(i, j)] = f"k={k}\nf={round(f_val.get((i,j,t),0),2)}"
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

            plt.title(f"Incumbent #{incumbent_counter:03d}  |  t = {t}", fontsize=15, pad=10)
            plt.axis('off')
            img_file = f"images/incumbent_{incumbent_counter:03d}_t{t}.png"
            plt.savefig(img_file, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"         → 图片已保存: {img_file}")
# ==================== 数据 ====================
J = [0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14]
P = [0, 1,2]
K = [0, 1,2]
T = [1,2]
arcs = [(i, j) for i in J for j in J if i != j]

# 方法2：用 numpy（更快）
D = np.loadtxt('n15.csv', delimiter=',')
print(D)

# ==================== 新增数据加载 ====================
demand = np.loadtxt('demand.csv', delimiter=',')
# print(demand)

cap_df = pd.read_csv('capacity10.csv', index_col=0)
C = cap_df['C'].to_dict()

model = gp.Model("Final_Visualization")
model.setParam('OutputFlag', 0)

# ==================== 变量（使用清晰索引命名） ====================
y = {}   # y_i_j_k_t
for (i, j) in arcs:
    for k in K:
        for t in T:
            y[i, j, k, t] = model.addVar(
                vtype=GRB.BINARY,
                name=f"y_{i}_{j}_{k}_{t}"
            )

h = {}   # h_i_k_t
for i in J:
    for k in K:
        for t in T:
            h[i, k, t] = model.addVar(
                vtype=GRB.BINARY,
                name=f"h_{i}_{k}_{t}"
            )

# ==================== 新增连续流量变量 f ====================
f = {}   # f_i_j_k_t
for (i, j) in arcs:
        for t in T:
            f[i, j, t] = model.addVar(
                vtype=GRB.CONTINUOUS,
                name=f"f_{i}_{j}_{t}"
            )

# ==================== 约束 ====================

# (1) 流平衡约束
for i in J:
    for t in T:
        for k in K:
            model.addConstr(
                gp.quicksum(y[j, i, k, t] for j in J if j != i) ==
                gp.quicksum(y[i, j, k, t] for j in J if j != i),
                name=f"FlowBalance_{i}_{t}_{k}"
            )

# (2) 从普通节点集合出发
for i in J:
    for t in T:
        if i not in P:
            model.addConstr(
                gp.quicksum(y[i, j, k, t] for k in K for j in J if j != i) >= 1,
                name=f"StartFromnodeSet_{i}_{t}"
            )

# (3) 返回普通节点集合
for i in J:
    for t in T:
        if i not in P:
            model.addConstr(
                gp.quicksum(y[j, i, k, t] for k in K for j in J if j != i) >= 1,
                name=f"ReturnTonodeSet_{i}_{t}"
            )

# (4) 节点使用限制
for t in T:
    for p in P:
        model.addConstr(
            gp.quicksum(h[p, k, t] for k in K) <= 2,
            name=f"NodeUsageLimit_{p}_{t}"
        )

# (5) y 与 h 的关联约束 *************************
for i in J:
    for j in J:
        if j != i:
            for t in T:
                for k in K:
                    model.addConstr(
                        y[i, j, k, t] <= h[i, k, t],
                        name=f"LinkYtoH_{i}_{j}_{t}_{k}"
                    )

# (6) 禁止同一路线同一时间使用双向弧 ******************
for i in J:
    for j in J:
        if i != j:
            for t in T:
                for k in K:
                    model.addConstr(
                        y[i, j, k, t] + y[j, i, k, t] <= 1,
                        name=f"NoBidirectional_{i}_{j}_{t}_{k}"
                    )
# # (7) 新增
for t in T:
    for k in K:
        model.addConstr(
            gp.quicksum(y[i, j, k, t] for i in J for j in J if i != j) <=6,
            name=f"CNN_{k}_{t}"
        )

# (8) 从停车场集合出发
for k in K:
    for t in T:
        model.addConstr(
            gp.quicksum(y[p, j, k, t] for p in P for j in J if j != p) == 1,
            name=f"StartFromParkingSet{p}_{k}_{t}"
        )

# (9) 返回停车场集合
for k in K:
    for t in T:
        model.addConstr(
            gp.quicksum(y[j, p, k, t] for p in P for j in J if j != p) == 1,
            name=f"ReturnToParkingSet{p}_{k}_{t}"
        )

# (9) 每个停车场必须用
for p in P:
    for t in T:
        model.addConstr(
            gp.quicksum(y[j, p, k, t] for k in K for j in J if j != p) == 1,
            name=f"ParkingMustUseSet{p}_{k}_{t}"
        )


for p in P:
    for j in J:
        if j != p:
            for t in T:
                model.addConstr(
                    gp.quicksum(y[p, j, k, t]* demand[t-1,p] for k in K) <= f[p,j, t],
                    name=f"Flow_P_J_{p}_{j}_{t}"
                )
for p in P:
    for j in J:
        if j != p:
            for t in T:
                model.addConstr(
                    gp.quicksum(y[j, p, k, t]* demand[t-1,p] for k in K) <= f[j,p, t],
                    name=f"Flow_J_P_{p}_{j}_{t}"
                )
# # 流量守恒约束-关于
for t in T:
    for j in J:
        if j not in P:
            model.addConstr(
                gp.quicksum(f[j, i,  t] for i in J if i != j) ==
                gp.quicksum(f[i, j,  t] for i in J if i != j),
                name=f"FlowConservation_f_{j}_{t}"
            )
#
# for k in K:
#     for t in T:
#         model.addConstr(
#         gp.quicksum(y[j, p, k, t] for i in {1,7,14} for j in {3,9,13})>=h[3,1,1])
# ##1
# f[4,8,1].lb=35
# f[8,1,1].lb=35
# ##2
# f[6,3,1].lb=25
# f[3,2,1].lb=25
# ##3
# f[5,9,1].lb=25
# f[7,5,1].lb=25
# f[9,0,1].lb=25
# # (10) 路线不要太长
# V = set(J) | set(P)
# for k in K:
#     for t in T:
#         model.addConstr(
#             gp.quicksum(y[i, j, k, t] for i in V for j in V if j != i) <=5,
#             name=f"RouteLongSet_{k}_{t}"
#         )
# y[4,8,0,1].lb=1

# ==================== 新增连续流量约束 ====================
#==================== Big-M 约束（使用每个时刻不同的 M_t） ====================
M=80

# Big-M 约束：流量只能在被选中的弧上流动
for (i, j) in arcs:
    for t in T:
        model.addConstr(
            f[i, j, t] <= M * gp.quicksum(y[i, j, k, t] for k in K),
            name=f"BigM_f_y_{i}_{j}_{t}"
        )
#


# # 节点容量约束（使用流量 f）
# for j in J:
#     if j not in P:
#         for t in T:
#             model.addConstr(
#                 gp.quicksum(f[i, j, t] for i in J if i != j) <= C[j],
#                 name=f"NodeCapacityFlow_{j}_{t}"
#             )

# 需求满足约束（总流量从停车场出发）
# for t in T:
#     for p in P:
#         model.addConstr(
#             gp.quicksum(f[p, j, k, t] for p in P for j in J for k in K if j != p) >= De[t],
#             name=f"DemandSatisfaction_{t}"
#         )

# 目标
model.setObjective(gp.quicksum(D[i,j]*y[i, j, k, t] for i in J for j in J for k in K for t in T if j!=i), GRB.MINIMIZE)
model._y = y
model._h = h   # 把 y 传给 callback
model.Params.LazyConstraints = 1       # 开启 Lazy Constraints
model.optimize(min_cut_callback)       # 使用 callback
model.write("model0.lp")

# ==================== 输出与可视化 ====================
print("=" * 100)
print("模型求解结果（按时间 t 分开展示）")
print("=" * 100)

if model.Status == GRB.OPTIMAL:
    print("状态: 可行\n")

    for t in T:
        print(f"\n{'='*30} 时间 t = {t} {'='*30}")

        # 打印 h
        print("【h 变量】")
        for i in J:
            for k in K:
                if h[i, k, t].X > 0.5:
                    print(f"  h_{i}_{k}_{t} = 1")

        # 打印 y
        print("\n【y 变量 (弧)】")
        route_arcs = {k: [] for k in K}
        for k in K:
            for (i, j) in arcs:
                if y[i, j, k, t].X > 0.5:
                    route_arcs[k].append((i, j))

        active_k = [k for k in K if route_arcs[k]]
        for k in active_k:
            print(f"  k = {k}: {route_arcs[k]}")

        # ==================== 新增：表格形式输出线路与流量 ====================
        print(f"\n【时刻 t = {t} 各路线详细线路与流量表格】")
        for k in K:
            route_data = []
            for (i, j) in arcs:
                if y[i, j, k, t].X > 0.5:
                    route_data.append({
                        '起点 i': i,
                        '终点 j': j,
                        '流量 f': round(f[i, j, t].X, 2)
                    })

            if route_data:
                df = pd.DataFrame(route_data)
                print(f"\n路线 k = {k}：")
                print(df.to_string(index=False))
            else:
                print(f"\n路线 k = {k}：未使用")

        # ==================== 绘图（每个 k 不同颜色 + 不同线型 + 流量标注） ====================
        G = nx.DiGraph()
        G.add_nodes_from(J)

        # 按 k 分组边
        edges_by_k = {k: [] for k in K}
        for k in K:
            for (i, j) in route_arcs[k]:
                G.add_edge(i, j)
                edges_by_k[k].append((i, j))

        if G.edges():
            node_colors = ['lightgreen' if v in P else 'lightblue' for v in J]

            plt.figure(figsize=(12, 9))
            pos = nx.spring_layout(G, seed=42, k=2.2)

            # 先画节点
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000)
            nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')

            # 定义不同 k 的颜色和线型
            color_map = {0: '#e74c3c', 1: '#3498db', 2: '#2ecc71', 3: '#9b59b6'}
            style_map = {0: 'solid', 1: 'dashed', 2: 'dotted', 3: 'dashdot'}

            # 按 k 分别绘制边
            for k in K:
                if edges_by_k[k]:
                    nx.draw_networkx_edges(
                        G, pos,
                        edgelist=edges_by_k[k],
                        edge_color=color_map.get(k, 'gray'),
                        style=style_map.get(k, 'solid'),
                        width=2.8,
                        arrows=True,
                        arrowsize=22,
                        connectionstyle='arc3,rad=0.1'
                    )

            # ==================== 关键修改：边上同时显示 k 和 f[i,j,t] ====================
            edge_labels = {}
            for k in K:
                for (i, j) in edges_by_k[k]:
                    flow_val = round(f[i, j, t].X, 2)
                    edge_labels[(i, j)] = f"k={k}\nf={flow_val}"

            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

            # 图上显示变量信息（保留）
            h_text = "\n".join([f"h_{i}_{k}_{t}=1" for i in J for k in K if h[i, k, t].X > 0.5])
            y_text = "\n".join([f"y_{i}_{j}_{k}_{t}=1" for k in active_k for (i, j) in route_arcs[k]])

            info_text = f"Time t = {t}\n\n【h】\n{h_text}\n\n【y】\n{y_text}"
            props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9)
            # plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
            #          fontsize=9, verticalalignment='top', bbox=props, family='monospace')

            plt.title(f"Routes at Time t = {t}\n(Edge label: k + f[i,j,t])", fontsize=15, pad=10)
            plt.axis('off')

            filename = f"route_t{t}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  → 已保存图片: {filename}")

else:
    print(">>> 模型不可行 (Infeasible)")
    print(">>> None")
print("\n" + "=" * 100)