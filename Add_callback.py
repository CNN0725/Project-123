import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")
# ==================== Min-Cut Callback ====================
def min_cut_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        print("\n" + "="*70)
        print("[Callback] 检测到新整数解，开始检查子回路...")

        # 正确获取 y 的解（避免 KeyError）
        y_val = {}
        for (i, j, k, t) in y:
            try:
                y_val[y[i, j, k, t]] = model.cbGetSolution(y[i, j, k, t])
            except:
                y_val[y[i, j, k, t]] = 0.0

        for k in K:
            for t in T:
                # 构建支持图
                G = nx.DiGraph()
                for (i, j) in arcs:
                    if y_val.get(y[i, j, k, t], 0) > 0.5:
                        G.add_edge(i, j, capacity=1)
                if len(G.edges()) < 2:
                    continue
                # 尝试从停车场出发做 Min-Cut
                for p in P:
                    for j in J:
                        if j not in P:
                            try:
                                cut_value, (S, S_complement) = nx.minimum_cut(
                                    G, p, j, capacity='capacity'
                                )
                                if cut_value < 1:
                                    print(f"[发现子回路] t={t}, k={k}, p={p}, j={j}")
                                    print(f"             S = {list(S)}")
                                    print(f"             S' = {list(S_complement)}")
                                    # 添加 Lazy 约束
                                    model.cbLazy(
                                        gp.quicksum(y[j, i, k, t] for j in S for i in S_complement
                                                    if (j, i) in arcs) >= 1
                                    )
                                    model.cbLazy(
                                        gp.quicksum(y[i, j, k, t] for i in S_complement for j in S
                                                    if (i, j) in arcs) >= 1
                                    )
                            except:
                                pass
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
#                 gp.quicksum(f[i, j, k, t] for i in J for k in K if i != j) <= C[j],
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
model._y = y                           # 把 y 传给 callback
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