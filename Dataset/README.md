# STEP 1 随机生成chiplet 系统的.cfg文件, 并且将其转换成ilp布局求解器可以接受的json输入文件
python input_preprocess.py --generate-random-cfg --generate-count 8000 --generate-out-dir config

.cfg文件在flow_GCN/Dataset/config 下，输入的json文件在flow_GCN/Dataset/dataset/input_test 下

随机生成chiplet 系统的规则如下：
    芯片数：3~20 随机
    尺寸：3~30，长宽比 0.8~1.25
    功耗：1~200 随机
    连接：必连通，带宽只能是 128/256/512/1024
    格式：完全对齐官方 cpu-dram.cfg 标准

    连接关系生成规则
    保证整张图一定是连通图
    不会出现某几个芯片孤立、和其他芯片完全不连通的情况。
    所有 chiplet 最终都在同一个连通分量里。
    第一步：先生成一棵随机生成树（spanning tree）
    随机打乱所有芯片编号。
    从一个节点开始，每次随机连一个未连接的节点。
    这样最少、最精简地把所有芯片连起来，保证连通。
    每条边的带宽从 [128, 256, 512, 1024] 随机选一个。
    第二步：随机增加额外边
    遍历所有还没连边的芯片对 (i,j)，i<j。
    每一对有 25%（0.25）概率 多加一条连接。
    每条新加边的带宽同样从 [128, 256, 512, 1024] 随机选。
    连接矩阵是对称的
    chiplet i 到 j 的带宽 = chiplet j 到 i 的带宽。
    最终生成对称方阵 connections_matrix。
    带宽只能是固定四档
    不随机乱造数值，只从：
    128 / 256 / 512 / 1024
    四个值里随机抽取。


#STEP 2 将chiplet 系统的输入文件放入ILP求解器中求解合法布局，ILP gap取值从0-1随机，time limit 300s
python gen_legal_pla.py --start 0 --end 6000 --no-console > /root/workspace/flow_GCN/gcn_thermal/dataset/dataset.log 2>&1

输出的数据集在flow_GCN/Dataset/dataset/output下
    fig是输出的合法布局示意图片
    log是ILP求解器的输出
    placement是输出的合法布局的chiplet的左下角坐标

# STEP 3 处理数据集
python process_dataset.py

1. 将flow_GCN/gcn_thermal/dataset/input_test下的json文件与flow_GCN/gcn_thermal/dataset/output/placement下的json文件对应起来
2. flow_GCN/gcn_thermal/dataset/input_test下的json文件中的connections下面的元素不需要"EMIBType","EMIB_length","EMIB_max_width","EMIB_bump_width"这些字段
3. flow_GCN/gcn_thermal/dataset/output/placement下的json文件中不需要"connections","wirelength","area","aspect_ratio"这些字段
4. 将这些文件对应起来,按照system_i,这个i的编号就是数据集中每一条元素的标识, 帮我整理成统一的数据集结构,每一个system_i对应一个系统，
    其中包含flow_GCN/gcn_thermal/dataset/input_test下的system_i.json文件中的chiplet信息以及connection信息，
    并且包含flow_GCN/gcn_thermal/dataset/output/placement下的system_i.json文件中的placement信息

数据集格式：
{
  "system_i": {
    "system_id": "system_i",          // 字符串，系统唯一编号
    "chiplets": [                     // 数组，所有 chiplet 基本物理信息
      {
        "name": "str",                // chiplet 名称（A/B/C/D/E）
        "width": float,               // 宽度
        "height": float,              // 高度
        "power": int/float            // 功耗
      }
    ],
    "connections": [                  // 数组，chiplet 之间的连接关系
      {
        "node1": "str",               // 连接的第一个 chiplet
        "node2": "str",               // 连接的第二个 chiplet
        "wireCount": int              // 连线数量（作为边权重）
      }
    ],
    "placement": [                    // 数组，chiplet 布局坐标信息
      {
        "name": "str",                // chiplet 名称
        "x-position": float,           // 布局 x 坐标
        "y-position": float,           // 布局 y 坐标
        "width": float,               // 宽度
        "height": float,              // 高度
        "rotation": int,              // 旋转角度 0/1
        "power": float                // 功耗
      }
    ]
  }
}

# STEP 4 使用hotspot得到GCN需要的thermal相关数据
flow_GCN/Dataset/dataset/hotspot/gen.sh 1 1000
为system_1.json 到 system_1000.json生成hotspot所需的.clp