import tensorlayerx as tlx
def to_homograph(x_dict ,edge_index_dict, num_nodes_dict, edge_value_dict):
    node_type_num = len(num_nodes_dict)
    x_list = list(x_dict)
    node_type_list = list(num_nodes_dict.keys())
    node_num_list = list(num_nodes_dict.values())
    add_num_list = add_num(node_num_list)
    # print(add_num_list)
    x_feature = x_dict[node_type_list[0]]
    x_feature = tlx.stack(x_feature)

    for i in node_type_list[1:]:

        x_feature = tlx.concat((x_feature,tlx.stack(x_dict[i])), axis=0)

    edge_type_list = list(edge_index_dict.keys())

    edge_index = edge_index_dict[edge_type_list[0]]

    _,num = edge_index.shape

    node_src = tlx.slice(edge_index,[0,0],[1,num])
    node_dst = tlx.slice(edge_index,[1,0],[1,num])

    node_src = tlx.add(node_src,add_num_list[node_type_list.index(edge_type_list[0][0])])
    node_dst = tlx.add(node_dst,add_num_list[node_type_list.index(edge_type_list[0][2])])

    edge_index = tlx.concat((tlx.stack(node_src),tlx.stack(node_dst)),axis=0)
    
    edge_value = edge_value_dict[edge_type_list[0]]
    edge_value = tlx.stack(edge_value[0])
    # print('edge_value:',edge_value)


    for i in edge_type_list[1:]:
        edge_index_tem = edge_index_dict[i]
        _,num = edge_index_tem.shape

        node_src = tlx.slice(edge_index_tem,[0,0],[1,num])
        node_dst = tlx.slice(edge_index_tem,[1,0],[1,num])

        node_src = tlx.add(node_src,add_num_list[node_type_list.index(i[0])])
        node_dst = tlx.add(node_dst,add_num_list[node_type_list.index(i[2])])

        edge_index_tem = tlx.concat((tlx.stack(node_src),tlx.stack(node_dst)),axis=0)
        edge_index = tlx.concat((edge_index,tlx.stack(edge_index_tem)),axis=1)
        # print('-----')
        # print(edge_value)
        # print(edge_value_dict[i])

        edge_value = tlx.concat((edge_value,tlx.stack(edge_value_dict[i][0])),axis=0)

        
    return [x_feature, edge_index, edge_value]

def to_heterograph(x_value, edge_index, num_node_dict):

    x_dict = {}
    node_type_list = list(num_node_dict.keys())
    node_num_list = list(num_node_dict.values())
    node_index_list =  add_num(node_num_list)
    type_num = len(node_type_list)

    for node_type in node_type_list:
        index = node_type_list.index(node_type)
        if(index == type_num-1):
            x_dict[node_type] = x_value[node_index_list[index]:,:]
        else:
            x_dict[node_type] = x_value[node_index_list[index]:node_index_list[index+1],:]
    return x_dict

    # print(x_feature)
    # for node_type, x_node in x_dict.items():

def add_num(int_list):
    out = []
    num = len(int_list)
    out.append(0)
    for i in range(num-1):
        out.append(out[-1]+int_list[i])
    return out
