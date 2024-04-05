#from trajectory_utils import prediction_output_to_trajectories
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns

def prediction_output_to_trajectories(prediction_output_dict,
                                      dt,
                                      max_h,
                                      ph,
                                      map=None,
                                      prune_ph_to_future=False):

    prediction_timesteps = prediction_output_dict.keys()

    output_dict = dict()
    histories_dict = dict()
    futures_dict = dict()

    for t in prediction_timesteps:
        histories_dict[t] = dict()
        output_dict[t] = dict()
        futures_dict[t] = dict()
        prediction_nodes = prediction_output_dict[t].keys()
        for node in prediction_nodes:
            predictions_output = prediction_output_dict[t][node]
            position_state = {'position': ['x', 'y']}

            history = node.get(np.array([t - max_h, t]), position_state)  # History includes current pos
            history = history[~np.isnan(history.sum(axis=1))]
            #pdb.set_trace()
            future = node.get(np.array([t + 1, t + ph]), position_state)
            # replace nan to 0
            #future[np.isnan(future)] = 0
            future = future[~np.isnan(future.sum(axis=1))]

            if prune_ph_to_future:
                predictions_output = predictions_output[:, :, :future.shape[0]]
                if predictions_output.shape[2] == 0:
                    continue

            trajectory = predictions_output

            if map is None:
                histories_dict[t][node] = history
                output_dict[t][node] = trajectory
                futures_dict[t][node] = future
            else:
                histories_dict[t][node] = map.to_map_points(history)
                output_dict[t][node] = map.to_map_points(trajectory)
                futures_dict[t][node] = map.to_map_points(future)

    return output_dict, histories_dict, futures_dict



def plot_trajectories(ax,
                      prediction_dict,
                      histories_dict,
                      futures_dict,
                      line_alpha=0.7,
                      line_width=0.2,
                      edge_width=2,
                      circle_edge_width=0.5,
                      node_circle_size=0.3,
                      batch_num=0,
                      kde=False):

    # ax.axis('scaled')
    # ax.axis([-2,5,0.5,4])
    # kde=True

    cmap = ['k', 'b', 'y', 'g', 'r']

    for node in histories_dict:
        history = histories_dict[node]
        future = futures_dict[node]
        predictions = prediction_dict[node]

        if np.isnan(history[-1]).any():
            continue

        ax.plot(history[:, 0], history[:, 1], 'k--')

        for sample_num in range(prediction_dict[node].shape[1]):

            if kde and predictions.shape[1] >= 50:
                line_alpha = 0.2
                #print(predictions.shape[2])
                #for t in range(predictions.shape[2]):
                t = predictions.shape[2] - 1
                    # sns.kdeplot(predictions[batch_num, :, t, 0], predictions[batch_num, :, t, 1],
                    #             ax=ax, shade=True, shade_lowest=False,
                    #             color=np.random.choice(cmap), alpha=0.8)
                sns.kdeplot(
                    x=predictions[batch_num, :, t, 0], 
                    y=predictions[batch_num, :, t, 1],
                    ax=ax, 
                    fill=True, 
                    thresh=0.05,
                    color="blue",#np.random.choice(cmap), 
                    alpha=0.8
                )

            # ax.plot(predictions[batch_num, sample_num, :, 0], predictions[batch_num, sample_num, :, 1],
            #         color=cmap[node.type.value],
            #         linewidth=line_width, alpha=line_alpha)

            ax.plot(future[:, 0],
                    future[:, 1],
                    'w--',
                    path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])

            # Current Node Position
            circle = plt.Circle((history[-1, 0],
                                 history[-1, 1]),
                                node_circle_size,
                                facecolor='g',
                                edgecolor='k',
                                lw=circle_edge_width,
                                zorder=3)
            ax.add_artist(circle)
    #ax.axis('equal')


def calculate_axis_limits(prediction_dict):
    # 全ての軌道から最小および最大の座標を見つける
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    for trajectory in prediction_dict.values():
        xs, ys = trajectory[:, 0], trajectory[:, 1]
        min_x, max_x = min(np.min(xs), min_x), max(np.max(xs), max_x)
        min_y, max_y = min(np.min(ys), min_y), max(np.max(ys), max_y)
    return min_x, max_x, min_y, max_y

def visualize_prediction(ax,
                         prediction_output_dict,
                         dt,
                         max_hl,
                         ph,
                         robot_node=None,
                         map=None,
                         **kwargs):

    prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(prediction_output_dict,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)

    #assert(len(prediction_dict.keys()) <= 1)
    if len(prediction_dict.keys()) == 0:
        return
    ts_key = list(prediction_dict.keys())[0]

    prediction_dict = prediction_dict[ts_key]
    histories_dict = histories_dict[ts_key]
    futures_dict = futures_dict[ts_key]
    #print(len(futures_dict.keys()))

    #min_x, max_x, min_y, max_y = calculate_axis_limits(prediction_dict)

    # ax.set_xticks(range(-8,3))
    # ax.set_xlim([-8,2])
    # ax.set_yticks([1,9])
    # ax.set_ylim(1, 8)
    # ax.set_aspect('equal', adjustable='box')

    if map is not None:
        ax.imshow(map.as_image(), origin='lower', alpha=0.5)
    plot_trajectories(ax, prediction_dict, histories_dict, futures_dict, *kwargs)