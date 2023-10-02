"""Optionally define a custom strategy.

Needed only when the strategy is not yet implemented in Flower or because you want to
extend or modify the functionality of an existing strategy.
"""
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Union

import flwr as fl
import torch
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from models import (
    copy_gp_to_lp,
    get_parameters,
    get_state_dict_from_param,
    param_idx_to_local_params,
    param_model_rate_mapping,
)
import copy
from utils import make_optimizer, make_scheduler
import numpy as np


class HeteroFL(fl.server.strategy.Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        shape_of_model_rate = None,
        net=None,
        optim_scheduler_settings = None,
        evaluate_fn = None
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        # # created client_to_model_mapping
        # self.client_to_model_rate_mapping: Dict[str, ClientProxy] = {}

        self.net = net
        self.optim_scheduler_settings = optim_scheduler_settings
        self.shape_of_model_rate = shape_of_model_rate
        # self.active_cl_mr = None
        self.active_cl_labels = None
        # required for scheduling the lr
        self.optimizer = None,
        self.scheduler = None

    def __repr__(self) -> str:
        return "HeteroFL"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""

        ndarrays = get_parameters(self.net)

        self.active_cl_labels = client_manager.client_label_split.copy()
        self.optimizer = make_optimizer( self.optim_scheduler_settings["optimizer"], self.net.parameters() , lr=self.optim_scheduler_settings["lr"], momentum=self.optim_scheduler_settings["momentum"], weight_decay=self.optim_scheduler_settings["weight_decay"])
        self.scheduler = make_scheduler(self.optim_scheduler_settings["scheduler"], self.optimizer , milestones=self.optim_scheduler_settings["milestones"])
        return fl.common.ndarrays_to_parameters(ndarrays)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        print("in configure fit , server round no. = {}".format(server_round))
        # Sample clients
        # no need to change this
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        # for sampling we pass the criterion to select the required clients
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )

        # update client model rate mapping
        client_manager.update(server_round)

        global_parameters = get_state_dict_from_param(self.net , parameters)

        # self.active_cl_mr = OrderedDict()

        # Create custom configs
        fit_configurations = []
        lr = self.optimizer.param_groups[0]['lr']
        print(f'lr = {lr}')

        for idx, client in enumerate(clients):
            model_rate = client_manager.get_client_to_model_mapping(client.cid)
            client_values_shape = self.shape_of_model_rate[model_rate]
            local_param = copy_gp_to_lp(global_parameters , client_values_shape)
            # self.active_cl_mr[client.cid] = model_rate
            # local param are in the form of state_dict, so converting them only to values of tensors
            local_param_fitres = [v.cpu() for v in local_param.values()]
            fit_configurations.append((client, FitIns(ndarrays_to_parameters(local_param_fitres), {'lr' : lr})))

        self.scheduler.step()
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        print(f'In Aggregate Tiger Nageswara rao and number of client proxys = {len(results)}')

        def aggregate_layer(layer_updates):
            # print("in aggregate in-between layers")
            """Padding layers with 0 to max size, then average them"""
            # Get the layer's largest form
            max_ch = np.max([np.shape(l) for l in layer_updates], axis=0)
            layer_agg = np.zeros(max_ch)
            count_layer = np.zeros(max_ch)  # to average by num of models that size
            for l in layer_updates:
                local_ch = np.shape(l)
                pad_shape = [(0, a) for a in (max_ch - local_ch)]
                l_padded = np.pad(l, pad_shape, constant_values=0.0)
                ones_of_shape = np.ones(local_ch)
                ones_pad = np.pad(ones_of_shape, pad_shape, constant_values=0.0)
                count_layer = np.add(count_layer, ones_pad)
                layer_agg = np.add(layer_agg, l_padded)
            if np.any(count_layer == 0.0):
                print(count_layer)
                raise ValueError("Diving with 0")
            layer_agg = layer_agg / count_layer
            return layer_agg


        def aggregate_layer_weight(layer_updates):
            # print("in aggregate in-between layers")
            """Padding layers with 0 to max size, then average them"""
            # Get the layer's largest form
            max_ch = np.max([np.shape(l) for l in layer_updates], axis=0)
            layer_agg = np.zeros(max_ch)
            count_layer = np.zeros(max_ch)  # to average by num of models that size
            for m , l in enumerate(layer_updates):              
                local_ch = np.shape(l)
                pad_shape = [(0, a) for a in (max_ch - local_ch)]
                l_padded = np.pad(l, pad_shape, constant_values=0.0)
                
                ones_of_shape = np.ones(local_ch) 
                ones_pad = np.pad(ones_of_shape, pad_shape, constant_values=0.0)
                count_layer = np.add(count_layer, ones_pad)
                layer_agg = np.add(layer_agg, l_padded)
            count_layer[count_layer == 0] = 1
            if np.any(count_layer == 0.0):
                print(count_layer)
                raise ValueError("Diving with 0")
            layer_agg = layer_agg / count_layer
            # print(f'aggregated layer = {layer_agg}')
            return layer_agg

        
        def aggregate_last_layer(layer_updates, label_split):
            # print('in aggregate last layer')
            """Padding layers with 0 to max size, then average them"""
            # Get the layer's largest form
            max_ch = np.max([np.shape(l) for l in layer_updates], axis=0)
            layer_agg = np.zeros(max_ch)
            count_layer = np.zeros(max_ch)  # to average by num of models that size
            for m , l in enumerate(layer_updates):
                local_ch = np.shape(l)
                label_mask = np.zeros(local_ch)
                label_mask[label_split[m].type(torch.int).numpy()] = 1
                l = l * label_mask
                ones_of_shape = np.ones(local_ch)
                ones_of_shape =  ones_of_shape * label_mask
                count_layer = np.add(count_layer, ones_of_shape)
                layer_agg = np.add(layer_agg, l)
            count_layer[count_layer == 0] = 1
            if np.any(count_layer == 0.0):
                raise ValueError("Diving with 0")
            layer_agg = layer_agg / count_layer
            return layer_agg

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer for layer in parameters_to_ndarrays(fit_res.parameters)] for _ , fit_res in results
        ]

        global_values = []
        for v in self.net.state_dict().values():
            global_values.append(copy.deepcopy(v))

        num_layers = len(weighted_weights[0])

        label_split = [self.active_cl_labels[int(cp.cid)].type(torch.int) for cp, _ in results]

        flat_label_split = torch.cat([t.view(-1) for t in label_split])
        unique_labels = torch.unique(flat_label_split)
        all_labels = torch.arange(10)
        missing_labels = torch.tensor(list(set(all_labels.tolist()) - set(unique_labels.tolist())))
        print(f'missing labels = {missing_labels}')


        agg_layers = []
        last_weight_layer = [None for i in range(10)]
        for i, l in enumerate(zip(*weighted_weights), start=1):
            if( i == num_layers - 1):
                
                for i in range(10):
                    agg_layer_list = []
                    for clnt , layer in enumerate(l):
                        if(torch.any(label_split[clnt] == i)):
                            agg_layer_list.append(layer[i])
                    if len(agg_layer_list) != 0:
                        last_weight_layer[i] = (aggregate_layer_weight(agg_layer_list))
                        # pad with prev.
            elif(i == num_layers):
                store_prev = {}
                
                for k in missing_labels:
                    k = k.item()
                    store_prev[k] = global_values[-1][k]
                
                global_values[-1] = torch.from_numpy((aggregate_last_layer(l , label_split)))
                
                for k in missing_labels:
                    k = k.item()
                    global_values[-1][k] = store_prev[k]
            else:
                agg_layers.append(aggregate_layer(l))
        


        # keep the rest of global parameters as it is
        for i , layer in enumerate(agg_layers):
            layer = torch.from_numpy(layer)
            layer_shape = layer.shape
            slices = [slice(0, dim) for dim in layer_shape]
            global_values[i][slices] = layer
        
        for i , inner_layer in enumerate(last_weight_layer):

            if(inner_layer is None):
                continue
            
            inner_layer = torch.from_numpy(inner_layer)
            inner_layer_shape = inner_layer.shape
            slices = [slice(0, dim) for dim in inner_layer_shape]
            global_values[num_layers - 2][i][:inner_layer_shape[0]] = inner_layer

        
        new_state_dict = {}
        for i , k in enumerate(self.net.state_dict().keys()):
            new_state_dict[k] = global_values[i]
        self.net.load_state_dict(new_state_dict)
        

        for i in range(len(global_values)):
            global_values[i] = global_values[i].numpy()


        return ndarrays_to_parameters(global_values) , {}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # print('In configure evaluate')
        # # if self.fraction_evaluate == 0.0:
        # #     return []
        # # config = {}
        # # evaluate_ins = EvaluateIns(parameters, config)

        # sample_size, min_num_clients = self.num_fit_clients(
        #     client_manager.num_available()
        # )

        # # for sampling we pass the criterion to select the required clients
        # clients = client_manager.sample(
        #     num_clients=sample_size,
        #     min_num_clients=min_num_clients,
        # )

        # global_parameters = get_state_dict_from_param(self.net , parameters)

        # # Create custom configs
        # evaluate_configurations = []

        # for idx, client in enumerate(clients):
        #     model_rate = client_manager.get_client_to_model_mapping(client.cid)
        #     client_values_shape = self.shape_of_model_rate[model_rate]
        #     local_param = copy_gp_to_lp(global_parameters , client_values_shape)
        #     # local param are in the form of state_dict, so converting them only to values of tensors
        #     local_param_fitres = [v.cpu() for v in local_param.values()]
        #     evaluate_configurations.append((client, EvaluateIns(ndarrays_to_parameters(local_param_fitres), {})))

        # return evaluate_configurations

        return None

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        # if not results:
        #     return None, {}
        # print('In aggregate evaluate')
        # print(f'len of results = {len(results)}')
        # loss_aggregated = weighted_loss_avg(
        #     [
        #         (evaluate_res.num_examples, evaluate_res.loss)
        #         for _, evaluate_res in results
        #     ]
        # )

        # accuracy_aggregated = 0
        # for cp, y in results:
        #     # print(f"{cp.cid}-->{y.metrics['accuracy']}", end=" ")
        #     accuracy_aggregated += y.metrics["accuracy"]
        # accuracy_aggregated /= len(results)

        # metrics_aggregated = {"accuracy": accuracy_aggregated}
        # print(f"\npaneer lababdar {metrics_aggregated}")
        # return loss_aggregated, metrics_aggregated
    
        return None

    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics
    
    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
