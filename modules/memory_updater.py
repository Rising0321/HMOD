from torch import nn
import torch


class MemoryUpdater(nn.Module):
    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        pass


class SequenceMemoryUpdater(MemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device, passing_dep):
        super(SequenceMemoryUpdater, self).__init__()
        self.memory = memory
        self.layer_norm = torch.nn.LayerNorm(memory_dimension)
        self.message_dimension = message_dimension
        self.device = device

        self.passing_dep = passing_dep

        self.fc1s = torch.nn.ModuleList()
        for i in range(passing_dep):
            self.fc1s.append(torch.nn.Linear(message_dimension, message_dimension))
            torch.nn.init.xavier_normal_(self.fc1s[i].weight)

        self.fc2s = torch.nn.ModuleList()
        for i in range(passing_dep):
            self.fc2s.append(torch.nn.Linear(message_dimension * 2, message_dimension))
            torch.nn.init.xavier_normal_(self.fc2s[i].weight)

        self.relu1 = torch.nn.ReLU()

        self.relu2 = torch.nn.ReLU()

    def update_memory(self, unique_node_ids, unique_messages, other_message=None, timestamps=None):
        if len(unique_node_ids) <= 0:
            return
        if timestamps is not None:
            assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                              "update memory to time in the past"
            self.memory.last_update[unique_node_ids] = timestamps.float()
        ret_message = None
        if other_message is not None:
            ret_message = torch.zeros([self.passing_dep, len(unique_node_ids), self.message_dimension]).to(self.device)
            pre_message = unique_messages
            for i in range(self.passing_dep):
                message_with_diffrent_level = other_message[i][unique_node_ids]
                message_with_diffrent_level_fc = self.fc1s[i](message_with_diffrent_level)
                message_with_diffrent_level_pool = torch.max(self.relu1(message_with_diffrent_level_fc), dim=-2).values
                cat_message = torch.cat([pre_message, message_with_diffrent_level_pool], dim=1)
                pre_message = self.fc2s[i](cat_message)
                ret_message[i] = pre_message.to(self.device)
                '''
                cat_message = torch.cat([unique_messages, other_message[i][unique_node_ids].to(self.device)], dim=1)
                agg_message = self.relu1(self.fc1s[0](cat_message))
                cat_message2 = torch.cat([pre_message, agg_message], dim=1)
                pre_message = self.relu2(self.fc2s[0](cat_message2))
                ret_message[i] = pre_message.to(self.device)
                '''
                '''
                unique_messages = self.fc2(
                    torch.cat([unique_messages,
                               self.fc1(
                                   torch.cat([unique_messages, other_message[unique_node_ids].to(self.device)],
                                             dim=1))],
                              dim=1))
                '''
            unique_messages = pre_message
        memory = self.memory.get_memory()[unique_node_ids]

        updated_memory = self.memory_updater(unique_messages, memory)

        self.memory.set_memory(unique_node_ids, updated_memory.detach())
        return ret_message

    def get_updated_memory(self, unique_node_ids, unique_messages, other_message=None, timestamps=None, memory=None):
        if len(unique_node_ids) <= 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()
        if memory is None:
            updated_memory = self.memory.memory.data.clone()
        else:
            updated_memory = memory

        # print(other_message.shape[1])
        if other_message is not None:

            pre_message = unique_messages
            for i in range(self.passing_dep):
                message_with_diffrent_level = other_message[i][unique_node_ids]
                message_with_diffrent_level_fc = self.fc1s[i](message_with_diffrent_level)
                message_with_diffrent_level_pool = torch.max(self.relu1(message_with_diffrent_level_fc), dim=-2).values

                cat_message = torch.cat([pre_message, message_with_diffrent_level_pool], dim=1)
                pre_message = self.fc2s[i](cat_message)

            unique_messages = pre_message
        # print(unique_messages.shape[1])
        updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])

        updated_last_update = self.memory.last_update.data.clone()
        if timestamps is not None:
            assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                              "update memory to time in the past"
            updated_last_update[unique_node_ids] = timestamps.float()

        return updated_memory, updated_last_update


class GRUMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device, passing_dep):
        super(GRUMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device, passing_dep)

        self.memory_updater = nn.GRUCell(input_size=message_dimension,
                                         hidden_size=memory_dimension)


class RNNMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device, passing_dep):
        super(RNNMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device, passing_dep)

        self.memory_updater = nn.RNNCell(input_size=message_dimension,
                                         hidden_size=memory_dimension)


def get_memory_updater(module_type, memory, message_dimension, memory_dimension, device, passing_dep):
    if module_type == "gru":
        return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device, passing_dep)
    elif module_type == "rnn":
        return RNNMemoryUpdater(memory, message_dimension, memory_dimension, device, passing_dep)
