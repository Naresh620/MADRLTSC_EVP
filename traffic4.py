import traci
from collections import defaultdict
from deep_policy_network import Agent
import numpy as np
import math 

sumoBinary = "/usr/bin/sumo-gui"
sumoCmd = [sumoBinary, "-c", "simple.sumocfg"]
traci.start(sumoCmd)

class State:
    def __init__(self, tl_id, curr_phase, controlled_lanes, lane_waiting_time, lane_vehicle_count,steps_in_phase):
        self.tl_id = tl_id
        self.phase = curr_phase
        self.controlled_lanes = ['1to2_0','1to2_1','1to2_2','5to2_0','5to2_1','5to2_2','3to2_0','3to2_1','3to2_2','4to2_0','4to2_1','4to2_2']
        self.lane_waiting_time = defaultdict(int)
        self.lane_vehicle_count = defaultdict(int)
        self.steps_in_phase = steps_in_phase
        for lane_id in controlled_lanes:
            self.lane_waiting_time[lane_id] = lane_waiting_time[lane_id]
            self.lane_vehicle_count[lane_id] = lane_vehicle_count[lane_id]
 
step = 1 
agent = Agent(ALPHA=0.0005, input_dims=26, GAMMA=0.99,
                  n_actions=2, layer1_size=128, layer2_size=256)
score_history = []
score = 0
steps_in_phase = 0
curr_state = {}

s = [0]
avg = [0]
vc = [0]
def performAction(agent,curr_state,tl_id):
    observation = [curr_state[tl_id].phase,curr_state[tl_id].steps_in_phase]
    for lane in curr_state[tl_id].controlled_lanes:
        observation.append((1+curr_state[tl_id].lane_waiting_time[lane])/(1+curr_state[tl_id].lane_vehicle_count[lane]))
        observation.append(curr_state[tl_id].lane_vehicle_count[lane])
    return agent.choose_action(np.array(observation))

while step < 2000:
    
    vehicle_ids = traci.vehicle.getIDList()
    lane_vehicle_count = defaultdict(int)
    lane_waiting_time = defaultdict(int)
    for v_id in vehicle_ids:
        lane = traci.vehicle.getLaneID(v_id)
        lane_vehicle_count[lane] += 1
        lane_waiting_time[lane] += traci.vehicle.getWaitingTime(v_id)

    traffic_light_ids = traci.trafficlight.getIDList()
    
    for tl_id in traffic_light_ids:
        id = tl_id
        curr_phase = traci.trafficlight.getPhase(tl_id)
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        curr_state[tl_id] = State(tl_id,curr_phase,controlled_lanes,lane_waiting_time,lane_vehicle_count,steps_in_phase)    
        action = performAction(agent,curr_state,tl_id)
        traci.trafficlight.setPhase(tl_id,action*2)

    traci.simulationStep()
    
    vehicle_ids = traci.vehicle.getIDList()
    average_waiting_time = 0
    
    lane_vehicle_count = defaultdict(int)
    lane_waiting_time = defaultdict(int)
    for v_id in vehicle_ids:
        lane = traci.vehicle.getLaneID(v_id)
        lane_vehicle_count[lane] += 1
        lane_waiting_time[lane] += traci.vehicle.getWaitingTime(v_id)

    

    traffic_light_ids = traci.trafficlight.getIDList()
    for tl_id in traffic_light_ids:
        id = tl_id
        curr_phase = traci.trafficlight.getPhase(tl_id)
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        observation = [curr_state[tl_id].phase,curr_state[tl_id].steps_in_phase]
        curr_state[tl_id] = State(tl_id,curr_phase,controlled_lanes,lane_waiting_time,lane_vehicle_count,steps_in_phase)
        vehicle_count = 0
        for lane in curr_state[tl_id].controlled_lanes:
            average_waiting_time += (1+curr_state[tl_id].lane_waiting_time[lane])/(1+curr_state[tl_id].lane_vehicle_count[lane]);
            lane_average = (1+curr_state[tl_id].lane_waiting_time[lane])/(1+curr_state[tl_id].lane_vehicle_count[lane])
            vehicle_count += curr_state[tl_id].lane_vehicle_count[lane]
            observation.append(lane_average)
            observation.append(curr_state[tl_id].lane_vehicle_count[lane])
        s.append(step)
        avg.append(average_waiting_time)
        vc.append(vehicle_count)
        reward = - vehicle_count
        print("Reward:",reward)
        agent.store_transition(observation, action, reward)
        score += reward
        
    
    if step%10 == 0:
        score_history.append(score)
        agent.learn()  

    
    step += 1
import json
with open('v.json', 'w') as f:
    json.dump(avg, f)
    json.dump(vc, f)
traci.close()

