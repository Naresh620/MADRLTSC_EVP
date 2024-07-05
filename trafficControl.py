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
                  n_actions=4, layer1_size=64, layer2_size=64)
score_history = []
score = 0
steps_in_phase = 0
curr_state = {}

def performAction(agent,curr_state,tl_id):
    observation = [curr_state[tl_id].phase,curr_state[tl_id].steps_in_phase]
    for lane in curr_state[tl_id].controlled_lanes:
        observation.append((1+curr_state[tl_id].lane_waiting_time[lane])/(1+curr_state[tl_id].lane_vehicle_count[lane]))
        observation.append(curr_state[tl_id].lane_vehicle_count[lane])
    return agent.choose_action(np.array(observation))

x = [0]
sd = [0]
mean = [0]
vc = [0]
while step < 3500:
    
    vehicle_ids = traci.vehicle.getIDList()
    lane_vehicle_count = defaultdict(int)
    lane_waiting_time = defaultdict(int)
    for v_id in vehicle_ids:
        lane = traci.vehicle.getLaneID(v_id)
        vType = traci.vehicle.getVehicleClass(v_id)
        factor = 1
        if vType == "emergency":
            factor = 10000
        lane_vehicle_count[lane] += 100
        lane_waiting_time[lane] += factor * traci.vehicle.getWaitingTime(v_id)

    curr_step = (step/4 %2)*2
    traffic_light_ids = traci.trafficlight.getIDList()
    for tl_id in traffic_light_ids:
        id = tl_id
        curr_phase = traci.trafficlight.getPhase(tl_id)
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        curr_state[tl_id] = State(tl_id,curr_phase,controlled_lanes,lane_waiting_time,lane_vehicle_count,steps_in_phase)    
        action = performAction(agent,curr_state,tl_id)
        print("Action",action)
        traci.trafficlight.setPhase(tl_id,(action%2)*2)
    


    traci.simulationStep()
    
    vehicle_ids = traci.vehicle.getIDList()
    average_waiting_time = 0
    vehicle_count = 0
    lane_vehicle_count = defaultdict(int)
    lane_waiting_time = defaultdict(int)
    for v_id in vehicle_ids:
        lane = traci.vehicle.getLaneID(v_id)
        vType = traci.vehicle.getVehicleClass(v_id)
        factor = 1
        if vType == "emergency":
            factor = 1000
        lane_vehicle_count[lane] += 1
        lane_waiting_time[lane] += factor * traci.vehicle.getWaitingTime(v_id)
        vehicle_count += 1

    

    traffic_light_ids = traci.trafficlight.getIDList()
    for tl_id in traffic_light_ids:
        id = tl_id
        curr_phase = traci.trafficlight.getPhase(tl_id)
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        observation = [curr_state[tl_id].phase,curr_state[tl_id].steps_in_phase]
        curr_state[tl_id] = State(tl_id,curr_phase,controlled_lanes,lane_waiting_time,lane_vehicle_count,steps_in_phase)
        vehicle_count = 1
        lane_count = 1
        for lane in curr_state[tl_id].controlled_lanes:
            average_waiting_time += (1+curr_state[tl_id].lane_waiting_time[lane])/(1+curr_state[tl_id].lane_vehicle_count[lane]);
            observation.append((1+curr_state[tl_id].lane_waiting_time[lane])/(1+curr_state[tl_id].lane_vehicle_count[lane]))
            observation.append(curr_state[tl_id].lane_vehicle_count[lane])
            vehicle_count += curr_state[tl_id].lane_vehicle_count[lane]
            lane_count += 1

        average_waiting_time /= lane_count
        standard_deviation = 1
        for lane in curr_state[tl_id].controlled_lanes:
            standard_deviation += (((1+curr_state[tl_id].lane_waiting_time[lane])/(1+curr_state[tl_id].lane_vehicle_count[lane]))-(average_waiting_time))**2;
        standard_deviation /= lane_count
        sd.append(standard_deviation)
        mean.append(average_waiting_time)
        vc.append(vehicle_count)
        n = len(mean)
        reward = - average_waiting_time 
        print("reward:",reward)
        print("Standard Deviation:",average_waiting_time,"Mean:",standard_deviation)
        agent.store_transition(observation, action, reward)
        score += reward
        
    
    if step%10 == 0:
        score_history.append(score)
        agent.learn()  
        print('Episode: ', step/4,'score: %.1f' % score)

    x.append(step)
    step += 1

import matplotlib.pyplot as plt
import numpy as np
xpoints_scaled = (xpoints - np.min(xpoints)) / (np.max(xpoints) - np.min(xpoints))
ypoints1_scaled = (ypoints1 - np.min(ypoints1)) / (np.max(ypoints1) - np.min(ypoints1))
ypoints2_scaled = (ypoints2 - np.min(ypoints2)) / (np.max(ypoints2) - np.min(ypoints2))

plt.plot(xpoints_scaled, ypoints1_scaled, '.')
plt.plot(xpoints_scaled, ypoints2_scaled, '.')
plt.show()

traci.close()

