from enum import Enum
import math
from typing import Optional
from queue import PriorityQueue
import random

# ===================================================
# EVENT DEFINITION
# ===================================================

class EventType(Enum):
    ARRIVAL = "ARRIVAL"
    DEPARTURE = "DEPARTURE"


class ScheduleType(Enum):
    FCFS = "FCFS"
    LCFS = "LCFS"
    PRIORITY = "PRIORITY"


class Event:
    
    def __init__(self, time: float, type: EventType, client_id: int, cancelled: bool = False):
        self.time = time
        self.type = type
        self.client_id = client_id
        self.cancelled = cancelled

    def cancel(self):
        self.cancelled = True

    def is_cancelled(self) -> bool:
        return self.cancelled

    def __lt__(self, other):
        """Less than comparison for priority queue ordering"""
        if self.time != other.time:
            return self.time < other.time
        # If times are equal, use client_id as tiebreaker
        return self.client_id < other.client_id
    
    def __eq__(self, other):
        """Equality comparison"""
        return (self.time == other.time and 
                self.type == other.type and 
                self.client_id == other.client_id)


# ===================================================
# CLIENT STRUCTURE
# ===================================================

class Client:

    def __init__(self, id: int, arrival_time: float, priority: int = 1):
        self.id = id
        self.arrival_time = arrival_time
        self.service_start_time = None
        self.departure_time = None
        self.priority = priority

    def waiting_time(self):
        if self.service_start_time is not None:
            return self.service_start_time - self.arrival_time
        else:
            return None
    
    def service_time(self):
        if self.departure_time is not None and self.service_start_time is not None:
            return self.departure_time - self.service_start_time
        else:
            return None
    
    def total_time_in_system(self):
        if self.departure_time is not None:
            return self.departure_time - self.arrival_time
        else:
            return None
        

# ===================================================
# QUEUE MANAGEMENT
# ===================================================

class QueueSystem:

    def __init__(self, num_servers: int, queue_capacity: int, schedule_type: ScheduleType = ScheduleType.FCFS):
        self.num_servers = num_servers
        self.queue_capacity = queue_capacity
        self.schedule_type = schedule_type
        
        # Initial state
        self.current_time = 0.0
        self.server_busy = 0
        self.waiting_queue: list[Client] = []

        # Client tracking
        self.active_clients: dict[int, Client] = {}
        self.served_clients: list[Client] = []
        self.serving_clients: dict[int, Client] = {}

        # Statistics
        self.total_arrivals = 0
        self.total_served = 0
        self.total_dropped = 0

        # Time-weighted statistics
        self.area_under_queue_curve = 0.0
        self.area_under_system_curve = 0.0
        self.area_under_server_curve = 0.0
        self.last_event_time = 0.0
        self.last_queue_size = 0
        self.last_system_size = 0

        # Sample statistics
        self.sampled_queue_sizes: list[tuple[float, int]] = []  # (time, queue_size)

    def update_time_weighted_stats(self, new_time: float):
        time_delta = new_time - self.last_event_time
        
        # Accumulate area under curves
        self.area_under_queue_curve += self.last_queue_size * time_delta
        self.area_under_system_curve += self.last_system_size * time_delta
        self.area_under_server_curve += self.server_busy * time_delta
        
        self.last_event_time = new_time

    def sample_statistics(self):
        self.sampled_queue_sizes.append((self.current_time, len(self.waiting_queue)))

    def is_server_available(self) -> bool:
        return self.server_busy < self.num_servers
    
    def is_queue_full(self) -> bool:
        return len(self.waiting_queue) >= self.queue_capacity
    
    def allocate_server(self):
        self.server_busy += 1
        self.last_system_size += 1

    def release_server(self):
        self.server_busy -= 1
        self.last_system_size -= 1

    def add_client_to_queue(self, client: Client):
        if self.schedule_type == ScheduleType.PRIORITY:
            drop = False
            if self.is_queue_full():
                drop = True
            index = 0
            while index < len(self.waiting_queue) and self.waiting_queue[index].priority <= client.priority:
                index += 1
            self.waiting_queue.insert(index, client)
            if drop:
                dropped_client = self.waiting_queue.pop()  # Remove the lowest priority client
                self.active_clients.pop(dropped_client.id, None)
                self.last_queue_size -= 1
                self.last_system_size -= 1
                self.total_dropped += 1
        else:
            if self.is_queue_full():
                self.total_dropped += 1
                return
            else:
                if self.schedule_type == ScheduleType.FCFS:
                    self.waiting_queue.append(client)
                elif self.schedule_type == ScheduleType.LCFS:
                    self.waiting_queue.insert(0, client)

        self.active_clients[client.id] = client
        self.last_queue_size += 1
        self.last_system_size += 1

    def remove_client_from_queue(self) -> Optional[Client]:
        if self.waiting_queue:
            client = self.waiting_queue.pop(0)
            self.last_queue_size -= 1
            self.last_system_size -= 1
            return client
        return None
    
    def get_queue_histogram(self) -> dict[int, int]:
        histogram = {}
        for _, q_size in self.sampled_queue_sizes:
            histogram[q_size] = histogram.get(q_size, 0) + 1
        return histogram
    
    def get_waiting_time_histogram(self, bins: int = 10) -> dict[float, int]:
        waiting_times = [c.waiting_time() for c in self.served_clients if c.waiting_time() is not None]
        if not waiting_times:
            return {}
        min_time = min(waiting_times)
        max_time = max(waiting_times)
        bin_size = (max_time - min_time) / bins if max_time > min_time else 1
        histogram = {}
        for wt in waiting_times:
            bin_index = int((wt - min_time) / bin_size)
            bin_index = min(bin_index, bins - 1)  # Ensure it falls within the last bin
            bin_key = round(min_time + bin_index * bin_size, 2)
            histogram[bin_key] = histogram.get(bin_key, 0) + 1
        return histogram
    

# ===================================================
# FUTURE EVENT SET
# ===================================================

class FutureEventSet:

    def __init__(self):
        self.events = PriorityQueue()
        self.event_count = 0

    def schedule(self, event: Event):
        self.events.put(event)
        self.event_count += 1

    def get_next_event(self) -> Optional[Event]:
        if not self.is_empty():
            # Check for cancelled events
            while not self.events.empty():
                event = self.events.get()
                if not event.is_cancelled():
                    return event
        return None
    
    def is_empty(self) -> bool:
        return self.events.empty()
    
    def size(self) -> int:
        return self.events.qsize()
    

# ===================================================
# EVENT HANDLERS
# ===================================================

class EventHandler:

    @staticmethod
    def handle_arrival(event: Event, queue_system: QueueSystem, fes: FutureEventSet, 
                       service_rate: float, arrival_rate: float, priority_rate: list[float] = [1.0, 0.0, 0.0],
                       preemptive: bool = False):
        queue_system.total_arrivals += 1
        priority = random.choices([1, 2, 3], weights=priority_rate)[0]
        client = Client(id=event.client_id, arrival_time=event.time, priority=priority)

        # Handle preemption if the prority of the event is higher than any active client (lower number means higher priority)
        if preemptive:
            for active_client in queue_system.serving_clients.values():
                if active_client.priority > client.priority:
                    # Drop the active client
                    active_client.departure_time = event.time
                    # Cancel its departure event
                    for e in list(fes.events.queue):
                        if e.client_id == active_client.id and e.type == EventType.DEPARTURE:
                            e.cancel()
                    queue_system.served_clients.append(active_client)
                    queue_system.total_dropped += 1
                    queue_system.release_server()
                    queue_system.active_clients.pop(active_client.id)
                    queue_system.serving_clients.pop(active_client.id)
                    break

        if queue_system.is_server_available():
            # Allocate server
            queue_system.allocate_server()
            client.service_start_time = event.time
            queue_system.active_clients[client.id] = client
            queue_system.serving_clients[client.id] = client

            # Schedule departure
            service_time = random.expovariate(service_rate)
            departure_event = Event(time=event.time + service_time, 
                                    type=EventType.DEPARTURE, client_id=client.id)
            fes.schedule(departure_event)

        else:
            # Add to waiting queue
            queue_system.add_client_to_queue(client)

        # Schedule next arrival
        inter_arrival_time = random.expovariate(arrival_rate)
        next_arrival_event = Event(time=event.time + inter_arrival_time, 
                                   type=EventType.ARRIVAL, client_id=event.client_id + 1)
        fes.schedule(next_arrival_event)

    @staticmethod
    def handle_departure(event: Event, queue_system: QueueSystem, fes: FutureEventSet, 
                         service_rate: float):
        client = queue_system.active_clients.pop(event.client_id)
        queue_system.serving_clients.pop(event.client_id, None)
        client.departure_time = event.time
        queue_system.served_clients.append(client)
        queue_system.total_served += 1
        queue_system.release_server()

        # Check if there are clients waiting
        next_client = queue_system.remove_client_from_queue()
        if next_client:
            # Start serving next client
            next_client.service_start_time = event.time
            queue_system.allocate_server()
            queue_system.serving_clients[next_client.id] = next_client
            # Schedule departure for next client
            service_time = random.expovariate(service_rate)
            departure_event = Event(time=event.time + service_time, 
                                    type=EventType.DEPARTURE, client_id=next_client.id)
            fes.schedule(departure_event)


# ===================================================
# SIMULATION ENGINE
# ===================================================

class QueueSimulator:

    def __init__(self, num_servers: int, queue_capacity: int, 
                 arrival_rate: float, service_rate: float, sim_time: float, 
                 schedule_type: ScheduleType = ScheduleType.FCFS, priority_rate: list[float] = [1.0, 0.0, 0.0],
                 preemptive: bool = False):
        self.queue_system = QueueSystem(num_servers, queue_capacity, schedule_type)
        self.fes = FutureEventSet()
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.sim_time = sim_time
        self.priority_rate = priority_rate
        self.next_client_id = 1
        self.preemptive = preemptive

    def event_loop(self):
        # Schedule the first arrival
        first_arrival_event = Event(time=0.0, type=EventType.ARRIVAL, client_id=self.next_client_id)
        self.fes.schedule(first_arrival_event)
        self.next_client_id += 1

        while not self.fes.is_empty():
            event = self.fes.get_next_event()
            if event.time > self.sim_time:
                break

            # Update time-weighted statistics
            self.queue_system.update_time_weighted_stats(event.time)
            self.queue_system.sample_statistics()
            self.queue_system.current_time = event.time

            if event.type == EventType.ARRIVAL:
                EventHandler.handle_arrival(event, self.queue_system, self.fes, 
                                            self.service_rate, self.arrival_rate, self.priority_rate,
                                            preemptive=self.preemptive)
                self.next_client_id += 1
            elif event.type == EventType.DEPARTURE:
                EventHandler.handle_departure(event, self.queue_system, self.fes, 
                                              self.service_rate)

    def get_statistics(self):
        avg_queue_length = (self.queue_system.area_under_queue_curve / self.sim_time 
                            if self.sim_time > 0 else 0)
        avg_system_length = (self.queue_system.area_under_system_curve / self.sim_time 
                                if self.sim_time > 0 else 0)
        perc_dropped = (self.queue_system.total_dropped / self.queue_system.total_arrivals * 100
                        if self.queue_system.total_arrivals > 0 else 0)
        avg_waiting_time = (sum(c.waiting_time() for c in self.queue_system.served_clients if c.waiting_time() is not None) /
                            self.queue_system.total_served if self.queue_system.total_served > 0 else 0)
        std_waiting_time = ( (sum((c.waiting_time() - avg_waiting_time) ** 2 for c in self.queue_system.served_clients if c.waiting_time() is not None) 
                             / self.queue_system.total_served) ** 0.5
                             if self.queue_system.total_served > 0 else 0)
        avg_service_time = (sum(c.service_time() for c in self.queue_system.served_clients if c.service_time() is not None) /
                            self.queue_system.total_served if self.queue_system.total_served > 0 else 0)
        avg_server_utilization = (self.queue_system.area_under_server_curve / (self.sim_time * self.queue_system.num_servers)
                                  if self.sim_time > 0 else 0)
        return {
            "Total Arrivals": self.queue_system.total_arrivals,
            "Total Served": self.queue_system.total_served,
            "Total Dropped": self.queue_system.total_dropped,
            "Percentage Dropped": perc_dropped,
            "Average Queue Length": avg_queue_length,
            "Average System Size": avg_system_length,
            "Average Service Time": avg_service_time,
            "Average Waiting Time": avg_waiting_time,
            "Std Dev Waiting Time": std_waiting_time,
            "Average Server Utilization": avg_server_utilization
        }
    
    def print_statistics(self, bins: int = 10):
        stats = self.get_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        queue_histogram = self.queue_system.get_queue_histogram() 
        total_samples = sum(queue_histogram.values())
        visual_queue_histogram = [math.ceil(value / total_samples * 50) for value in queue_histogram.values()]
        print("Queue Size Histogram:")
        for size in sorted(queue_histogram.keys()):
            print(f" Size {size}:\t{"#" * visual_queue_histogram[size]} ({queue_histogram[size]})")

        waiting_histogram = self.queue_system.get_waiting_time_histogram(bins=bins)
        total_clients = sum(waiting_histogram.values())
        visual_waiting_histogram = [math.ceil(value / total_clients * 50) for value in waiting_histogram.values()]
        print("Waiting Time Histogram:")
        index = 0
        for bin_key in sorted(waiting_histogram.keys()):
            print(f" <= {bin_key}:\t{"#" * visual_waiting_histogram[index]} ({waiting_histogram[bin_key]})")
            index += 1


# ===================================================
# MAIN EXECUTION
# ===================================================

if __name__ == "__main__":
    random.seed(42)

    # Simulation parameters
    ARRIVAL_RATE = 1/5.0  # Average of 5 time units between arrivals
    SERVICE_RATE = 1/10.0  # Average of 10 time units per service
    SIM_TIME = 10000.0     # Total simulation time
    SCHEDULING = ScheduleType.LCFS
    PRIORITY_RATE = [1., 0., 0.]  # Probabilities for priorities 1, 2, 3
    PREEMPTIVE = False

    # Setups
    NUM_SERVERS = [1, 2, 3]
    QUEUE_CAPACITY = [2, 5, 10]

    # Initialize and run the simulator for each configuration
    for servers in NUM_SERVERS:
        for capacity in QUEUE_CAPACITY:
            print(f"\n--- Simulation with {servers} servers, queue capacity {capacity} and scheduling {SCHEDULING} ---")
            simulator = QueueSimulator(servers, capacity, 
                                       ARRIVAL_RATE, SERVICE_RATE, SIM_TIME, schedule_type=SCHEDULING, priority_rate=PRIORITY_RATE,
                                       preemptive=PREEMPTIVE)
            simulator.event_loop()
            simulator.print_statistics()

