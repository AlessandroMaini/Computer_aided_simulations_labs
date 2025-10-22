from enum import Enum
from typing import Optional
from queue import PriorityQueue
import random

# ===================================================
# EVENT DEFINITION
# ===================================================

class EventType(Enum):
    ARRIVAL = "ARRIVAL"
    DEPARTURE = "DEPARTURE"


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
        self.departure_time = None
    
    def service_time(self):
        if self.departure_time is not None:
            return self.departure_time - self.arrival_time
        else:
            return None
        

# ===================================================
# PROCESSOR SHARING SYSTEM
# ===================================================

class ProcessorSharingSystem:
    def __init__(self, capacity: float):

        self.current_time = 0.0
        self.capacity = capacity  # Total processing capacity
        self.customers_in_system: dict[int, Client] = {}
        self.per_client_capacity: float = capacity  # Capacity per client (will be updated)
        self.served_clients: list[Client] = []

        self.total_arrivals = 0
        self.total_served = 0
        self.total_dropped = 0

        self.last_event_time = 0.0
        self.area_under_system_curve = 0.0  # For average number in system calculation
        self.last_system_size = 0

        self.sampled_system_size: list[tuple[float, int]] = []  # (time, system_size)

    def update_time_weighted_stats(self, new_time: float):
        time_delta = new_time - self.last_event_time
        
        # Accumulate area under curves
        self.area_under_system_curve += self.last_system_size * time_delta
        
        self.last_event_time = new_time

    def sample_system_size(self):
        self.sampled_system_size.append((self.current_time, len(self.customers_in_system)))
        
    def is_system_full(self):
        if len(self.customers_in_system) >= self.capacity:
            return True
        return False
    
    def add_client(self):
        self.last_system_size += 1
        self.update_per_client_capacity()

    def release_client(self):
        self.total_served += 1
        self.last_system_size -= 1
        self.update_per_client_capacity()

    def update_per_client_capacity(self):
        n = self.last_system_size
        if n > 0:
            self.per_client_capacity = self.capacity / n
        else:
            self.per_client_capacity = self.capacity

    def get_system_size_histogram(self) -> dict[int, int]:
        histogram = {}
        for _, size in self.sampled_system_size:
            histogram[size] = histogram.get(size, 0) + 1
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
    def handle_arrival(event: Event, ps_system: ProcessorSharingSystem, fes: FutureEventSet, 
                       service_rate: float, arrival_rate: float):
        ps_system.total_arrivals += 1
        client = Client(id=event.client_id, arrival_time=event.time)

        if not ps_system.is_system_full():
            ps_system.customers_in_system[client.id] = client
            prev_capacity = ps_system.per_client_capacity
            ps_system.add_client()
            # Schedule departure
            new_service_rate = service_rate * ps_system.per_client_capacity / ps_system.capacity
            service_time = random.expovariate(new_service_rate)
            departure_event = Event(time=event.time + service_time, 
                                    type=EventType.DEPARTURE, client_id=client.id)
            fes.schedule(departure_event)
            EventHandler.update_service_time_for_all_clients(event, ps_system, fes, prev_capacity)
        else:
            # Drop the client
            ps_system.total_dropped += 1

        # Schedule next arrival
        inter_arrival_time = random.expovariate(arrival_rate)
        next_arrival_event = Event(time=event.time + inter_arrival_time, 
                                   type=EventType.ARRIVAL, client_id=event.client_id + 1)
        fes.schedule(next_arrival_event)

    @staticmethod
    def handle_departure(event: Event, ps_system: ProcessorSharingSystem, fes: FutureEventSet, 
                         service_rate: float):
        client = ps_system.customers_in_system.pop(event.client_id)
        client.departure_time = event.time
        ps_system.served_clients.append(client)
        prev_capacity = ps_system.per_client_capacity
        ps_system.release_client()
        EventHandler.update_service_time_for_all_clients(event, ps_system, fes, prev_capacity)

    @staticmethod
    def update_service_time_for_all_clients(event: Event, ps_system: ProcessorSharingSystem, fes: FutureEventSet, 
                                        prev_capacity: float):
        new_capacity = ps_system.per_client_capacity
        current_time = event.time
        for other_client in ps_system.customers_in_system.values():
            if other_client.id == event.client_id:
                continue
            # Cancel old departure
            for ev in list(fes.events.queue):
                if ev.type == EventType.DEPARTURE and ev.client_id == other_client.id and not ev.is_cancelled():
                    original_dep_time = ev.time
                    ev.cancel()
            # Reschedule with new service rate
            remaining_time = original_dep_time - current_time
            new_remaining_time = remaining_time * (prev_capacity / new_capacity)
            new_departure_event = Event(time=current_time + new_remaining_time, 
                                        type=EventType.DEPARTURE, client_id=other_client.id)
            fes.schedule(new_departure_event)


# ===================================================
# SIMULATION ENGINE
# ===================================================

class ProcessorSharingSimulator:

    def __init__(self, capacity: float, arrival_rate: float, service_rate: float, sim_time: float):
        self.ps_system = ProcessorSharingSystem(capacity=capacity)
        self.fes = FutureEventSet()
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.sim_time = sim_time
        self.next_client_id = 1

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
            self.ps_system.update_time_weighted_stats(event.time)
            self.ps_system.sample_system_size()
            self.ps_system.current_time = event.time

            if event.type == EventType.ARRIVAL:
                EventHandler.handle_arrival(event, self.ps_system, self.fes, 
                                            self.service_rate, self.arrival_rate)
                self.next_client_id += 1
            elif event.type == EventType.DEPARTURE:
                EventHandler.handle_departure(event, self.ps_system, self.fes, 
                                              self.service_rate)

    def get_statistics(self):
        avg_system_length = (self.ps_system.area_under_system_curve / self.sim_time
                             if self.sim_time > 0 else 0)
        perc_dropped = (self.ps_system.total_dropped / self.ps_system.total_arrivals * 100
                        if self.ps_system.total_arrivals > 0 else 0)
        avg_service_time = (sum(c.service_time() for c in self.ps_system.served_clients if c.service_time() is not None) /
                            self.ps_system.total_served if self.ps_system.total_served > 0 else 0)
        return {
            "Total Arrivals": self.ps_system.total_arrivals,
            "Total Served": self.ps_system.total_served,
            "Total Dropped": self.ps_system.total_dropped,
            "Percentage Dropped": perc_dropped,
            "Average System Size": avg_system_length,
            "Average Service Time": avg_service_time,
        }
    
    def print_statistics(self):
        stats = self.get_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")
        # Print histograms
        system_size_histogram = self.ps_system.get_system_size_histogram()
        total_samples = sum(system_size_histogram.values())
        visual_system_size_histogram = [int(value / total_samples * 50) for value in system_size_histogram.values()]
        print("System Size Histogram:")
        for size in sorted(system_size_histogram.keys()):
            print(f" Size {size}:\t{"#" * visual_system_size_histogram[size]} ({system_size_histogram[size]})")


# ===================================================
# MAIN EXECUTION
# ===================================================

if __name__ == "__main__":
    random.seed(42)

    # Simulation parameters
    ARRIVAL_RATE = 1/1.11  # Average of 1 time unit between arrivals
    SERVICE_RATE = 1/1.0  # Average of 10 time units per service
    SIM_TIME = 10000.0     # Total simulation time

    # Setups
    CAPACITY = [10, 50, 100]

    # Initialize and run the simulator for each configuration
    for capacity in CAPACITY:
        print(f"\n--- Simulation with capacity {capacity} and processor sharing ---")
        simulator = ProcessorSharingSimulator(capacity, ARRIVAL_RATE, SERVICE_RATE, SIM_TIME)
        simulator.event_loop()
        simulator.print_statistics()
