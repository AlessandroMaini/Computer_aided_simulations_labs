from enum import Enum
import math
from typing import Optional
from queue import PriorityQueue
import random
from transient_detection import TransientDetection

# ===================================================
# EVENT DEFINITION
# ===================================================

class EventType(Enum):
    ARRIVAL = "ARRIVAL"
    DEPARTURE = "DEPARTURE"


class ScheduleType(Enum):
    FCFS = "FCFS"


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

    def __init__(self, id: int, arrival_time: float):
        self.id = id
        self.arrival_time = arrival_time
        self.service_start_time = None
        self.departure_time = None

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
        if self.is_queue_full():
            self.total_dropped += 1
            return

        # Only FCFS is supported in this simplified version.
        self.waiting_queue.append(client)

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
    
    def get_queue_samples(self) -> list[tuple[float, int]]:
        return self.sampled_queue_sizes
    
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
                       service_rate: float, arrival_rate: float):
        queue_system.total_arrivals += 1
        client = Client(id=event.client_id, arrival_time=event.time)

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

        # Schedule next arrival (single arrival process, increment client id by 1)
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
                 schedule_type: ScheduleType = ScheduleType.FCFS, num_starting_customers: int = 1, transient_detector: TransientDetection = None):
        self.queue_system = QueueSystem(num_servers, queue_capacity, schedule_type)
        self.fes = FutureEventSet()
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.sim_time = sim_time
        self.next_client_id = 1
        self.num_starting_customers = num_starting_customers
        self.transient_detector = transient_detector
        self.transient_end_time = None
        self.avg_queue_length_over_windows: list[tuple[float, float]] = []
        self.avg_sum_squared_diffs: list[tuple[float, float]] = []

    def event_loop(self):
        # Seed initial customers directly at time 0 (do not create multiple arrival streams)
        for _ in range(self.num_starting_customers):
            client = Client(id=self.next_client_id, arrival_time=0)
            # Count them as arrivals
            self.queue_system.total_arrivals += 1

            # If server available, put directly into service and schedule departure
            if self.queue_system.is_server_available():
                self.queue_system.allocate_server()
                client.service_start_time = 0.0
                self.queue_system.active_clients[client.id] = client
                self.queue_system.serving_clients[client.id] = client
                service_time = random.expovariate(self.service_rate)
                departure_event = Event(time=service_time,
                                        type=EventType.DEPARTURE, client_id=client.id)
                self.fes.schedule(departure_event)
            else:
                # Place in waiting queue
                self.queue_system.add_client_to_queue(client)

            self.next_client_id += 1

        # Schedule the first external arrival to start a single arrival stream
        inter_arrival_time = random.expovariate(self.arrival_rate)
        first_arrival_event = Event(time=inter_arrival_time,
                                    type=EventType.ARRIVAL, client_id=self.next_client_id)
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

            # Update transient detector
            if self.transient_detector and self.transient_end_time is None:
                self.transient_detector.add_value(event.time, len(self.queue_system.waiting_queue))
                next_interval_start = self.transient_detector.compute_next_interval_start()
                if event.time >= next_interval_start:
                    new_avg = self.transient_detector.compute_average()
                    self.transient_detector.add_average(event.time, new_avg)
                    self.avg_queue_length_over_windows.append((event.time, new_avg))
                    avg_sum_squared_diffs = self.transient_detector.get_avg_squared_diffs()
                    self.avg_sum_squared_diffs.append((event.time, avg_sum_squared_diffs))
                    if self.transient_detector.is_transient_over():
                        self.transient_end_time = self.transient_detector.get_transient_end_time()
                        print(f"Transient detected at time {self.transient_end_time}.")

            if event.type == EventType.ARRIVAL:
                EventHandler.handle_arrival(event, self.queue_system, self.fes, 
                                            self.service_rate, self.arrival_rate)
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
        avg_squared_diffs = [ssd for _, ssd in self.avg_sum_squared_diffs if ssd is not None]
        min_squared_diff = min(avg_squared_diffs) if avg_squared_diffs else None
        if min_squared_diff is not None:
            print(f"Minimum Average Sum of Squared Differences: {min_squared_diff}")

        self.plot_queue_size_over_time()

    def plot_queue_size_over_time(self):
        import matplotlib.pyplot as plt

        samples = self.queue_system.get_queue_samples()
        times = [t for t, _ in samples]
        sizes = [s for _, s in samples]

        plt.figure(figsize=(10, 5))
        plt.step(times, sizes, where='post')

        # Plot transient end time if detected
        if self.transient_end_time is not None:
            plt.axvline(x=self.transient_end_time, color='r', linestyle='--', label='Transient End Time')

        # Plot average queue lengths as horizontal lines spanning [t, t+window_size]
        if self.avg_queue_length_over_windows:
            # Ensure sorted by start time
            windows = sorted(self.avg_queue_length_over_windows, key=lambda x: x[0])

            # Try to get window_size from transient_detector, otherwise infer from consecutive starts
            window_size = 100.0  # Default value
            if self.transient_detector is not None:
                window_size = getattr(self.transient_detector, "window_size", None) or window_size

            if window_size is None and len(windows) > 1:
                window_size = windows[1][0] - windows[0][0]

            if window_size is None:
                window_size = 0.0

            for i, (t, avg) in enumerate(windows):
                x = [t, t + window_size]
                y = [avg, avg]
                label = 'Avg Queue Length Over Windows' if i == 0 else None
                plt.plot(x, y, color='green', linewidth=2, label=label)

        # Plot average sum of squared differences as a dotted line
        if self.avg_sum_squared_diffs:
            x = [t for t, _ in self.avg_sum_squared_diffs]
            y = [ssd for _, ssd in self.avg_sum_squared_diffs]
            plt.plot(x, y, color='orange', linestyle='--', label='Avg Sum Squared Diffs')

        # Plot also the minimum threshold line
        if self.transient_detector is not None:
            threshold = self.transient_detector.threshold
            plt.axhline(y=threshold, color='purple', linestyle=':', label='Threshold')

        # Show legend if any label was set (transient/end or avg windows)
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Queue Size')
        plt.title('Queue Size Over Time')
        plt.grid()
        plt.show()


# ===================================================
# MAIN EXECUTION
# ===================================================

if __name__ == "__main__":
    random.seed(123)

    # Simulation parameters
    SERVICE_RATE = 1  # Average of 1 time unit per service
    ARRIVAL_RATE = 1.2 * SERVICE_RATE  # Load factor of 1.2
    SIM_TIME = 20000.0     # Total simulation time
    SCHEDULING = ScheduleType.FCFS

    # Setups
    NUM_SERVERS = 1
    QUEUE_CAPACITY = 1000
    NUM_STARTING_CUSTOMERS = 1

    # Transient detection parameters
    WINDOW_SIZE = SIM_TIME // 100  # Duration of the sliding window T
    STRIDE = WINDOW_SIZE // 10       # Stride S
    NUM_INTERVALS = 50    # Number of intervals N
    THRESHOLD = .001 * WINDOW_SIZE     # Variation threshold P

    transient_detector = TransientDetection(WINDOW_SIZE, STRIDE, NUM_INTERVALS, THRESHOLD)

    # Initialize and run the simulator for each configuration
    print(f"\n--- Simulation with {NUM_SERVERS} servers, queue capacity {QUEUE_CAPACITY} and scheduling {SCHEDULING} ---")
    simulator = QueueSimulator(NUM_SERVERS, QUEUE_CAPACITY, 
                                ARRIVAL_RATE, SERVICE_RATE, SIM_TIME, schedule_type=SCHEDULING,
                                num_starting_customers=NUM_STARTING_CUSTOMERS, 
                                transient_detector=transient_detector)
    simulator.event_loop()
    simulator.print_statistics()

