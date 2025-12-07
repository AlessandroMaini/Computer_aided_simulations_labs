from enum import Enum
from typing import Optional
from queue import PriorityQueue
import random
import matplotlib.pyplot as plt
from rv_generation import RVGenerator
import numpy as np

# ==================================================
# INTERARRIVAL AND SERVICE DISTRIBUTIONS
# ==================================================

class DistributionType(Enum):
    EXPONENTIAL = "EXPONENTIAL"
    HYPEREXPONENTIAL = "HYPEREXPONENTIAL"
    ERLANG_K = "ERLANG-K"
    PARETO = "PARETO"


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

        # Interarrival and service time tracking for CV calculation
        self.interarrival_times: list[float] = []
        self.service_times: list[float] = []

    def update_time_weighted_stats(self, new_time: float):
        time_delta = new_time - self.last_event_time
        
        # Accumulate area under curves
        self.area_under_queue_curve += self.last_queue_size * time_delta
        self.area_under_system_curve += self.last_system_size * time_delta
        self.area_under_server_curve += self.server_busy * time_delta
        
        self.last_event_time = new_time

    def add_interarrival_time(self, time: float):
        self.interarrival_times.append(time)
    
    def add_service_time(self, time: float):
        self.service_times.append(time)

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
                       interarrival_type: DistributionType, service_type: DistributionType,
                       interarrival_params: dict = {}, service_params: dict = {}):
        queue_system.total_arrivals += 1
        client = Client(id=event.client_id, arrival_time=event.time)

        if queue_system.is_server_available():
            # Allocate server
            queue_system.allocate_server()
            client.service_start_time = event.time
            queue_system.active_clients[client.id] = client
            queue_system.serving_clients[client.id] = client

            # Schedule departure
            service_time = EventHandler._generate_new_time(service_type, service_params)
            queue_system.add_service_time(service_time)
            departure_event = Event(time=event.time + service_time, 
                                    type=EventType.DEPARTURE, client_id=client.id)
            fes.schedule(departure_event)

        else:
            # Add to waiting queue
            queue_system.add_client_to_queue(client)

        # Schedule next arrival (single arrival process, increment client id by 1)
        inter_arrival_time = EventHandler._generate_new_time(interarrival_type, interarrival_params)
        queue_system.add_interarrival_time(inter_arrival_time)
        next_arrival_event = Event(time=event.time + inter_arrival_time,
                                   type=EventType.ARRIVAL, client_id=event.client_id + 1)
        fes.schedule(next_arrival_event)
    

    @staticmethod
    def handle_departure(event: Event, queue_system: QueueSystem, fes: FutureEventSet, 
                         service_type: DistributionType, service_params: dict = {}):
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
            service_time = EventHandler._generate_new_time(service_type, service_params)
            queue_system.add_service_time(service_time)
            departure_event = Event(time=event.time + service_time, 
                                    type=EventType.DEPARTURE, client_id=next_client.id)
            fes.schedule(departure_event)

    @staticmethod
    def _generate_new_time(distribution_type: DistributionType, params: dict) -> float:
        if distribution_type == DistributionType.EXPONENTIAL:
            _lambda = params.get("lambda", 1.0)
            return np.random.exponential(1/_lambda, size=1)[0]
        elif distribution_type == DistributionType.HYPEREXPONENTIAL:
            lambdas = params.get("lambdas", [1.0])
            probabilities = params.get("probabilities", [1.0])
            return RVGenerator.hyperexponential_sample(lambdas, probabilities, 1)[0]
        elif distribution_type == DistributionType.ERLANG_K:
            _lambda = params.get("lambda", 1.0)
            k = params.get("k", 1)
            return RVGenerator.erlang_k_sample(_lambda, k, 1)[0]
        elif distribution_type == DistributionType.PARETO:
            alpha = params.get("alpha", 1.0)
            scale = params.get("scale", 1.0)
            return RVGenerator.pareto_sample(alpha, 1, scale=scale)[0]
        else:
            raise ValueError(f"Unsupported distribution type: {distribution_type}")


# ===================================================
# SIMULATION ENGINE
# ===================================================

class QueueSimulator:

    def __init__(self, num_servers: int, queue_capacity: int, sim_time: float, 
                 schedule_type: ScheduleType = ScheduleType.FCFS, num_starting_customers: int = 1,
                 interarrival_type: DistributionType = DistributionType.EXPONENTIAL, 
                 service_type: DistributionType = DistributionType.EXPONENTIAL,
                 interarrival_params: dict = {"lambda": 1.0}, service_params: dict = {"lambda": 1.0}):
        self.queue_system = QueueSystem(num_servers, queue_capacity, schedule_type)
        self.fes = FutureEventSet()
        self.sim_time = sim_time
        self.next_client_id = 1
        self.num_starting_customers = num_starting_customers
        self.interarrival_type = interarrival_type
        self.service_type = service_type
        self.interarrival_params = interarrival_params
        self.service_params = service_params

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
                service_time = EventHandler._generate_new_time(self.service_type, self.service_params)
                departure_event = Event(time=service_time,
                                        type=EventType.DEPARTURE, client_id=client.id)
                self.fes.schedule(departure_event)
            else:
                # Place in waiting queue
                self.queue_system.add_client_to_queue(client)

            self.next_client_id += 1

        # Schedule the first external arrival to start a single arrival stream
        inter_arrival_time = EventHandler._generate_new_time(self.interarrival_type, self.interarrival_params)
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
            self.queue_system.current_time = event.time

            if event.type == EventType.ARRIVAL:
                EventHandler.handle_arrival(event, self.queue_system, self.fes, 
                                            self.interarrival_type, self.service_type,
                                            self.interarrival_params, self.service_params)
            elif event.type == EventType.DEPARTURE:
                EventHandler.handle_departure(event, self.queue_system, self.fes, 
                                              self.service_type, self.service_params)

    def get_statistics(self):
        # Essential metrics only
        avg_queue_length = (self.queue_system.area_under_queue_curve / self.sim_time 
                            if self.sim_time > 0 else 0)
        
        waiting_times = [c.waiting_time() for c in self.queue_system.served_clients if c.waiting_time() is not None]
        avg_waiting_time = np.mean(waiting_times) if len(waiting_times) > 0 else 0
        
        std_interarrival = (np.std(self.queue_system.interarrival_times, ddof=1) 
                           if len(self.queue_system.interarrival_times) > 1 else 0)
        std_service = (np.std(self.queue_system.service_times, ddof=1) 
                      if len(self.queue_system.service_times) > 1 else 0)
        
        utilization = (self.queue_system.area_under_server_curve / (self.sim_time * self.queue_system.num_servers)
                      if self.sim_time > 0 else 0)
        
        return {
            "avg_queue_length": avg_queue_length,
            "avg_waiting_time": avg_waiting_time,
            "std_interarrival": std_interarrival,
            "std_service": std_service,
            "utilization": utilization
        }
    
    def print_statistics(self):
        stats = self.get_statistics()
        print(f"  Queue Length: {stats['avg_queue_length']:.4f}")
        print(f"  Waiting Time: {stats['avg_waiting_time']:.4f}")
        print(f"  Utilization:  {stats['utilization']:.4f}")
        print(f"  Std Inter:    {stats['std_interarrival']:.4f}")
        print(f"  Std Service:  {stats['std_service']:.4f}")




# ===================================================
# TESTING FRAMEWORK WITH MULTIPLE RUNS
# ===================================================

def run_single_test(config, sim_time, num_servers, queue_capacity, seed):
    """Run a single simulation and return statistics"""
    np.random.seed(seed)
    random.seed(seed)
    
    simulator = QueueSimulator(
        num_servers, queue_capacity, sim_time,
        schedule_type=ScheduleType.FCFS,
        num_starting_customers=0,
        interarrival_type=config["interarrival_type"],
        interarrival_params=config["interarrival_params"],
        service_type=config["service_type"],
        service_params=config["service_params"]
    )
    
    simulator.event_loop()
    return simulator.get_statistics()


def run_comparative_analysis():
    """Run tests with multiple replications for statistical consistency"""
    
    # Common parameters
    SIM_TIME = 50000.0
    NUM_SERVERS = 1
    QUEUE_CAPACITY = 1000
    NUM_RUNS = 10  # Multiple runs for averaging
    BASE_SEED = 42
    
    # Target mean times
    MEAN_INTERARRIVAL = 1.0
    MEAN_SERVICE = 0.8  # rho = 0.8
    
    # Define test configurations - SIMPLIFIED
    test_configs = [
        {
            "name": "Exponential/Exponential",
            "interarrival_type": DistributionType.EXPONENTIAL,
            "interarrival_params": {"lambda": 1.0/MEAN_INTERARRIVAL},
            "service_type": DistributionType.EXPONENTIAL,
            "service_params": {"lambda": 1.0/MEAN_SERVICE}
        },
        {
            "name": "Erlang-3/Exponential",
            "interarrival_type": DistributionType.ERLANG_K,
            "interarrival_params": {"lambda": 3.0/MEAN_INTERARRIVAL, "k": 3},
            "service_type": DistributionType.EXPONENTIAL,
            "service_params": {"lambda": 1.0/MEAN_SERVICE}
        },
        {
            "name": "Hyperexp/Exponential",
            "interarrival_type": DistributionType.HYPEREXPONENTIAL,
            "interarrival_params": {"lambdas": [2.0, 0.666], "probabilities": [0.5, 0.5]},
            "service_type": DistributionType.EXPONENTIAL,
            "service_params": {"lambda": 1.0/MEAN_SERVICE}
        },
        {
            "name": "Pareto/Exponential",
            "interarrival_type": DistributionType.PARETO,
            "interarrival_params": {"alpha": 1.5, "scale": 0.3333},
            "service_type": DistributionType.EXPONENTIAL,
            "service_params": {"lambda": 1.0/MEAN_SERVICE}
        },
        {
            "name": "Exponential/Erlang-4",
            "interarrival_type": DistributionType.EXPONENTIAL,
            "interarrival_params": {"lambda": 1.0/MEAN_INTERARRIVAL},
            "service_type": DistributionType.ERLANG_K,
            "service_params": {"lambda": 4.0/MEAN_SERVICE, "k": 4}
        },
        {
            "name": "Erlang-3/Erlang-4",
            "interarrival_type": DistributionType.ERLANG_K,
            "interarrival_params": {"lambda": 3.0/MEAN_INTERARRIVAL, "k": 3},
            "service_type": DistributionType.ERLANG_K,
            "service_params": {"lambda": 4.0/MEAN_SERVICE, "k": 4}
        },
        {
            "name": "Exponential/Hyperexp",
            "interarrival_type": DistributionType.EXPONENTIAL,
            "interarrival_params": {"lambda": 1.0/MEAN_INTERARRIVAL},
            "service_type": DistributionType.HYPEREXPONENTIAL,
            "service_params": {"lambdas": [2, 0.909], "probabilities": [0.5, 0.5]}
        },
        {
            "name": "Hyperexp/Hyperexp",
            "interarrival_type": DistributionType.HYPEREXPONENTIAL,
            "interarrival_params": {"lambdas": [2.0, 0.666], "probabilities": [0.5, 0.5]},
            "service_type": DistributionType.HYPEREXPONENTIAL,
            "service_params": {"lambdas": [2, 0.909], "probabilities": [0.5, 0.5]}
        },
        {
            "name": "Exponential/Pareto",
            "interarrival_type": DistributionType.EXPONENTIAL,
            "interarrival_params": {"lambda": 1.0/MEAN_INTERARRIVAL},
            "service_type": DistributionType.PARETO,
            "service_params": {"alpha": 1.714, "scale": 0.3333}
        },
        {
            "name": "Pareto/Pareto",
            "interarrival_type": DistributionType.PARETO,
            "interarrival_params": {"alpha": 1.5, "scale": 0.3333},
            "service_type": DistributionType.PARETO,
            "service_params": {"alpha": 1.714, "scale": 0.3333}
        }
    ]
    
    # Store averaged results
    all_results = []
    
    print("\n" + "="*80)
    print(f"RUNNING COMPARATIVE ANALYSIS ({NUM_RUNS} runs per configuration)")
    print("="*80)
    
    # Run each configuration multiple times
    for i, config in enumerate(test_configs, 1):
        print(f"\n[{i}/{len(test_configs)}] {config['name']}")
        
        run_stats = []
        for run in range(NUM_RUNS):
            stats = run_single_test(config, SIM_TIME, NUM_SERVERS, QUEUE_CAPACITY, BASE_SEED + run)
            run_stats.append(stats)
            print(f"  Run {run+1}/{NUM_RUNS}: Q={stats['avg_queue_length']:.2f}, W={stats['avg_waiting_time']:.2f}")
        
        # Average across runs
        avg_stats = {
            "name": config["name"],
            "avg_queue_length": np.mean([s['avg_queue_length'] for s in run_stats]),
            "avg_waiting_time": np.mean([s['avg_waiting_time'] for s in run_stats]),
            "std_interarrival": np.mean([s['std_interarrival'] for s in run_stats]),
            "std_service": np.mean([s['std_service'] for s in run_stats]),
            "utilization": np.mean([s['utilization'] for s in run_stats])
        }
        all_results.append(avg_stats)
        
        print(f"  → Average: Q={avg_stats['avg_queue_length']:.4f}, W={avg_stats['avg_waiting_time']:.4f}")
    
    # Create simplified comparison plots
    create_comparison_plots(all_results)
    
    # Print comparison table
    print_comparison_table(all_results)


def create_comparison_plots(results):
    """Create simplified comparison plots focusing on key metrics"""
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Impact of Distribution Variability on Queue Performance (Same Mean)', 
                fontsize=14, fontweight='bold')
    
    names = [r['name'] for r in results]
    x_pos = np.arange(len(names))
    
    # 1. Average Queue Length (KEY METRIC)
    ax1 = axes[0]
    queue_lengths = [r['avg_queue_length'] for r in results]
    bars = ax1.bar(x_pos, queue_lengths, alpha=0.7, edgecolor='black')
    # Color by queue length value (normalized)
    norm_queue = np.array(queue_lengths)
    norm_queue = (norm_queue - norm_queue.min()) / (norm_queue.max() - norm_queue.min()) if norm_queue.max() > norm_queue.min() else np.zeros_like(norm_queue)
    colors_queue = plt.cm.RdYlGn_r(norm_queue)
    for bar, color in zip(bars, colors_queue):
        bar.set_color(color)
    ax1.set_ylabel('Average Queue Length', fontsize=11)
    ax1.set_title('Average Queue Length vs Distribution Type', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Average Waiting Time (KEY METRIC)
    ax2 = axes[1]
    waiting_times = [r['avg_waiting_time'] for r in results]
    bars = ax2.bar(x_pos, waiting_times, alpha=0.7, edgecolor='black')
    # Color by waiting time value (normalized)
    norm_wait = np.array(waiting_times)
    norm_wait = (norm_wait - norm_wait.min()) / (norm_wait.max() - norm_wait.min()) if norm_wait.max() > norm_wait.min() else np.zeros_like(norm_wait)
    colors_wait = plt.cm.RdYlGn_r(norm_wait)
    for bar, color in zip(bars, colors_wait):
        bar.set_color(color)
    ax2.set_ylabel('Average Waiting Time', fontsize=11)
    ax2.set_title('Average Waiting Time vs Distribution Type', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig("comparative_queue_performance.png", dpi=300, bbox_inches='tight')
    plt.show()


def print_comparison_table(results):
    """Print simplified comparison table"""
    print("\n" + "="*115)
    print("COMPARATIVE RESULTS SUMMARY (Averaged over 10 runs)")
    print("="*115)
    print(f"{'Configuration':<30} {'Std Inter':<15} {'Std Service':<15} {'Queue Length':<15} {'Waiting Time':<15} {'Util':<10}")
    print("-"*115)
    
    for r in results:
        print(f"{r['name']:<30} "
              f"{r['std_interarrival']:<15.4f} "
              f"{r['std_service']:<15.4f} "
              f"{r['avg_queue_length']:<15.4f} "
              f"{r['avg_waiting_time']:<15.4f} "
              f"{r['utilization']:<10.4f}")


# ===================================================
# MAIN EXECUTION
# ===================================================

if __name__ == "__main__":
    run_comparative_analysis()

