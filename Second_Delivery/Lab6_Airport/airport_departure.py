# Module imports
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from queue import PriorityQueue
from typing import Optional
import sys
import os

# Import ConfidenceInterval class
sys.path.append(os.path.dirname(__file__))
from confidence_interval import ConfidenceInterval

# --------------------------------------------------
# EVENT DEFINITION
# --------------------------------------------------
class EventType(Enum):
    FLIGHT_SCHEDULED = 0
    CUSTOMER_ARRIVAL = 1
    ENTER_CASHIER_QUEUE = 2
    EXIT_CASHIER_QUEUE = 3
    ENTER_SECURITY_QUEUE = 4
    EXIT_SECURITY_QUEUE = 5
    ENTER_BOARDING_QUEUE = 6
    EXIT_BOARDING_QUEUE = 7
    FLIGHT_DEPARTURE = 8

class Event:
    """Discrete event with type, time, and entity identifiers."""
    def __init__(self, event_type: EventType, time: float, customer_id: Optional[int] = None, flight_id: Optional[int] = None):
        self.event_type = event_type
        self.time = time
        self.customer_id = customer_id
        self.flight_id = flight_id
        self.cancelled = False

    def cancel(self):
        """Mark event as cancelled."""
        self.cancelled = True

    def is_cancelled(self) -> bool:
        """Check if event is cancelled."""
        return self.cancelled
    
    def __lt__(self, other) -> bool:
        """Compare events by time for priority queue."""
        return self.time < other.time

# ------------------------------------------------------
# PASSENGER
# ------------------------------------------------------
class PriorityClass(Enum):
    ECONOMY = 3
    BUSINESS = 2
    FIRST_CLASS = 1

class PassengerState(Enum):
    ARRIVED = 1
    IN_CASHIER_LAND = 2
    IN_SECURITY = 3
    IN_AIRSIDE = 4
    IN_CASHIER_AIR = 5
    IN_BOARDING = 6
    LEFT_AIRPORT = 7

class Passenger:
    """Passenger entity with ID, arrival time, priority class, and number of companions."""
    def __init__(self, passenger_id: int, arrival_time: float, priority_class: PriorityClass, flight_id: int, num_companions: int = 0):
        self.passenger_id = passenger_id
        self.arrival_time = arrival_time
        self.priority_class = priority_class
        self.flight_id = flight_id
        self.num_companions = num_companions
        self.state = PassengerState.ARRIVED
        self.checkin_time: Optional[float] = None
        self.cashier_queue_entry_time: Optional[float] = None
        self.cashier_entry_time: Optional[float] = None
        self.cashier_exit_time: Optional[float] = None
        self.security_queue_entry_time: Optional[float] = None
        self.security_entry_time: Optional[float] = None
        self.security_exit_time: Optional[float] = None
        self.boarding_queue_entry_time: Optional[float] = None
        self.boarding_entry_time: Optional[float] = None
        self.departure_time: Optional[float] = None

    def enter_cashier_queue(self, time: float):
        self.cashier_queue_entry_time = time
        if self.state == PassengerState.ARRIVED:
            self.state = PassengerState.IN_CASHIER_LAND
        elif self.state == PassengerState.IN_AIRSIDE:
            self.state = PassengerState.IN_CASHIER_AIR

    def enter_cashier(self, time: float):
        self.cashier_entry_time = time

    def exit_cashier(self, time: float):
        self.cashier_exit_time = time

    def enter_security_queue(self, time: float):
        self.security_queue_entry_time = time
        self.state = PassengerState.IN_SECURITY

    def enter_security(self, time: float):
        self.security_entry_time = time
        # Companions leave the airport
        self.num_companions = 0

    def exit_security(self, time: float):
        self.security_exit_time = time
        self.state = PassengerState.IN_AIRSIDE

    def enter_boarding_queue(self, time: float):
        self.boarding_queue_entry_time = time
        self.state = PassengerState.IN_BOARDING

    def enter_boarding(self, time: float):
        self.boarding_entry_time = time

    def depart(self, time: float):
        self.departure_time = time
        self.state = PassengerState.LEFT_AIRPORT

    def get_landside_dwell_time(self) -> Optional[float]:
        if self.security_entry_time is not None:
            return self.security_entry_time - self.arrival_time
        return None
    
    def get_airside_dwell_time(self) -> Optional[float]:
        if self.departure_time is not None and self.security_exit_time is not None:
            return self.departure_time - self.security_exit_time
        return None
    
    def get_in_cashier_queue_time(self) -> Optional[float]:
        if self.cashier_entry_time is not None and self.cashier_queue_entry_time is not None:
            return self.cashier_entry_time - self.cashier_queue_entry_time
        return None
    
    def get_in_security_queue_time(self) -> Optional[float]:
        if self.security_entry_time is not None and self.security_queue_entry_time is not None:
            return self.security_entry_time - self.security_queue_entry_time
        return None
    
    def get_in_boarding_queue_time(self) -> Optional[float]:
        if self.boarding_entry_time is not None and self.boarding_queue_entry_time is not None:
            return self.boarding_entry_time - self.boarding_queue_entry_time
        return None
    
    def get_in_cashier_time(self) -> Optional[float]:
        if self.cashier_exit_time is not None and self.cashier_entry_time is not None:
            return self.cashier_exit_time - self.cashier_entry_time
        return None
    
    def get_in_security_time(self) -> Optional[float]:
        if self.security_exit_time is not None and self.security_entry_time is not None:
            return self.security_exit_time - self.security_entry_time
        return None
    
    def get_in_boarding_time(self) -> Optional[float]:
        if self.departure_time is not None and self.boarding_entry_time is not None:
            return self.departure_time - self.boarding_entry_time
        return None

# ------------------------------------------------------
# FLIGHT
# ------------------------------------------------------
class Flight:
    """Flight entity with ID and take-off time."""
    def __init__(self, flight_id: int, takeoff_time: float, num_passengers: int, boarding_duration: float):
        self.flight_id = flight_id
        self.takeoff_time = takeoff_time
        self.num_passengers = num_passengers
        self.passengers: dict[int, Passenger] = {}
        self.served_passengers: int = 0
        self.boarding_start_time: float = takeoff_time - boarding_duration

    def add_passenger(self, passenger: Passenger):
        self.passengers[passenger.passenger_id] = passenger

    def get_drop_percentage(self) -> float:
        return (self.num_passengers - self.served_passengers) / self.num_passengers * 100 if self.num_passengers > 0 else 0.0

# ------------------------------------------------------
# FUTURE EVENT SET
# ------------------------------------------------------
class FutureEventSet:
    """Priority queue of events ordered by time."""
    def __init__(self):
        self.events: PriorityQueue[Event] = PriorityQueue()
        self.event_count: int = 0

    def schedule(self, event: Event):
        """Schedule event in priority queue."""
        self.events.put(event)
        self.event_count += 1

    def get_next_event(self) -> Optional[Event]:
        """Get next non-cancelled event."""
        if not self.is_empty():
            while not self.events.empty():
                event = self.events.get()
                if not event.is_cancelled():
                    return event
        return None
    
    def is_empty(self) -> bool:
        """Check if event queue is empty."""
        return self.events.empty()

# --------------------------------------------------
# METRICS COLLECTION
# --------------------------------------------------
class Metrics:
    """Class to collect and store simulation metrics."""
    def __init__(self, num_cashier_land: int = 1, num_cashier_air: int = 1, num_security: int = 1, num_boarding: int = 1):
        self.num_cashier_land = num_cashier_land
        self.num_cashier_air = num_cashier_air
        self.num_security = num_security
        self.num_boarding = num_boarding
        self.total_customer_arrivals: int = 0
        self.total_passengers_arrivals: int = 0
        self.total_flights: int = 0
        self.total_landside_dwell_time: float = 0.0
        self.total_airside_dwell_time: float = 0.0
        self.total_cashier_land_wait_time: float = 0.0
        self.total_cashier_air_wait_time: float = 0.0
        self.total_security_wait_time: float = 0.0
        self.total_boarding_wait_time: float = 0.0
        self.cashier_land_utilization_time: float = 0.0
        self.cashier_air_utilization_time: float = 0.0
        self.security_utilization_time: float = 0.0
        self.boarding_utilization_per_flight: dict[int, float] = {}  # Per-flight boarding utilization
        self.dropped_passengers: int = 0

        # Per-priority metrics
        self.security_wait_time_per_priority: dict[PriorityClass, list[float]] = {pc: [] for pc in PriorityClass}
        self.boarding_wait_time_per_priority: dict[PriorityClass, list[float]] = {pc: [] for pc in PriorityClass}
        self.total_time_per_priority: dict[PriorityClass, list[float]] = {pc: [] for pc in PriorityClass}

        # Time-weighted metrics
        self.last_event_time: float = 0.0
        self.area_under_cashier_land_queue: float = 0.0
        self.area_under_cashier_air_queue: float = 0.0
        self.area_under_security_queue: float = 0.0
        self.area_under_boarding_queue: float = 0.0
        self.area_under_airport_customers: float = 0.0
        self.area_under_landside_customers: float = 0.0
        self.area_under_airside_customers: float = 0.0

        # Sample statistics
        self.samples_number_in_airport: dict[float, int] = {}
        self.samples_number_in_landside: dict[float, int] = {}
        self.samples_number_in_airside: dict[float, int] = {}
        self.samples_number_in_security: dict[float, int] = {}
        self.samples_number_in_boarding: dict[float, int] = {}

        # Customer and flight tracking
        self.passengers: dict[int, Passenger] = {}
        self.flights: dict[int, Flight] = {}

    def add_passenger(self, passenger: Passenger):
        self.passengers[passenger.passenger_id] = passenger

    def remove_passenger(self, passenger: Passenger):
        self.passengers.pop(passenger.passenger_id, None)

    def add_flight(self, flight: Flight):
        self.flights[flight.flight_id] = flight
        self.total_flights += 1

    def remove_flight(self, flight: Flight):
        self.flights.pop(flight.flight_id, None)

    def update_time_weighted_metrics(self, current_time: float, num_in_airport: int, num_in_landside: int, 
                                     num_in_airside: int, num_in_security_queue: int, num_in_boarding_queue: int,
                                     num_in_cashier_land_queue: int, num_in_cashier_air_queue: int):
        time_delta = current_time - self.last_event_time
        self.area_under_airport_customers += num_in_airport * time_delta
        self.area_under_landside_customers += num_in_landside * time_delta
        self.area_under_airside_customers += num_in_airside * time_delta
        self.area_under_security_queue += num_in_security_queue * time_delta
        self.area_under_boarding_queue += num_in_boarding_queue * time_delta
        self.area_under_cashier_land_queue += num_in_cashier_land_queue * time_delta
        self.area_under_cashier_air_queue += num_in_cashier_air_queue * time_delta
        self.last_event_time = current_time

    def sample_statistics(self, current_time: float, num_in_airport: int, num_in_landside: int, 
                         num_in_airside: int, num_in_security: int, num_in_boarding: int):
        self.samples_number_in_airport[current_time] = num_in_airport
        self.samples_number_in_landside[current_time] = num_in_landside
        self.samples_number_in_airside[current_time] = num_in_airside
        self.samples_number_in_security[current_time] = num_in_security
        self.samples_number_in_boarding[current_time] = num_in_boarding

    def add_group_arrival(self, group_size: int):
        self.total_customer_arrivals += group_size + 1 # Including the main passenger
        self.total_passengers_arrivals += 1 # Each group contains one passenger

    def add_landside_dwell_time(self, dwell_time: float):
        self.total_landside_dwell_time += dwell_time

    def add_airside_dwell_time(self, dwell_time: float):
        self.total_airside_dwell_time += dwell_time

    def add_cashier_land_wait_time(self, wait_time: float):
        self.total_cashier_land_wait_time += wait_time

    def add_cashier_air_wait_time(self, wait_time: float):
        self.total_cashier_air_wait_time += wait_time

    def add_security_wait_time(self, wait_time: float, priority_class: PriorityClass = None):
        self.total_security_wait_time += wait_time
        if priority_class:
            self.security_wait_time_per_priority[priority_class].append(wait_time)

    def add_boarding_wait_time(self, wait_time: float, priority_class: PriorityClass = None):
        self.total_boarding_wait_time += wait_time
        if priority_class:
            self.boarding_wait_time_per_priority[priority_class].append(wait_time)

    def add_cashier_land_utilization_time(self, utilization_time: float):
        self.cashier_land_utilization_time += utilization_time
    
    def add_cashier_air_utilization_time(self, utilization_time: float):
        self.cashier_air_utilization_time += utilization_time

    def add_security_utilization_time(self, utilization_time: float):
        self.security_utilization_time += utilization_time

    def add_boarding_utilization_time(self, utilization_time: float, flight_id: int):
        if flight_id not in self.boarding_utilization_per_flight:
            self.boarding_utilization_per_flight[flight_id] = 0.0
        self.boarding_utilization_per_flight[flight_id] += utilization_time
    
    def add_total_time_per_priority(self, total_time: float, priority_class: PriorityClass):
        """Record total time from arrival to departure for a passenger."""
        self.total_time_per_priority[priority_class].append(total_time)
    
    def get_average_number_in_airport(self, total_simulation_time: float) -> float:
        return self.area_under_airport_customers / total_simulation_time if total_simulation_time > 0 else 0.0
    
    def get_average_number_in_landside(self, total_simulation_time: float) -> float:
        return self.area_under_landside_customers / total_simulation_time if total_simulation_time > 0 else 0.0
    
    def get_average_number_in_airside(self, total_simulation_time: float) -> float:
        return self.area_under_airside_customers / total_simulation_time if total_simulation_time > 0 else 0.0
    
    def get_average_cashier_land_queue_length(self, total_simulation_time: float) -> float:
        return self.area_under_cashier_land_queue / total_simulation_time if total_simulation_time > 0 else 0.0

    def get_average_cashier_air_queue_length(self, total_simulation_time: float) -> float:
        return self.area_under_cashier_air_queue / total_simulation_time if total_simulation_time > 0 else 0.0
    
    def get_average_security_queue_length(self, total_simulation_time: float) -> float:
        return self.area_under_security_queue / total_simulation_time if total_simulation_time > 0 else 0.0
    
    def get_average_boarding_queue_length(self, total_simulation_time: float) -> float:
        return self.area_under_boarding_queue / total_simulation_time if total_simulation_time > 0 else 0.0
    
    def get_average_landside_dwell_time(self) -> float:
        return self.total_landside_dwell_time / self.total_customer_arrivals if self.total_customer_arrivals > 0 else 0.0
    
    def get_average_airside_dwell_time(self) -> float:
        return self.total_airside_dwell_time / self.total_passengers_arrivals if self.total_passengers_arrivals > 0 else 0.0
    
    def get_average_cashier_land_wait_time(self) -> float:
        return self.total_cashier_land_wait_time / self.total_passengers_arrivals if self.total_passengers_arrivals > 0 else 0.0

    def get_average_cashier_air_wait_time(self) -> float:
        return self.total_cashier_air_wait_time / self.total_passengers_arrivals if self.total_passengers_arrivals > 0 else 0.0
    
    def get_average_security_wait_time(self) -> float:
        return self.total_security_wait_time / self.total_passengers_arrivals if self.total_passengers_arrivals > 0 else 0.0
    
    def get_average_boarding_wait_time(self) -> float:
        return self.total_boarding_wait_time / self.total_passengers_arrivals if self.total_passengers_arrivals > 0 else 0.0
    
    def get_cashier_land_utilization(self, total_simulation_time: float) -> float:
        total_capacity = total_simulation_time * self.num_cashier_land
        return self.cashier_land_utilization_time / total_capacity if total_capacity > 0 else 0.0

    def get_cashier_air_utilization(self, total_simulation_time: float) -> float:
        total_capacity = total_simulation_time * self.num_cashier_air
        return self.cashier_air_utilization_time / total_capacity if total_capacity > 0 else 0.0
    
    def get_security_utilization(self, total_simulation_time: float) -> float:
        total_capacity = total_simulation_time * self.num_security
        return self.security_utilization_time / total_capacity if total_capacity > 0 else 0.0
    
    def get_boarding_utilization(self, boarding_window_duration: float = 45.0) -> float:
        """Calculate average boarding utilization across all flights.
        Each flight has its own boarding window (default 45 minutes) with num_boarding servers."""
        if not self.boarding_utilization_per_flight:
            return 0.0
        
        # Calculate utilization for each flight and return average
        utilizations = []
        for flight_id, utilization_time in self.boarding_utilization_per_flight.items():
            # Each flight has boarding_window_duration minutes with num_boarding servers
            flight_capacity = boarding_window_duration * self.num_boarding
            flight_utilization = utilization_time / flight_capacity if flight_capacity > 0 else 0.0
            utilizations.append(flight_utilization)
        
        return sum(utilizations) / len(utilizations) if utilizations else 0.0
    
    def get_average_passengers_per_flight(self) -> float:
        return self.total_passengers_arrivals / self.total_flights if self.total_flights > 0 else 0.0
    
    def get_drop_percentage(self) -> float:
        return (self.dropped_passengers / self.total_passengers_arrivals * 100) if self.total_passengers_arrivals > 0 else 0.0
    
    def get_average_security_wait_per_priority(self, priority_class: PriorityClass) -> float:
        """Get average security wait time for a specific priority class."""
        times = self.security_wait_time_per_priority[priority_class]
        return sum(times) / len(times) if times else 0.0
    
    def get_average_boarding_wait_per_priority(self, priority_class: PriorityClass) -> float:
        """Get average boarding wait time for a specific priority class."""
        times = self.boarding_wait_time_per_priority[priority_class]
        return sum(times) / len(times) if times else 0.0
    
    def get_average_total_time_per_priority(self, priority_class: PriorityClass) -> float:
        """Get average total time (arrival to departure) for a specific priority class."""
        times = self.total_time_per_priority[priority_class]
        return sum(times) / len(times) if times else 0.0

# ------------------------------------------------------
# QUEUE SYSTEM
# ------------------------------------------------------
class QueueSystem:
    """Class to manage queues and servers in the airport lounge."""
    def __init__(self, num_servers: int, priority: bool = False):
        self.num_servers = num_servers
        self.priority: bool = priority
        
        # Initial state
        self.server_busy = 0
        self.waiting_queue: list[Passenger] = []
        self.last_queue_size = 0
        self.last_system_size = 0

        # Client tracking
        self.active_clients: dict[int, Passenger] = {}
        self.served_clients: list[Passenger] = []
        self.serving_clients: dict[int, Passenger] = {}

    def is_server_available(self) -> bool:
        return self.server_busy < self.num_servers
    
    def allocate_server(self):
        self.server_busy += 1
        self.last_system_size += 1

    def release_server(self):
        self.server_busy -= 1
        self.last_system_size -= 1

    def add_client_to_queue(self, client: Passenger):
        if self.priority:
            index = 0
            while index < len(self.waiting_queue) and self.waiting_queue[index].priority_class.value <= client.priority_class.value:
                index += 1
            self.waiting_queue.insert(index, client)
        else:
            self.waiting_queue.append(client)
        self.active_clients[client.passenger_id] = client
        self.last_queue_size += 1
        self.last_system_size += 1

    def remove_client_from_queue(self) -> Optional[Passenger]:
        if self.waiting_queue:
            client = self.waiting_queue.pop(0)
            self.last_queue_size -= 1
            self.last_system_size -= 1
            return client
        return None

# ------------------------------------------------------
# EVENT HANDLERS
# ------------------------------------------------------
class EventHandlers:
    """Class to handle events in the simulation."""
    
    @staticmethod
    def handle_flight_scheduled(event: Event, metrics: Metrics, future_event_set: FutureEventSet, next_passenger_id: int,
                                avg_companions: float, priority_probs: list[float],
                                avg_arrival_time_before_flight: float, stddev_arrival_time_before_flight: float) -> int:
        """Receives a scheduled flight and initializes passengers for the flight."""
        flight: Flight = metrics.flights[event.flight_id]
        takeoff_time = flight.takeoff_time
        
        for _ in range(flight.num_passengers):
            # Generate arrival time with positive-defined normal distribution
            time_before_flight = abs(np.random.normal(avg_arrival_time_before_flight, stddev_arrival_time_before_flight))
            arrival_time = max(0.0, takeoff_time - time_before_flight)  # Ensure non-negative arrival time
            priority_class = np.random.choice(list(PriorityClass), p=priority_probs)
            num_companions = np.random.poisson(avg_companions)
            passenger = Passenger(passenger_id=next_passenger_id, arrival_time=arrival_time, priority_class=priority_class, 
                                  flight_id=flight.flight_id, num_companions=num_companions)
            next_passenger_id += 1
            flight.add_passenger(passenger)
            
            arrival_event = Event(EventType.CUSTOMER_ARRIVAL, arrival_time, customer_id=passenger.passenger_id, flight_id=flight.flight_id)
            future_event_set.schedule(arrival_event)
        
        # Schedule flight departure event
        departure_event = Event(EventType.FLIGHT_DEPARTURE, takeoff_time, flight_id=flight.flight_id)
        future_event_set.schedule(departure_event)
        
        return next_passenger_id
    
    @staticmethod
    def handle_customer_arrival(event: Event, metrics: Metrics, future_event_set: FutureEventSet, buying_prob: float,
                            avg_landside_dwell_time: float, security_deadline: float, cashier_service_rate: float):
        """Handles customer arrival event."""
        if event.customer_id not in metrics.passengers:
            return
        passenger: Passenger = metrics.passengers[event.customer_id]
        flight: Flight = metrics.flights[passenger.flight_id]
        metrics.add_group_arrival(passenger.num_companions)
        
        buys: bool = np.random.rand() < buying_prob
        landside_dwell_time = np.random.exponential(avg_landside_dwell_time)
        security_cutoff_time = flight.takeoff_time - security_deadline
        expected_total_time = event.time + landside_dwell_time
        
        if buys:
            expected_cashier_time = cashier_service_rate * (1 + passenger.num_companions) * 2
            expected_total_time += expected_cashier_time
        
        # If not enough time, skip buying and cap dwell time
        if expected_total_time >= security_cutoff_time:
            buys = False
            landside_dwell_time = max(0, min(security_cutoff_time - event.time, landside_dwell_time))
        
        metrics.add_landside_dwell_time(landside_dwell_time * (1 + passenger.num_companions))
        
        # Schedule next event based on buying decision
        next_event_type = EventType.ENTER_CASHIER_QUEUE if buys else EventType.ENTER_SECURITY_QUEUE
        next_event = Event(next_event_type, event.time + landside_dwell_time, customer_id=passenger.passenger_id)
        future_event_set.schedule(next_event)
    
    @staticmethod
    def handle_enter_cashier_queue(event: Event, metrics: Metrics, future_event_set: FutureEventSet, cashier_queue: QueueSystem,
                                   service_rate: float):
        """Handles customer entering cashier queue."""
        if event.customer_id not in metrics.passengers:
            return
        passenger: Passenger = metrics.passengers[event.customer_id]
        passenger.enter_cashier_queue(event.time)
        cashier_queue.add_client_to_queue(passenger)
        
        if cashier_queue.is_server_available():
            next_passenger = cashier_queue.remove_client_from_queue()
            next_passenger.enter_cashier(event.time)
            cashier_queue.allocate_server()
            cashier_queue.serving_clients[next_passenger.passenger_id] = next_passenger
            # Service time scales with number of people (passenger + companions)
            service_time = np.random.exponential(service_rate) * (1 + next_passenger.num_companions)
            exit_cashier_event = Event(EventType.EXIT_CASHIER_QUEUE, event.time + service_time, customer_id=next_passenger.passenger_id)
            future_event_set.schedule(exit_cashier_event)
    
    @staticmethod
    def handle_exit_cashier_queue(event: Event, metrics: Metrics, future_event_set: FutureEventSet, cashier_queue: QueueSystem, side: str,
                                  service_rate: float):
        """Handles customer exiting cashier queue."""
        if event.customer_id not in metrics.passengers:
            return
        passenger: Passenger = metrics.passengers[event.customer_id]
        passenger.exit_cashier(event.time)
        if side == "landside":
            wait_time = passenger.get_in_cashier_queue_time()
            if wait_time is not None:
                metrics.add_cashier_land_wait_time(wait_time * (1 + passenger.num_companions))
            utilization_time = passenger.get_in_cashier_time()
            if utilization_time is not None:
                metrics.add_cashier_land_utilization_time(utilization_time)
        elif side == "airside":
            wait_time = passenger.get_in_cashier_queue_time()
            if wait_time is not None:
                metrics.add_cashier_air_wait_time(wait_time)
            utilization_time = passenger.get_in_cashier_time()
            if utilization_time is not None:
                metrics.add_cashier_air_utilization_time(utilization_time)
        else:
            raise ValueError("Side must be either 'landside' or 'airside'")
        
        if side == "landside":
            # Move to security queue
            enter_security_event = Event(EventType.ENTER_SECURITY_QUEUE, event.time, customer_id=passenger.passenger_id)
            future_event_set.schedule(enter_security_event)
        elif side == "airside":
            # Move to boarding queue
            boarding_time = max(metrics.flights[passenger.flight_id].boarding_start_time, event.time)
            enter_boarding_event = Event(EventType.ENTER_BOARDING_QUEUE, boarding_time, customer_id=passenger.passenger_id,
                                         flight_id=passenger.flight_id)
            future_event_set.schedule(enter_boarding_event)
        else:
            raise ValueError("Side must be either 'landside' or 'airside'")
        
        # Free up cashier server
        cashier_queue.release_server()
        cashier_queue.serving_clients.pop(passenger.passenger_id, None)
        
        # Check if there are more clients in the cashier queue
        next_passenger = cashier_queue.remove_client_from_queue()
        if not next_passenger:
            return
        next_passenger.enter_cashier(event.time)
        cashier_queue.allocate_server()
        cashier_queue.serving_clients[next_passenger.passenger_id] = next_passenger
        # Service time scales with number of people (passenger + companions)
        service_time = np.random.exponential(service_rate) * (1 + next_passenger.num_companions)
        exit_cashier_event = Event(EventType.EXIT_CASHIER_QUEUE, event.time + service_time, customer_id=next_passenger.passenger_id)
        future_event_set.schedule(exit_cashier_event)

    @staticmethod
    def handle_enter_security_queue(event: Event, metrics: Metrics, future_event_set: FutureEventSet, security_queue: QueueSystem,
                                    service_rate: float):
        """Handles customer entering security queue."""
        if event.customer_id not in metrics.passengers:
            return
        passenger: Passenger = metrics.passengers[event.customer_id]
        passenger.enter_security_queue(event.time)
        security_queue.add_client_to_queue(passenger)
        
        if security_queue.is_server_available():
            next_passenger = security_queue.remove_client_from_queue()
            next_passenger.enter_security(event.time)
            security_queue.allocate_server()
            security_queue.serving_clients[next_passenger.passenger_id] = next_passenger
            service_time = np.random.exponential(service_rate)
            exit_security_event = Event(EventType.EXIT_SECURITY_QUEUE, event.time + service_time, customer_id=next_passenger.passenger_id)
            future_event_set.schedule(exit_security_event)

    @staticmethod
    def handle_exit_security_queue(event: Event, metrics: Metrics, future_event_set: FutureEventSet, security_queue: QueueSystem, 
                                   buying_prob: float, service_rate: float, avg_airside_dwell_time: float,
                                   boarding_deadline: float, cashier_service_rate: float):
        """Handles customer exiting security queue."""
        if event.customer_id not in metrics.passengers:
            return
        passenger: Passenger = metrics.passengers[event.customer_id]
        flight: Flight = metrics.flights[passenger.flight_id]
        passenger.exit_security(event.time)
        
        wait_time = passenger.get_in_security_queue_time()
        if wait_time is not None:
            metrics.add_security_wait_time(wait_time, passenger.priority_class)
        
        utilization_time = passenger.get_in_security_time()
        if utilization_time is not None:
            metrics.add_security_utilization_time(utilization_time)
        
        buys: bool = np.random.rand() < buying_prob
        
        # Compute airside dwell time
        airside_dwell_time = np.random.exponential(avg_airside_dwell_time)
        boarding_cutoff_time = flight.takeoff_time - boarding_deadline
        expected_total_time = event.time + airside_dwell_time

        if buys:
            expected_cashier_time = cashier_service_rate * 2
            expected_total_time += expected_cashier_time
        
        # If not enough time, skip buying and cap dwell time
        if expected_total_time >= boarding_cutoff_time:
            buys = False
            airside_dwell_time = max(0, min(boarding_cutoff_time - event.time, airside_dwell_time))

        metrics.add_airside_dwell_time(airside_dwell_time)

        # Schedule next event based on buying decision
        next_event_type = EventType.ENTER_CASHIER_QUEUE if buys else EventType.ENTER_BOARDING_QUEUE
        next_event_time = event.time + airside_dwell_time if buys else max(event.time + airside_dwell_time, flight.boarding_start_time)
        next_event = Event(next_event_type, next_event_time, customer_id=passenger.passenger_id, flight_id=passenger.flight_id)
        future_event_set.schedule(next_event)
 
        # Free up security server
        security_queue.release_server()
        security_queue.serving_clients.pop(passenger.passenger_id, None)
        
        # Check if there are more clients in the security queue
        next_passenger = security_queue.remove_client_from_queue()
        if not next_passenger:
            return
        next_passenger.enter_security(event.time)
        security_queue.allocate_server()
        security_queue.serving_clients[next_passenger.passenger_id] = next_passenger
        service_time = np.random.exponential(service_rate)
        exit_security_event = Event(EventType.EXIT_SECURITY_QUEUE, event.time + service_time, customer_id=next_passenger.passenger_id)
        future_event_set.schedule(exit_security_event)
    
    @staticmethod
    def handle_enter_boarding_queue(event: Event, metrics: Metrics, future_event_set: FutureEventSet, boarding_queue: QueueSystem,
                                    service_rate: float):
        """Handles customer entering boarding queue."""
        if event.customer_id not in metrics.passengers:
            return
        passenger: Passenger = metrics.passengers[event.customer_id]
        passenger.enter_boarding_queue(event.time)
        boarding_queue.add_client_to_queue(passenger)
        
        if boarding_queue.is_server_available():
            next_passenger = boarding_queue.remove_client_from_queue()
            next_passenger.enter_boarding(event.time)
            boarding_queue.allocate_server()
            boarding_queue.serving_clients[next_passenger.passenger_id] = next_passenger
            service_time = np.random.exponential(service_rate)
            exit_boarding_event = Event(EventType.EXIT_BOARDING_QUEUE, event.time + service_time, customer_id=next_passenger.passenger_id,
                                        flight_id=next_passenger.flight_id)
            future_event_set.schedule(exit_boarding_event)

    @staticmethod
    def handle_exit_boarding_queue(event: Event, metrics: Metrics, future_event_set: FutureEventSet, boarding_queue: QueueSystem,
                                   service_rate: float):
        """Handles customer exiting boarding queue."""
        if event.customer_id not in metrics.passengers:
            return
        passenger: Passenger = metrics.passengers[event.customer_id]
        passenger.depart(event.time)
        
        wait_time = passenger.get_in_boarding_queue_time()
        if wait_time is not None:
            metrics.add_boarding_wait_time(wait_time, passenger.priority_class)
        
        utilization_time = passenger.get_in_boarding_time()
        if utilization_time is not None:
            metrics.add_boarding_utilization_time(utilization_time, passenger.flight_id)
        
        # Record total time from arrival to departure
        if passenger.departure_time and passenger.arrival_time:
            total_time = passenger.departure_time - passenger.arrival_time
            metrics.add_total_time_per_priority(total_time, passenger.priority_class)
        
        # Passenger has successfully boarded - remove from metrics
        metrics.remove_passenger(passenger)
        flight = metrics.flights[passenger.flight_id]
        flight.served_passengers += 1
                
        # Free up boarding server
        boarding_queue.release_server()
        boarding_queue.serving_clients.pop(passenger.passenger_id, None)
        
        # Check if there are more clients in the boarding queue
        next_passenger = boarding_queue.remove_client_from_queue()
        if not next_passenger:
            return
        next_passenger.enter_boarding(event.time)
        boarding_queue.allocate_server()
        boarding_queue.serving_clients[next_passenger.passenger_id] = next_passenger
        service_time = np.random.exponential(service_rate)
        exit_boarding_event = Event(EventType.EXIT_BOARDING_QUEUE, event.time + service_time, customer_id=next_passenger.passenger_id,
                                    flight_id=next_passenger.flight_id)
        future_event_set.schedule(exit_boarding_event)

    @staticmethod
    def handle_flight_departure(event: Event, metrics: Metrics, future_event_set: FutureEventSet, 
                               cashier_queue_land: QueueSystem, cashier_queue_air: QueueSystem,
                               security_queue: QueueSystem, boarding_queue: QueueSystem):
        """Handles flight departure event."""
        flight: Flight = metrics.flights[event.flight_id]
        passenger_ids = flight.passengers.keys()
        
        # Remove passengers from all queues
        boarding_queue.waiting_queue = [p for p in boarding_queue.waiting_queue if p.passenger_id not in passenger_ids]
        for pid in list(boarding_queue.serving_clients.keys()):
            if pid in passenger_ids:
                boarding_queue.serving_clients.pop(pid)
                boarding_queue.release_server()
        
        security_queue.waiting_queue = [p for p in security_queue.waiting_queue if p.passenger_id not in passenger_ids]
        for pid in list(security_queue.serving_clients.keys()):
            if pid in passenger_ids:
                security_queue.serving_clients.pop(pid)
                security_queue.release_server()
        
        cashier_queue_land.waiting_queue = [p for p in cashier_queue_land.waiting_queue if p.passenger_id not in passenger_ids]
        for pid in list(cashier_queue_land.serving_clients.keys()):
            if pid in passenger_ids:
                cashier_queue_land.serving_clients.pop(pid)
                cashier_queue_land.release_server()
        
        cashier_queue_air.waiting_queue = [p for p in cashier_queue_air.waiting_queue if p.passenger_id not in passenger_ids]
        for pid in list(cashier_queue_air.serving_clients.keys()):
            if pid in passenger_ids:
                cashier_queue_air.serving_clients.pop(pid)
                cashier_queue_air.release_server()
        
        # Remove all passengers and get drop statistics
        for passenger in flight.passengers.values():
            # Only count passengers still in the system
            if passenger.passenger_id in metrics.passengers:
                if passenger.state != PassengerState.LEFT_AIRPORT:
                    metrics.dropped_passengers += 1
                else:
                    flight.served_passengers += 1
                metrics.remove_passenger(passenger)
        
        # Cancel events related to removed customers
        for event in list(future_event_set.events.queue):
            if event.customer_id in passenger_ids:
                event.cancel() 

# --------------------------------------------------
# SIMULATION ENGINE
# --------------------------------------------------
class SimulationEngine:
    """Class to run the airport lounge simulation."""
    def __init__(self, simulation_time: float, buying_prob: float, num_cashier_servers_land: int, num_cashier_servers_air: int, num_security_servers: int, 
                 num_boarding_servers: int,
                 avg_landside_dwell_time: float, avg_airside_dwell_time: float, avg_companions: float, avg_passengers_per_flight: float,
                 priority_probs: list[float], avg_arrival_time_before_flight: float, stddev_arrival_time_before_flight: float,
                 cashier_service_rate: float, security_service_rate: float, boarding_service_rate: float,
                 security_deadline: float, boarding_window_start: float, boarding_deadline: float, flight_freq: float):
        self.simulation_time = simulation_time
        self.buying_prob = buying_prob
        self.future_event_set = FutureEventSet()
        self.metrics = Metrics(num_cashier_servers_land, num_cashier_servers_air, num_security_servers, num_boarding_servers)
        self.cashier_queue_land = QueueSystem(num_cashier_servers_land, priority=False)
        self.cashier_queue_air = QueueSystem(num_cashier_servers_air, priority=False)
        self.security_queue = QueueSystem(num_security_servers, priority=True)
        self.boarding_queues: dict[int, QueueSystem] = {}
        self.num_boarding_servers = num_boarding_servers
        self.next_passenger_id = 0
        self.next_flight_id = 0
        self.avg_landside_dwell_time = avg_landside_dwell_time
        self.avg_airside_dwell_time = avg_airside_dwell_time
        self.avg_companions = avg_companions
        self.avg_passengers_per_flight = avg_passengers_per_flight
        self.priority_probs = priority_probs
        self.avg_arrival_time_before_flight = avg_arrival_time_before_flight
        self.stddev_arrival_time_before_flight = stddev_arrival_time_before_flight
        self.cashier_service_rate = cashier_service_rate
        self.security_service_rate = security_service_rate
        self.boarding_service_rate = boarding_service_rate
        self.security_deadline = security_deadline
        self.boarding_window_start = boarding_window_start
        self.boarding_deadline = boarding_deadline
        self.flight_freq = flight_freq

    def event_loop(self):
        """Main event loop for the simulation."""
        # Schedule the first flight
        first_flight: Flight = Flight(flight_id=self.next_flight_id, takeoff_time=6*60, 
                                      num_passengers=int(np.random.poisson(self.avg_passengers_per_flight)),
                                      boarding_duration=self.boarding_window_start)
        self.metrics.add_flight(first_flight)
        self.boarding_queues[self.next_flight_id] = QueueSystem(self.num_boarding_servers, priority=True)
        flight_event = Event(EventType.FLIGHT_SCHEDULED, 0.0, flight_id=self.next_flight_id)
        self.future_event_set.schedule(flight_event)
        self.next_flight_id += 1

        while not self.future_event_set.is_empty():
            event = self.future_event_set.get_next_event()
            if event is None or event.time > self.simulation_time:
                break
            
            # Update time-weighted metrics
            num_in_airport = len(self.metrics.passengers)
            num_in_landside = sum(1 for p in self.metrics.passengers.values() if p.state in [PassengerState.ARRIVED, PassengerState.IN_CASHIER_LAND])
            num_in_airside = sum(1 for p in self.metrics.passengers.values() if p.state in [PassengerState.IN_AIRSIDE, PassengerState.IN_CASHIER_AIR])
            num_in_security = sum(1 for p in self.metrics.passengers.values() if p.state == PassengerState.IN_SECURITY)
            num_in_boarding = sum(1 for p in self.metrics.passengers.values() if p.state == PassengerState.IN_BOARDING)
            num_in_security_queue = len(self.security_queue.waiting_queue)
            num_in_boarding_queue = sum(len(queue.waiting_queue) for queue in self.boarding_queues.values())
            num_in_cashier_land_queue = len(self.cashier_queue_land.waiting_queue)
            num_in_cashier_air_queue = len(self.cashier_queue_air.waiting_queue)
            self.metrics.update_time_weighted_metrics(event.time, num_in_airport, num_in_landside, num_in_airside,
                                                      num_in_security_queue, num_in_boarding_queue, 
                                                      num_in_cashier_land_queue, num_in_cashier_air_queue)
            self.metrics.sample_statistics(event.time, num_in_airport, num_in_landside, num_in_airside,
                                         num_in_security, num_in_boarding)

            # Handle the event
            if event.event_type == EventType.FLIGHT_SCHEDULED:
                self.next_passenger_id = EventHandlers.handle_flight_scheduled(event, self.metrics, self.future_event_set, self.next_passenger_id,
                                                     self.avg_companions, self.priority_probs,
                                                     self.avg_arrival_time_before_flight, self.stddev_arrival_time_before_flight)
                next_flight_dep_time = event.time + 6*60 + self.flight_freq
                # Schedule the next flight (30 minutes later)
                schedule_time = event.time + self.flight_freq
                if (next_flight_dep_time % (24*60)) <= 23*60:
                    # Flights terminate at 11 PM
                    next_flight: Flight = Flight(flight_id=self.next_flight_id, takeoff_time=next_flight_dep_time,
                                                num_passengers=int(np.random.poisson(self.avg_passengers_per_flight)),
                                                boarding_duration=self.boarding_window_start)
                    self.metrics.add_flight(next_flight)
                    self.boarding_queues[self.next_flight_id] = QueueSystem(self.num_boarding_servers, priority=True)
                    flight_event = Event(EventType.FLIGHT_SCHEDULED, schedule_time, flight_id=self.next_flight_id)
                    self.future_event_set.schedule(flight_event)
                    self.next_flight_id += 1
            elif event.event_type == EventType.CUSTOMER_ARRIVAL:
                passenger = self.metrics.flights[event.flight_id].passengers[event.customer_id]
                self.metrics.add_passenger(passenger)
                EventHandlers.handle_customer_arrival(event, self.metrics, self.future_event_set, self.buying_prob,
                                                     self.avg_landside_dwell_time, self.security_deadline, self.cashier_service_rate)
            elif event.event_type == EventType.ENTER_CASHIER_QUEUE:
                side = "landside" if self.metrics.passengers[event.customer_id].state == PassengerState.ARRIVED else "airside"
                cashier_queue = self.cashier_queue_land if side == "landside" else self.cashier_queue_air
                EventHandlers.handle_enter_cashier_queue(event, self.metrics, self.future_event_set, cashier_queue,
                                                         self.cashier_service_rate)
            elif event.event_type == EventType.EXIT_CASHIER_QUEUE:
                side = "landside" if self.metrics.passengers[event.customer_id].state == PassengerState.IN_CASHIER_LAND else "airside"
                cashier_queue = self.cashier_queue_land if side == "landside" else self.cashier_queue_air
                EventHandlers.handle_exit_cashier_queue(event, self.metrics, self.future_event_set, cashier_queue, side,
                                                        self.cashier_service_rate)
            elif event.event_type == EventType.ENTER_SECURITY_QUEUE:
                EventHandlers.handle_enter_security_queue(event, self.metrics, self.future_event_set, self.security_queue,
                                                          self.security_service_rate)
            elif event.event_type == EventType.EXIT_SECURITY_QUEUE:
                EventHandlers.handle_exit_security_queue(event, self.metrics, self.future_event_set, self.security_queue,
                                                         self.buying_prob, self.security_service_rate, self.avg_airside_dwell_time,
                                                         self.boarding_deadline, self.cashier_service_rate)
            elif event.event_type == EventType.ENTER_BOARDING_QUEUE:
                boarding_queue = self.boarding_queues[event.flight_id]
                EventHandlers.handle_enter_boarding_queue(event, self.metrics, self.future_event_set, boarding_queue,
                                                          self.boarding_service_rate)
            elif event.event_type == EventType.EXIT_BOARDING_QUEUE:
                boarding_queue = self.boarding_queues[event.flight_id]
                EventHandlers.handle_exit_boarding_queue(event, self.metrics, self.future_event_set, boarding_queue,
                                                         self.boarding_service_rate)
            elif event.event_type == EventType.FLIGHT_DEPARTURE:
                boarding_queue = self.boarding_queues[event.flight_id]
                EventHandlers.handle_flight_departure(event, self.metrics, self.future_event_set,
                                                     self.cashier_queue_land, self.cashier_queue_air,
                                                     self.security_queue, boarding_queue)
                # Clean up boarding queue for the departed flight
                self.boarding_queues.pop(event.flight_id, None)
    
    def print_statistics(self, verbose: bool = True):
        """Print simulation statistics."""
        total_simulation_time = self.simulation_time
        
        if not verbose:
            # Print only KPIs for CI computation
            return {
                'avg_security_wait': self.metrics.get_average_security_wait_time(),
                'avg_boarding_wait': self.metrics.get_average_boarding_wait_time(),
                'avg_total_time': sum(sum(times) for times in self.metrics.total_time_per_priority.values()) / 
                                 sum(len(times) for times in self.metrics.total_time_per_priority.values()) if sum(len(times) for times in self.metrics.total_time_per_priority.values()) > 0 else 0.0,
                'drop_percentage': self.metrics.get_drop_percentage(),
                'security_utilization': self.metrics.get_security_utilization(total_simulation_time),
                'boarding_utilization': self.metrics.get_boarding_utilization(self.boarding_window_start)
            }
        
        print(f"\n{'='*60}")
        print(f"SIMULATION STATISTICS")
        print(f"{'='*60}")
        
        print(f"\n--- General Statistics ---")
        print(f"Total Customer Arrivals: {self.metrics.total_customer_arrivals}")
        print(f"Total Passenger Arrivals: {self.metrics.total_passengers_arrivals}")
        print(f"Total Flights: {self.metrics.total_flights}")
        print(f"Average Passengers per Flight: {self.metrics.get_average_passengers_per_flight():.2f}")
        print(f"Dropped Passengers: {self.metrics.dropped_passengers}")
        print(f"Drop Percentage: {self.metrics.get_drop_percentage():.2f}%")
        
        print(f"\n--- Airport Occupancy (Time-Weighted Averages) ---")
        print(f"Average Number in Airport: {self.metrics.get_average_number_in_airport(total_simulation_time):.2f}")
        print(f"Average Number in Landside: {self.metrics.get_average_number_in_landside(total_simulation_time):.2f}")
        print(f"Average Number in Airside: {self.metrics.get_average_number_in_airside(total_simulation_time):.2f}")
        
        print(f"\n--- Average Queue Lengths (Time-Weighted) ---")
        print(f"Average Cashier Landside Queue Length: {self.metrics.get_average_cashier_land_queue_length(total_simulation_time):.2f}")
        print(f"Average Cashier Airside Queue Length: {self.metrics.get_average_cashier_air_queue_length(total_simulation_time):.2f}")
        print(f"Average Security Queue Length: {self.metrics.get_average_security_queue_length(total_simulation_time):.2f}")
        print(f"Average Boarding Queue Length: {self.metrics.get_average_boarding_queue_length(total_simulation_time):.2f}")
        
        print(f"\n--- Dwell Times ---")
        print(f"Average Landside Dwell Time: {self.metrics.get_average_landside_dwell_time():.2f} minutes")
        print(f"Average Airside Dwell Time: {self.metrics.get_average_airside_dwell_time():.2f} minutes")
        
        print(f"\n--- Waiting Times (Overall) ---")
        print(f"Average Cashier Landside Wait Time: {self.metrics.get_average_cashier_land_wait_time():.2f} minutes")
        print(f"Average Cashier Airside Wait Time: {self.metrics.get_average_cashier_air_wait_time():.2f} minutes")
        print(f"Average Security Wait Time: {self.metrics.get_average_security_wait_time():.2f} minutes")
        print(f"Average Boarding Wait Time: {self.metrics.get_average_boarding_wait_time():.2f} minutes")
        
        print(f"\n--- Waiting Times by Priority Class ---")
        for priority in PriorityClass:
            sec_wait = self.metrics.get_average_security_wait_per_priority(priority)
            board_wait = self.metrics.get_average_boarding_wait_per_priority(priority)
            total_time = self.metrics.get_average_total_time_per_priority(priority)
            print(f"{priority.name:12} - Security: {sec_wait:6.2f} min | Boarding: {board_wait:6.2f} min | Total: {total_time:7.2f} min")
        
        print(f"\n--- Resource Utilization ---")
        print(f"Cashier Landside Utilization: {self.metrics.get_cashier_land_utilization(total_simulation_time)*100:.2f}%")
        print(f"Cashier Airside Utilization: {self.metrics.get_cashier_air_utilization(total_simulation_time)*100:.2f}%")
        print(f"Security Utilization: {self.metrics.get_security_utilization(total_simulation_time)*100:.2f}%")
        print(f"Boarding Utilization (avg per flight): {self.metrics.get_boarding_utilization(self.boarding_window_start)*100:.2f}%") 

        print(f"\n{'='*60}")
        return None
    
    def plot_time_series(self):
        """Plot time series of system state."""
        import matplotlib.pyplot as plt
        
        # Extract time series data
        times_airport = sorted(self.metrics.samples_number_in_airport.keys())
        counts_airport = [self.metrics.samples_number_in_airport[t] for t in times_airport]
        
        times_landside = sorted(self.metrics.samples_number_in_landside.keys())
        counts_landside = [self.metrics.samples_number_in_landside[t] for t in times_landside]
        
        times_airside = sorted(self.metrics.samples_number_in_airside.keys())
        counts_airside = [self.metrics.samples_number_in_airside[t] for t in times_airside]
        
        times_security = sorted(self.metrics.samples_number_in_security.keys())
        counts_security = [self.metrics.samples_number_in_security[t] for t in times_security]
        
        times_boarding = sorted(self.metrics.samples_number_in_boarding.keys())
        counts_boarding = [self.metrics.samples_number_in_boarding[t] for t in times_boarding]
        
        # ========== FIGURE 1: Wait Times by Priority Class (Security and Boarding) ==========
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        priority_names = [pc.name for pc in PriorityClass]
        security_waits = [self.metrics.get_average_security_wait_per_priority(pc) for pc in PriorityClass]
        boarding_waits = [self.metrics.get_average_boarding_wait_per_priority(pc) for pc in PriorityClass]
        
        x = np.arange(len(priority_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, security_waits, width, label='Security Wait', 
                       color='#3498db', alpha=0.8, edgecolor='black')
        bars2 = ax1.bar(x + width/2, boarding_waits, width, label='Boarding Wait', 
                       color='#e74c3c', alpha=0.8, edgecolor='black')
        
        ax1.set_title('Average Wait Times by Priority Class', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Priority Class', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Wait Time (minutes)', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(priority_names)
        ax1.legend(fontsize=11, loc='upper right')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        # plt.savefig('wait_times_by_class.png', dpi=300, bbox_inches='tight') # Save if needed
        # print(f"\n📊 Wait times by class plot saved to: wait_times_by_class.png")
        plt.show()
        
        # ========== FIGURE 2: Airport System Time Series (2 subplots) ==========
        fig2, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig2.suptitle('Airport System Time Series Analysis', fontsize=16, fontweight='bold')
        
        # Subplot 1: Total Passengers in Airport
        axes[0].plot(times_airport, counts_airport, 'b-', linewidth=2, label='Total in Airport')
        axes[0].set_title('Total Passengers in Airport', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Time (minutes)', fontsize=11)
        axes[0].set_ylabel('Number of Passengers', fontsize=11)
        axes[0].legend(fontsize=10, loc='upper right')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        # Subplot 2: Breakdown by Location (4 lines)
        axes[1].plot(times_landside, counts_landside, 'g-', label='Landside', linewidth=2)
        axes[1].plot(times_airside, counts_airside, 'r-', label='Airside', linewidth=2)
        axes[1].plot(times_security, counts_security, 'm-', label='In Security', linewidth=2)
        axes[1].plot(times_boarding, counts_boarding, 'c-', label='In Boarding', linewidth=2)
        axes[1].set_title('Passengers by Location', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Time (minutes)', fontsize=11)
        axes[1].set_ylabel('Number of Passengers', fontsize=11)
        axes[1].legend(fontsize=10, loc='upper right')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        # plt.savefig('airport_time_series.png', dpi=300, bbox_inches='tight') # Save if needed
        # print(f"📊 Time series plot saved to: airport_time_series.png")
        plt.show()


def run_multiple_simulations(num_runs: int, base_seed: int, show_plots: bool = False, verbose: bool = False):
    """
    Run multiple independent simulations with different seeds and compute confidence intervals.
    
    Args:
        num_runs: Number of independent simulation runs
        base_seed: Base seed value (each run uses base_seed + i)
        show_plots: Whether to show plots for the last run
        verbose: Whether to print detailed statistics for each run
    
    Returns:
        Dictionary with CI results for key metrics
    """
    print(f"\n{'='*80}")
    print(f"RUNNING {num_runs} INDEPENDENT SIMULATIONS")
    print(f"{'='*80}\n")
    
    # Define simulation parameters (same as main)
    SIMULATION_TIME = 24 * 60  # 1440 minutes = 24 hours
    BUYING_PROB = 0.8
    NUM_CASHIER_SERVERS_LAND = 17
    NUM_CASHIER_SERVERS_AIR = 8
    NUM_SECURITY_SERVERS = 8
    NUM_BOARDING_SERVERS = 2
    AVG_LANDSIDE_DWELL_TIME = 30.0
    AVG_AIRSIDE_DWELL_TIME = 15.0
    AVG_COMPANIONS = 1.5
    AVG_PASSENGERS_PER_FLIGHT = 100.0
    PRIORITY_PROBS = [0.7, 0.2, 0.1]  # Economy, Business, First Class
    AVG_ARRIVAL_TIME_BEFORE_FLIGHT = 120.0
    STDDEV_ARRIVAL_TIME_BEFORE_FLIGHT = 30.0
    CASHIER_SERVICE_RATE = 2.0
    SECURITY_SERVICE_RATE = 1.5
    BOARDING_SERVICE_RATE = 0.5
    SECURITY_DEADLINE = 40
    BOARDING_WINDOW_START = 45
    BOARDING_DEADLINE = 20
    FLIGHT_FREQUENCY = 20
    
    # Initialize confidence interval calculators for key metrics
    ci_calculators = {
        'avg_security_wait': ConfidenceInterval(min_samples_count=5, max_interval_width=0.10, confidence_level=0.95),
        'avg_boarding_wait': ConfidenceInterval(min_samples_count=5, max_interval_width=0.10, confidence_level=0.95),
        'avg_total_time': ConfidenceInterval(min_samples_count=5, max_interval_width=0.05, confidence_level=0.95),
        'drop_percentage': ConfidenceInterval(min_samples_count=5, max_interval_width=0.10, confidence_level=0.95),
        'security_utilization': ConfidenceInterval(min_samples_count=5, max_interval_width=0.05, confidence_level=0.95),
        'boarding_utilization': ConfidenceInterval(min_samples_count=5, max_interval_width=0.05, confidence_level=0.95)
    }
    
    all_kpis = []
    last_engine = None
    
    # Run simulations
    for i in range(num_runs):
        seed = base_seed + i
        np.random.seed(seed)
        
        print(f"Run {i+1}/{num_runs} (seed={seed})...", end=" ")
        
        # Create and run simulation
        engine = SimulationEngine(SIMULATION_TIME, BUYING_PROB, NUM_CASHIER_SERVERS_LAND, NUM_CASHIER_SERVERS_AIR,
                                 NUM_SECURITY_SERVERS, NUM_BOARDING_SERVERS, AVG_LANDSIDE_DWELL_TIME,
                                 AVG_AIRSIDE_DWELL_TIME, AVG_COMPANIONS, AVG_PASSENGERS_PER_FLIGHT,
                                 PRIORITY_PROBS, AVG_ARRIVAL_TIME_BEFORE_FLIGHT, STDDEV_ARRIVAL_TIME_BEFORE_FLIGHT,
                                 CASHIER_SERVICE_RATE, SECURITY_SERVICE_RATE, BOARDING_SERVICE_RATE,
                                 SECURITY_DEADLINE, BOARDING_WINDOW_START, BOARDING_DEADLINE, FLIGHT_FREQUENCY)
        engine.event_loop()
        
        # Get KPIs (non-verbose mode)
        kpis = engine.print_statistics(verbose=False)
        all_kpis.append(kpis)
        
        # Add to CI calculators
        for metric_name, value in kpis.items():
            ci_calculators[metric_name].add_data_point(value)
        
        print(f"Drop rate: {kpis['drop_percentage']:.2f}%")
        
        if verbose:
            engine.print_statistics(verbose=True)
        
        last_engine = engine
    
    # Compute confidence intervals
    print(f"\n{'='*80}")
    print(f"CONFIDENCE INTERVAL RESULTS (95% confidence, {num_runs} runs)")
    print(f"{'='*80}\n")
    
    ci_results = {}
    for metric_name, calculator in ci_calculators.items():
        if calculator.has_enough_data():
            result = calculator.compute_interval()
            if result:
                is_final, (lower, upper) = result
                mean = np.mean([kpi[metric_name] for kpi in all_kpis])
                ci_results[metric_name] = (mean, lower, upper, is_final)
                
                status = "✓ CONVERGED" if is_final else "○ MORE DATA NEEDED"
                print(f"{metric_name:25} {status:20} Mean: {mean:8.3f}  CI: [{lower:8.3f}, {upper:8.3f}]  Width: {upper-lower:7.3f}")
            else:
                print(f"{metric_name:25} {'✗ INSUFFICIENT DATA':20}")
        else:
            print(f"{metric_name:25} {'✗ INSUFFICIENT DATA':20}")
    
    print(f"\n{'='*80}\n")
    
    # Show plots for last run if requested
    if show_plots and last_engine:
        print("Generating plots for last run...")
        last_engine.plot_time_series()
    
    return ci_results, all_kpis

    
# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------
if __name__ == "__main__":
    # Choose mode: single run or multiple runs with CI
    MODE = "multiple"  # Change to "single" for single detailed run with plots
    
    if MODE == "single":
        # Single detailed run with plots
        np.random.seed(42)
        
        # ============================================================================
        # SIMULATION HYPERPARAMETERS - All configurable parameters
        # ============================================================================
        
        # --- Temporal Configuration ---
        SIMULATION_TIME = 24 * 60  # 24 hours (full working day) in minutes
        
        # --- Flight Scheduling ---
        FLIGHT_FREQUENCY = 20  # Minutes between each flight departure
        AVG_PASSENGERS_PER_FLIGHT = 100.0  # Average number of passengers per flight
        
        # --- Resource Allocation (Server Counts) ---
        NUM_CASHIER_SERVERS_LAND = 17  # Number of cashier servers in landside
        NUM_CASHIER_SERVERS_AIR = 8    # Number of cashier servers in airside
        NUM_SECURITY_SERVERS = 8       # Number of security screening servers
        NUM_BOARDING_SERVERS = 2       # Number of boarding gate servers
        
        # --- Service Rates (minutes per customer) ---
        CASHIER_SERVICE_RATE = 2.0   # Average service time at cashier (exponential)
        SECURITY_SERVICE_RATE = 1.5  # Average service time at security (exponential)
        BOARDING_SERVICE_RATE = 0.5  # Average service time at boarding (exponential)
        
        # --- Passenger Behavior ---
        BUYING_PROB = 0.8              # Probability passenger shops at cashiers (0.0-1.0)
        AVG_COMPANIONS = 1.5           # Average number of companions per customer group
        AVG_LANDSIDE_DWELL_TIME = 30.0 # Average time spent shopping landside (minutes)
        AVG_AIRSIDE_DWELL_TIME = 15.0  # Average time spent shopping airside (minutes)
        
        # --- Arrival Patterns ---
        AVG_ARRIVAL_TIME_BEFORE_FLIGHT = 120.0     # Average arrival time before flight (minutes)
        STDDEV_ARRIVAL_TIME_BEFORE_FLIGHT = 30.0   # Std deviation of arrival times (minutes)
        
        # --- Priority Class Distribution ---
        PRIORITY_PROBS = [0.7, 0.2, 0.1]  # [Economy, Business, First Class] probabilities
        
        # --- Time Constraints ---
        SECURITY_DEADLINE = 40      # Must finish security AT LEAST this many min before takeoff
        BOARDING_WINDOW_START = 45  # Boarding begins this many minutes before takeoff
        BOARDING_DEADLINE = 20      # Must finish boarding AT LEAST this many min before takeoff

        # Initialize and run simulation
        simulation_engine = SimulationEngine(SIMULATION_TIME, BUYING_PROB, NUM_CASHIER_SERVERS_LAND, NUM_CASHIER_SERVERS_AIR,
                                             NUM_SECURITY_SERVERS, NUM_BOARDING_SERVERS, AVG_LANDSIDE_DWELL_TIME,
                                             AVG_AIRSIDE_DWELL_TIME, AVG_COMPANIONS, AVG_PASSENGERS_PER_FLIGHT,
                                             PRIORITY_PROBS, AVG_ARRIVAL_TIME_BEFORE_FLIGHT, STDDEV_ARRIVAL_TIME_BEFORE_FLIGHT,
                                             CASHIER_SERVICE_RATE, SECURITY_SERVICE_RATE, BOARDING_SERVICE_RATE,
                                             SECURITY_DEADLINE, BOARDING_WINDOW_START, BOARDING_DEADLINE, FLIGHT_FREQUENCY)
        simulation_engine.event_loop()
        simulation_engine.print_statistics(verbose=True)
        simulation_engine.plot_time_series()
        
    elif MODE == "multiple":
        # Multiple runs with confidence intervals
        NUM_RUNS = 20  # Number of independent replications
        BASE_SEED = 0  # Base seed value
        SHOW_PLOTS = True  # Show plots for last run
        VERBOSE = False  # Set to True to see detailed stats for each run
        
        ci_results, all_kpis = run_multiple_simulations(NUM_RUNS, BASE_SEED, SHOW_PLOTS, VERBOSE)
        
        # Print summary statistics
        print("\nSummary Statistics Across All Runs:")
        print(f"{'='*80}")
        for metric in all_kpis[0].keys():
            values = [kpi[metric] for kpi in all_kpis]
            print(f"{metric:25} Mean: {np.mean(values):8.3f}  Std: {np.std(values):7.3f}  Min: {np.min(values):8.3f}  Max: {np.max(values):8.3f}")
        print(f"{'='*80}")