from enum import Enum
from queue import PriorityQueue
from typing import Optional
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import time
from itertools import product
from transient_detection import TransientDetection as TransientDetector
from confidence_interval import ConfidenceInterval


# ------------------------------------------------------------------------------
# EVENT DEFINITION
# ------------------------------------------------------------------------------
class EventType(Enum):
    """
    Enumeration of event types in the car sharing system.
    
    Attributes:
        USER_REQUEST: Event triggered when a user requests a car
        USER_ABANDON: Event triggered when a user abandons their request after waiting too long
        CAR_PARKING: Event triggered when a car completes a trip or relocation and attempts to park
        CAR_AVAILABLE: Event triggered when a car becomes available (charged to minimum autonomy)
        CAR_RELOCATE: Event triggered to redistribute cars across the system
    """
    USER_REQUEST = "USER_REQUEST"
    USER_ABANDON = "USER_ABANDON"
    CAR_PARKING = "CAR_PARKING"
    CAR_AVAILABLE = "CAR_AVAILABLE"
    CAR_RELOCATE = "CAR_RELOCATE"

class Event:
    """
    Class representing an event in the car sharing discrete-event simulation.
    
    Events are scheduled in the future event set and processed in chronological order.
    Each event has a type, time, and optional entity identifiers (user, car, station).
    
    Attributes:
        event_type (EventType): Type of event
        time (float): Simulation time when event occurs (in minutes)
        user_id (Optional[int]): ID of user involved in event (if applicable)
        car_id (Optional[int]): ID of car involved in event (if applicable)
        station_id (Optional[int]): ID of station involved in event (if applicable)
        cancelled (bool): Flag indicating if event has been cancelled
    """
    def __init__(self, event_type: EventType, time: float, user_id: Optional[int] = None, 
                 car_id: Optional[int] = None, station_id: Optional[int] = None):
        self.event_type = event_type
        self.time = time
        self.user_id = user_id
        self.car_id = car_id
        self.station_id = station_id
        self.cancelled: bool = False

    def cancel(self):
        """Mark this event as cancelled so it will be skipped during processing."""
        self.cancelled = True

    def is_cancelled(self) -> bool:
        """Check if this event has been cancelled."""
        return self.cancelled
    
    def __lt__(self, other) -> bool:
        """
        Compare events by time for priority queue ordering.
        
        Args:
            other: Another Event object to compare with
            
        Returns:
            True if this event occurs before the other event
        """
        return self.time < other.time
   

# ------------------------------------------------------------------------------
# CAR
# ------------------------------------------------------------------------------
class Car:
    """
    Class representing a car in the car sharing system.
    
    Cars can be in various states: available, charging, assigned to user, or relocating.
    They consume autonomy when driving and must charge at stations when autonomy is low.
    
    Attributes:
        car_id (int): Unique identifier for the car
        location (tuple): Current (x, y) coordinates on the map
        autonomy (float): Current remaining range in hectometers (hm)
        max_autonomy (float): Maximum range capacity in hm
        charging_rate (float): Charging speed in hm per minute
        min_autonomy (float): Minimum required autonomy to be considered available
        max_destination_charging_distance (float): Maximum distance acceptable to travel between charging station and destination
        speed (float): Current speed in km/h
        station_id (Optional[int]): ID of station where car is parked (None if not at station)
        available (bool): Whether car is available for assignment to users
        assigned_user_id (Optional[int]): ID of user currently assigned to car (None if no user)
        charging (bool): Whether car is currently charging
        relocating (bool): Whether car is being relocated by the system
        relocating_station_id (Optional[int]): Destination station ID when relocating
    """
    def __init__(self, car_id: int, location: tuple, max_autonomy: float, charging_rate: float, min_autonomy: float, 
                 max_destination_charging_distance: float, speed: float, station_id: int):
        self.car_id = car_id
        self.location = location
        self.autonomy = max_autonomy
        self.max_autonomy = max_autonomy
        self.charging_rate = charging_rate
        self.min_autonomy = min_autonomy
        self.max_destination_charging_distance = max_destination_charging_distance
        self.speed = speed
        self.station_id = station_id
        self.available: bool = True
        self.assigned_user_id: Optional[int] = None
        self.charging: bool = True
        self.relocating: bool = False
        self.relocating_station_id: Optional[int] = None

    def drive(self, distance: float, destination: tuple) -> bool:
        """
        Drive the car to a new location, consuming autonomy.
        
        Args:
            distance: Distance to travel in grid units
            destination: Target (x, y) coordinates
            
        Returns:
            True if drive successful
        """
        self.location = destination
        self.autonomy -= distance
        return True

    def charge(self, time: float):
        """
        Charge the car for a given time interval if currently charging.
        
        Updates autonomy and availability status based on charging progress.
        Car becomes available when autonomy reaches min_autonomy threshold.
        
        Args:
            time: Time interval for charging in minutes
        """
        if self.charging:
            self.autonomy = min(self.autonomy + self.charging_rate * time, self.max_autonomy)
            if self.autonomy >= self.min_autonomy:
                self.available = True
    
    def park_in_station(self, station_id: int):
        """
        Mark the car as parked at a specific charging station.
        
        Args:
            station_id: ID of the station where car is parked
        """
        self.station_id = station_id

    def unpark_from_station(self):
        """Remove the car's association with a charging station."""
        self.station_id = None

    def assign_user(self, user_id: int):
        """
        Assign this car to a user for a trip.
        
        Sets car as unavailable, stops charging, and records user assignment.
        
        Args:
            user_id: ID of user to assign to this car
        """
        if self.available:
            self.charging = False
            self.assigned_user_id = user_id
            self.available = False
        else:
            print("Car is not available for assignment.")
    
    def release_user(self):
        """
        Release the user from this car after trip completion.
        
        Updates availability based on remaining autonomy. Car is only available
        if autonomy is above minimum threshold.
        """
        if self.assigned_user_id is not None:
            self.assigned_user_id = None
            if self.autonomy < self.min_autonomy:
                self.available = False
            else:
                self.available = True
        else:
            print("No user is assigned to this car.")

    def is_available(self) -> bool:
        """
        Check if car is available for user assignment.
        
        Returns:
            True if car is available and can accept a user
        """
        return self.available

    def relocate(self, station_id: int):
        """
        Mark the car as relocating to a given station.
        
        Args:
            station_id: Destination station ID for relocation
        """
        self.relocating = True
        self.relocating_station_id = station_id

    def is_relocating(self) -> bool:
        """
        Check if car is currently being relocated.
        
        Returns:
            True if car is in relocation mode
        """
        return self.relocating

    def get_relocating_station_id(self) -> int:
        """
        Get the destination station ID for current relocation.
        
        Returns:
            Station ID where car is being relocated to
        """
        return self.relocating_station_id
    
    def stop_relocating(self):
        """Stop relocation mode and clear destination station."""
        self.relocating = False
        self.relocating_station_id = None


# ------------------------------------------------------------------------------
# USER
# ------------------------------------------------------------------------------
class User:
    """
    Class representing a user in the car sharing system.
    
    Users request cars, wait for assignment, and travel to destinations.
    If no car becomes available within max_waiting_time, users abandon their request.
    
    Attributes:
        user_id (int): Unique identifier for the user
        location (tuple): Origin (x, y) coordinates
        destination (tuple): Destination (x, y) coordinates
        max_waiting_time (float): Maximum time user will wait before abandoning (minutes)
        max_pickup_distance (float): Maximum distance user accepts for car pickup (hm)
        request_time (float): Simulation time when user made request (minutes)
        waiting (bool): Whether user is currently waiting for car
        assigned_car_id (Optional[int]): ID of assigned car (None if no car assigned)
        service_time (Optional[float]): Time when car was assigned (minutes)
        served (bool): Whether user completed their trip
    """
    def __init__(self, user_id: int, location: tuple, destination: tuple, max_waiting_time: float, 
                 max_pickup_distance: float, request_time: float):
        self.user_id = user_id
        self.location = location
        self.destination = destination
        self.max_waiting_time = max_waiting_time
        self.max_pickup_distance = max_pickup_distance
        self.request_time = request_time
        self.waiting: bool = True
        self.assigned_car_id: Optional[int] = None
        self.service_time: Optional[float] = None
        self.served: bool = False

    def assign_car(self, car_id: int, service_time: float):
        """
        Assign a car to this user and record service start time.
        
        Args:
            car_id: ID of car assigned to user
            service_time: Simulation time when car was assigned (minutes)
        """
        self.assigned_car_id = car_id
        self.waiting = False
        self.service_time = service_time

    def resign_car(self):
        """
        Release the car after trip completion and mark user as served.
        """
        self.assigned_car_id = None
        self.served = True

    def request_car(self):
        """
        Mark user as requesting/waiting for a car.
        """
        self.waiting = True

    def abandon_request(self):
        """
        User abandons their request after waiting too long without car assignment.
        """
        self.waiting = False
        self.assigned_car_id = None

    def put_in_queue(self):
        """
        Place user in the waiting queue for car assignment.
        """
        self.waiting = True

    def is_waiting(self) -> bool:
        """
        Check if user is currently waiting for car assignment.
        
        Returns:
            True if user is in waiting state
        """
        return self.waiting

    def waiting_time(self) -> float:
        """
        Calculate the waiting time between request and service.
        
        Returns:
            Waiting time in minutes, or None if user hasn't been served yet.
            Returns 0.0 if calculated value would be negative (data inconsistency).
        """
        if self.service_time is not None:
            wt = self.service_time - self.request_time
            if wt < 0:
                return 0.0
            return wt
        return None


# ------------------------------------------------------------------------------
# CHARGING STATION
# ------------------------------------------------------------------------------
class ChargingStation:
    """
    Class representing a charging station in the car sharing system.
    
    Stations provide parking spots where cars can charge their batteries.
    Each station has a fixed capacity and tracks which cars are currently parked.
    
    Attributes:
        station_id (int): Unique identifier for the station
        location (tuple): (x, y) coordinates on the map
        capacity (int): Maximum number of cars that can park simultaneously
        occupied_spots (int): Current number of cars parked at station
        parked_cars (list[int]): List of car IDs currently parked
        full (bool): Whether station is at full capacity
    """
    def __init__(self, station_id: int, location: tuple, capacity: int):
        self.station_id = station_id
        self.location = location 
        self.capacity = capacity
        self.occupied_spots: int = 0
        self.parked_cars: list[int] = []
        self.full: bool = False

    def park_car(self, car_id: int) -> bool:
        """
        Park a car at this charging station if space available.
        
        Updates occupied_spots counter and full status. Adds car to parked_cars list.
        
        Args:
            car_id: ID of car to park
            
        Returns:
            True if car was successfully parked, False if station is full
        """
        if not self.full:
            self.parked_cars.append(car_id)
            self.occupied_spots += 1
            if self.occupied_spots == self.capacity:
                self.full = True
            return True
        else:
            print("No available spots at the charging station.")
            return False

    def remove_car(self, car_id: int) -> bool:
        """
        Remove a car from this charging station.
        
        Decrements occupied_spots counter and clears full status.
        Removes car from parked_cars list.
        
        Args:
            car_id: ID of car to remove
            
        Returns:
            True if car was found and removed, False if car not at station
        """
        if car_id in self.parked_cars:
            self.parked_cars.remove(car_id)
            self.occupied_spots -= 1
            self.full = False
            return True
        else:
            print("Car not found at the charging station.")
            return False

    def is_full(self) -> bool:
        """
        Check if station is at full capacity.
        
        Returns:
            True if all parking spots are occupied
        """
        return self.full
        

# ------------------------------------------------------------------------------
# FUTURE EVENT SET
# ------------------------------------------------------------------------------
class FutureEventSet:
    """
    Class representing the future event set for discrete-event simulation.
    
    Manages a priority queue of events ordered by time. Events are processed
    chronologically, and cancelled events are automatically skipped.
    
    Attributes:
        events (PriorityQueue[Event]): Priority queue of scheduled events
        event_count (int): Total number of events scheduled (including cancelled)
    """
    def __init__(self):
        self.events: PriorityQueue[Event] = PriorityQueue()
        self.event_count: int = 0

    def schedule(self, event: Event):
        """
        Schedule an event in the future event set.
        
        Events are automatically ordered by time in the priority queue.
        
        Args:
            event: Event to schedule
        """
        self.events.put(event)
        self.event_count += 1

    def get_next_event(self) -> Optional[Event]:
        """
        Retrieve and remove the next non-cancelled event from the set.
        
        Automatically skips over any cancelled events in the queue.
        
        Returns:
            Next event to process, or None if queue is empty
        """
        if not self.is_empty():
            while not self.events.empty():
                event = self.events.get()
                if not event.is_cancelled():
                    return event
        return None
    
    def is_empty(self) -> bool:
        """
        Check if the future event set is empty.
        
        Returns:
            True if no events are scheduled
        """
        return self.events.empty()
    

# -------------------------------------------------------------------------------
# CAR SHARING SYSTEM
# -------------------------------------------------------------------------------
class CarSharingSystem:
    """
    Class representing the car sharing system and managing all statistics.
    
    This class is the central repository for all system state and performance metrics.
    It tracks cars, users, stations, and collects both instantaneous and time-weighted
    statistics for system analysis and performance evaluation.
    
    Attributes:
        cars (list[Car]): All cars in the system
        stations (list[ChargingStation]): All charging stations
        
        -- Cumulative Statistics --
        total_user_requests (int): Total number of user requests
        total_abandoned_requests (int): Total number of abandoned requests
        total_waiting_time (float): Cumulative waiting time across all users (minutes)
        total_trip_distance (float): Cumulative trip distance across all trips (grid units)
        last_relocation_time (float): Time of last relocation event (minutes)
        
        -- Entity Tracking Dictionaries --
        waiting_users (dict[int, User]): Users currently waiting for car assignment
        active_users (dict[int, User]): All users currently in the system
        abandoned_users (dict[int, User]): Users who abandoned their requests
        charging_cars (dict[int, Car]): Cars currently charging at stations
        dispersed_cars (dict[int, Car]): Cars not at stations (stranded after trip)
        
        -- Time-Series Sample Statistics --
        sampled_num_serving_users (list[tuple[float, int]]): Serving users count samples (time, count)
        sampled_num_total_users (list[tuple[float, int]]): Total active users count samples (time, count)
        num_samples (int): Total number of samples taken
        
        -- Peak Window Statistics --
        peak_window_stats (list[dict]): Statistics for each peak arrival rate window
        peak_window_area_available (float): Integral of available cars during peak windows
        peak_window_area_used (float): Integral of used cars during peak windows
        peak_window_area_station (float): Integral of station utilization during peak windows
        peak_window_area_queue (float): Integral of queue length during peak windows
        peak_window_total_waiting_time (float): Total waiting time during peak windows
        peak_window_total_duration (float): Total duration of all peak windows (minutes)
        in_peak_window (bool): Flag indicating if currently in a peak window
        peak_window_start_time (float): Start time of current peak window (minutes)
        peak_window_start_requests (int): Total requests at start of peak window
        peak_window_start_abandoned (int): Total abandoned at start of peak window
        peak_window_start_waiting_time (float): Total waiting time at start of peak window
        
        -- Time-Weighted Statistics (for rates/utilization) --
        area_under_available_cars (float): Integral of available car count over time
        area_under_used_cars (float): Integral of cars in use over time
        area_under_used_station (float): Integral of occupied station spots over time
        area_under_queue_length (float): Integral of waiting queue length over time
        last_event_time (float): Time of last event processed (minutes)
        last_available_car_count (int): Number of available cars at last event
        last_used_car_count (int): Number of cars in use at last event
        last_used_station_spots (int): Number of occupied station spots at last event
        last_queue_length (int): Current waiting queue length
    """
    def __init__(self, cars: list[Car], stations: list[ChargingStation]):
        """
        Initialize the car sharing system with cars and stations.
        
        Sets up all tracking dictionaries and statistical accumulators.
        Initializes time-weighted statistics with proper starting values.
        
        Args:
            cars: List of all cars in the system
            stations: List of all charging stations
        """
        self.cars = cars
        self.stations = stations

        # Initialize statistics
        self.total_user_requests: int = 0
        self.total_abandoned_requests: int = 0
        self.total_waiting_time: float = 0.0
        self.total_trip_distance: float = 0.0

        # Relocation tracking
        self.last_relocation_time: float = 0.0

        # Client tracking
        self.waiting_users: dict[int, User] = {}
        self.active_users: dict[int, User] = {}
        self.abandoned_users: dict[int, User] = {}

        # Car tracking
        self.charging_cars: dict[int, Car] = {car.car_id: car for car in cars if car.charging}
        self.dispersed_cars: dict[int, Car] = {}

        # Sample statistics
        self.sampled_num_serving_users: list[tuple[float, int]] = []  # (time, num_serving_users)
        self.sampled_num_total_users: list[tuple[float, int]] = []  # (time, num_total_users)
        self.num_samples: int = 0

        # Peak window statistics tracking (windows with max arrival rate)
        self.peak_window_stats: list[dict] = []  # Statistics for each peak window
        self.peak_window_area_available: float = 0.0  # Area under available cars curve for peak windows
        self.peak_window_area_used: float = 0.0  # Area under used cars curve for peak windows
        self.peak_window_area_station: float = 0.0  # Area under station utilization curve for peak windows
        self.peak_window_total_waiting_time: float = 0.0  # Total waiting time during peak windows
        self.peak_window_total_duration: float = 0.0  # Total duration of all peak windows
        self.in_peak_window: bool = False  # Flag indicating if currently in a peak window
        self.peak_window_start_time: float = 0.0  # Start time of current peak window
        self.peak_window_start_requests: int = 0  # Total requests at start of peak window
        self.peak_window_start_abandoned: int = 0  # Total abandoned at start of peak window
        self.peak_window_start_waiting_time: float = 0.0  # Total waiting time at start of peak window

        # Initialize time-weighted statistics
        self.area_under_available_cars: float = 0.0
        self.area_under_used_cars: float = 0.0
        self.area_under_used_station: float = 0.0
        self.area_under_queue_length: float = 0.0  # Area under waiting queue length curve
        self.peak_window_area_queue: float = 0.0  # Area under queue length during peak windows
        self.last_event_time: float = 0.0
        self.last_available_car_count: int = sum(1 for c in cars if c.is_available())
        self.last_used_car_count: int = 0
        self.last_used_station_spots: int = len(cars)
        self.last_queue_length: int = 0  # Current waiting queue length

    def update_time_weighted_statistics(self, new_time: float):
        """
        Update time-weighted statistics using integration.
        
        Computes the area under curves for available cars, used cars, and station
        utilization by multiplying last recorded values by time elapsed since last event.
        This enables calculation of time-averaged rates and utilization metrics.
        
        Must be called before processing each event to accurately capture system state.
        
        Args:
            new_time: Current simulation time (minutes)
        """
        start = self.last_event_time
        time_delta = new_time - start
        if time_delta > 0:
            self.area_under_available_cars += self.last_available_car_count * time_delta
            self.area_under_used_cars += self.last_used_car_count * time_delta
            self.area_under_used_station += self.last_used_station_spots * time_delta
            self.area_under_queue_length += self.last_queue_length * time_delta
            
            # Update peak window statistics if in peak window
            if self.in_peak_window:
                self.peak_window_area_available += self.last_available_car_count * time_delta
                self.peak_window_area_used += self.last_used_car_count * time_delta
                self.peak_window_area_station += self.last_used_station_spots * time_delta
                self.peak_window_area_queue += self.last_queue_length * time_delta

        self.last_event_time = new_time

    def sample_statistics(self, current_time: float):
        """
        Take a snapshot sample of current system state for time-series analysis.
        
        Records instantaneous values of:
        - Number of users currently being served
        - Total number of active users in the system
        
        These samples are used for transient detection and trend analysis.
        
        Args:
            current_time: Current simulation time (minutes)
        """
        self.num_samples += 1
        # Count serving users: cars with assigned_user_id that are not relocating
        num_serving_users = sum(1 for car in self.cars 
                               if car.assigned_user_id is not None 
                               and not car.is_relocating())
        self.sampled_num_serving_users.append((current_time, num_serving_users))
        self.sampled_num_total_users.append((current_time, len(self.active_users)))

    def update_speed(self, new_speed: float):
        """
        Update the speed of all cars in the system simultaneously.
        
        Used to implement dynamic speed adjustments (e.g., congestion modeling).
        
        Args:
            new_speed: New speed value in km/h to apply to all cars
        """
        for car in self.cars:
            car.speed = new_speed
    
    def compute_congestion_speed(self, base_speed: float, congestion_enabled: bool = True, 
                                congestion_factor: float = 0.5) -> float:
        """
        Compute speed adjusted for traffic congestion based on active cars.
        
        Args:
            base_speed: Base speed in km/h (no congestion)
            congestion_enabled: If False, return base_speed (no congestion effect)
            congestion_factor: Maximum speed reduction factor (0.0 to 1.0)
                             0.5 means speed can drop to 50% of base when all cars active
        
        Returns:
            Adjusted speed in km/h
        """
        if not congestion_enabled:
            return base_speed
        
        active_cars = self.last_used_car_count
        total_cars = len(self.cars)
        
        if total_cars == 0:
            return base_speed
        
        # Congestion increases linearly with active car ratio
        # speed = base_speed * (1 - congestion_factor * active_ratio)
        active_ratio = active_cars / total_cars
        speed_multiplier = 1.0 - (congestion_factor * active_ratio)
        
        return base_speed * speed_multiplier

    def add_trip_distance(self, distance: float):
        """
        Accumulate total trip distance for completed trips.
        
        Args:
            distance: Distance of completed trip in grid units
        """
        self.total_trip_distance += distance

    def add_waiting_time(self, user: User):
        """
        Accumulate total waiting time across all served users.
        
        Args:
            user: User whose waiting time should be added
        """
        self.total_waiting_time += user.waiting_time()

    def add_charging_car(self, car: Car):
        """
        Register a car as currently charging at a station.
        
        Args:
            car: Car to add to charging cars dictionary
        """
        self.charging_cars[car.car_id] = car

    def remove_charging_car(self, car: Car):
        """
        Remove a car from the charging cars registry (e.g., when leaving station).
        
        Args:
            car: Car to remove from charging cars dictionary
        """
        if car.car_id in self.charging_cars:
            del self.charging_cars[car.car_id]

    def charge_cars(self, interval: float):
        """
        Charge all cars currently at charging stations for given time interval.
        
        Called at each event to advance charging progress for all plugged-in cars.
        
        Args:
            interval: Time interval for charging in minutes
        """
        for car in self.charging_cars.values():
            car.charge(interval)

    def add_dispersed_car(self, car: Car):
        """
        Register a car as dispersed (not charging and with low autonomy).
        
        Dispersed cars are prioritized for relocation to bring them back into service.
        
        Args:
            car: Car to add to dispersed cars dictionary
        """
        self.dispersed_cars[car.car_id] = car

    def remove_dispersed_car(self, car: Car):
        """
        Remove a car from dispersed cars registry (e.g., after relocation).
        
        Args:
            car: Car to remove from dispersed cars dictionary
        """
        if car.car_id in self.dispersed_cars:
            del self.dispersed_cars[car.car_id]

    def get_dispersed_cars(self) -> list[Car]:
        """
        Get list of all currently dispersed cars.
        
        Returns:
            List of dispersed Car objects
        """
        return list(self.dispersed_cars.values())

    def add_active_user(self, user: User):
        """
        Register a user as active in the system (from request to trip completion).
        
        Args:
            user: User to add to active users dictionary
        """
        self.active_users[user.user_id] = user

    def remove_active_user(self, user: User):
        """
        Remove a user from active users (after trip completion or abandonment).
        
        Args:
            user: User to remove from active users dictionary
        """
        if user.user_id in self.active_users:
            del self.active_users[user.user_id]

    def get_active_user(self, user_id: int) -> Optional[User]:
        """
        Retrieve an active user by ID.
        
        Args:
            user_id: ID of user to retrieve
            
        Returns:
            User object if found, None otherwise
        """
        return self.active_users.get(user_id, None)

    def add_client_to_waiting(self, user: User):
        """
        Add a user to the waiting queue (no car immediately available).
        
        Args:
            user: User to add to waiting users dictionary
        """
        self.waiting_users[user.user_id] = user
        self.last_queue_length = len(self.waiting_users)

    def remove_client_from_waiting(self, user: User):
        """
        Remove a user from waiting queue (assigned car or abandoned).
        
        Args:
            user: User to remove from waiting users dictionary
        """
        if user.user_id in self.waiting_users:
            del self.waiting_users[user.user_id]
            self.last_queue_length = len(self.waiting_users)
    
    def get_waiting_clients(self) -> list[User]:
        """
        Get list of all users currently waiting for car assignment.
        
        Returns:
            List of waiting User objects
        """
        return list(self.waiting_users.values())
    
    def update_available_car_count(self):
        """
        Recompute and update the count of currently available cars.

        Must be called after any event that changes car availability status.
        Used for time-weighted statistics calculation.
        """
        self.last_available_car_count = sum(1 for c in self.cars if c.is_available())

    def update_used_car_count(self):
        """
        Recompute and update the count of cars currently assigned to users.
        
        Must be called after any event that changes car assignment status.
        Used for time-weighted statistics calculation.
        """
        self.last_used_car_count = sum(1 for c in self.cars if c.assigned_user_id is not None)

    def user_requested_car(self):
        """
        Increment total user requests counter when new request arrives.
        """
        self.total_user_requests += 1

    def user_abandoned(self, user: User):
        """
        Record a user abandonment (waited too long without car assignment).
        
        Args:
            user: User who abandoned their request
        """
        self.total_abandoned_requests += 1
        self.abandoned_users[user.user_id] = user

    def occupy_station_spot(self):
        """
        Increment station spot usage counter when car parks at station.

        Must be called whenever a car parks at a station.
        """
        self.last_used_station_spots += 1

    def release_station_spot(self):
        """
        Decrement station spot usage counter when car leaves station.

        Must be called whenever a car departs from a station.
        """
        self.last_used_station_spots -= 1

    def relocation_performed(self, time: float):
        """
        Record a relocation event in system statistics.
        
        Args:
            time: Simulation time when relocation occurred (minutes)
        """
        self.last_relocation_time = time
    
    def get_num_serving_users_over_time(self) -> tuple[list[float], list[int]]:
        """
        Extract time series of users currently being served (in cars).
        
        Returns:
            Tuple of (times, num_serving_users) where times are floats and counts are ints
        """
        times = [t for t, _ in self.sampled_num_serving_users]
        num_serving = [n for _, n in self.sampled_num_serving_users]
        return times, num_serving
    
    def get_num_total_users_over_time(self) -> tuple[list[float], list[int]]:
        """
        Extract time series of total active users in the system.
        
        Returns:
            Tuple of (times, num_total_users) where times are floats and counts are ints
        """
        times = [t for t, _ in self.sampled_num_total_users]
        num_total = [n for _, n in self.sampled_num_total_users]
        return times, num_total
    
    def start_peak_window(self, current_time: float):
        """
        Mark the start of a peak arrival rate window for statistics tracking.
        
        Records baseline statistics at the start of the peak window to enable
        computation of metrics specific to high-demand periods.
        
        Args:
            current_time: Current simulation time (minutes)
        """
        if not self.in_peak_window:
            self.in_peak_window = True
            self.peak_window_start_time = current_time
            self.peak_window_start_requests = self.total_user_requests
            self.peak_window_start_abandoned = self.total_abandoned_requests
            self.peak_window_start_waiting_time = self.total_waiting_time
    
    def end_peak_window(self, current_time: float):
        """
        Mark the end of a peak arrival rate window and record statistics.
        
        Computes and stores all relevant metrics for the completed peak window including:
        - Vehicle availability rate
        - Vehicle utilization rate
        - Average waiting time
        - Charging station utilization
        - Number of requests served and abandoned
        
        Args:
            current_time: Current simulation time (minutes)
        """
        if self.in_peak_window:
            self.in_peak_window = False
            window_duration = current_time - self.peak_window_start_time
            
            if window_duration > 0:
                # Compute metrics for this peak window
                window_requests = self.total_user_requests - self.peak_window_start_requests
                window_abandoned = self.total_abandoned_requests - self.peak_window_start_abandoned
                window_served = window_requests - window_abandoned
                window_waiting_time = self.total_waiting_time - self.peak_window_start_waiting_time
                
                # Time-weighted averages for this window
                avg_available = self.peak_window_area_available / window_duration
                avg_used = self.peak_window_area_used / window_duration
                avg_station_occupied = self.peak_window_area_station / window_duration
                avg_queue_length = self.peak_window_area_queue / window_duration
                
                total_cars = len(self.cars)
                total_station_spots = sum(station.capacity for station in self.stations)
                
                # Compute rates and percentages
                availability_rate = (avg_available / total_cars * 100) if total_cars > 0 else 0.0
                utilization_rate = (avg_used / total_cars * 100) if total_cars > 0 else 0.0
                station_utilization = (avg_station_occupied / total_station_spots * 100) if total_station_spots > 0 else 0.0
                avg_waiting_time = (window_waiting_time / window_served) if window_served > 0 else 0.0
                abandonment_rate = (window_abandoned / window_requests * 100) if window_requests > 0 else 0.0
                
                # Store statistics for this peak window
                peak_stats = {
                    'start_time': self.peak_window_start_time,
                    'end_time': current_time,
                    'duration': window_duration,
                    'requests': window_requests,
                    'served': window_served,
                    'abandoned': window_abandoned,
                    'abandonment_rate': abandonment_rate,
                    'availability_rate': availability_rate,
                    'utilization_rate': utilization_rate,
                    'station_utilization': station_utilization,
                    'avg_waiting_time': avg_waiting_time,
                    'total_waiting_time': window_waiting_time,
                    'avg_queue_length': avg_queue_length
                }
                self.peak_window_stats.append(peak_stats)
                
                # Update cumulative peak window duration
                self.peak_window_total_duration += window_duration
                
                # Reset peak window accumulators for next peak window
                self.peak_window_area_available = 0.0
                self.peak_window_area_used = 0.0
                self.peak_window_area_station = 0.0
                self.peak_window_area_queue = 0.0
    
    def get_peak_window_statistics(self) -> dict:
        """
        Compute aggregate statistics across all peak arrival rate windows.
        
        Returns:
            Dictionary containing:
            - num_peak_windows: Number of peak windows observed
            - total_peak_duration: Total time spent in peak windows (minutes)
            - avg_availability_rate: Average vehicle availability during peaks (%)
            - avg_utilization_rate: Average vehicle utilization during peaks (%)
            - avg_station_utilization: Average charging station usage during peaks (%)
            - avg_waiting_time: Average user waiting time during peaks (minutes)
            - avg_abandonment_rate: Average abandonment rate during peaks (%)
            - total_peak_requests: Total requests during peak windows
            - total_peak_served: Total requests served during peak windows
            - total_peak_abandoned: Total requests abandoned during peak windows
        """
        if not self.peak_window_stats:
            return {
                'num_peak_windows': 0,
                'total_peak_duration': 0.0,
                'avg_availability_rate': 0.0,
                'avg_utilization_rate': 0.0,
                'avg_station_utilization': 0.0,
                'avg_waiting_time': 0.0,
                'avg_abandonment_rate': 0.0,
                'total_peak_requests': 0,
                'total_peak_served': 0,
                'total_peak_abandoned': 0,
            }
        
        # Aggregate statistics across all peak windows
        total_peak_requests = sum(w['requests'] for w in self.peak_window_stats)
        total_peak_served = sum(w['served'] for w in self.peak_window_stats)
        total_peak_abandoned = sum(w['abandoned'] for w in self.peak_window_stats)
        
        # Weighted averages by duration
        total_duration = sum(w['duration'] for w in self.peak_window_stats)
        if total_duration > 0:
            avg_availability = sum(w['availability_rate'] * w['duration'] for w in self.peak_window_stats) / total_duration
            avg_utilization = sum(w['utilization_rate'] * w['duration'] for w in self.peak_window_stats) / total_duration
            avg_station_util = sum(w['station_utilization'] * w['duration'] for w in self.peak_window_stats) / total_duration
            avg_abandonment = sum(w['abandonment_rate'] * w['duration'] for w in self.peak_window_stats) / total_duration
            avg_queue_length = sum(w['avg_queue_length'] * w['duration'] for w in self.peak_window_stats) / total_duration
        else:
            avg_availability = 0.0
            avg_utilization = 0.0
            avg_station_util = 0.0
            avg_abandonment = 0.0
            avg_queue_length = 0.0
        
        # Weighted average waiting time
        total_waiting_time = sum(w['total_waiting_time'] for w in self.peak_window_stats)
        avg_waiting = (total_waiting_time / total_peak_served) if total_peak_served > 0 else 0.0
        
        return {
            'num_peak_windows': len(self.peak_window_stats),
            'total_peak_duration': total_duration,
            'avg_availability_rate': avg_availability,
            'avg_utilization_rate': avg_utilization,
            'avg_station_utilization': avg_station_util,
            'avg_waiting_time': avg_waiting,
            'avg_abandonment_rate': avg_abandonment,
            'avg_queue_length': avg_queue_length,
            'total_peak_requests': total_peak_requests,
            'total_peak_served': total_peak_served,
            'total_peak_abandoned': total_peak_abandoned,
        }


# ------------------------------------------------------------------------------
# EVENT HANDLERS
# ------------------------------------------------------------------------------
class EventHandler:
    """
    Class containing static methods to handle different event types in the simulation.
    
    This class implements the core discrete-event simulation logic, processing each
    event type and updating system state accordingly. All methods are static as they
    operate on passed system state rather than instance variables.
    
    Event handling methods include:
    - handle_user_request: Process new user car requests
    - handle_user_abandon: Process user abandonment after timeout
    - handle_car_parking: Process car arrival at destination/station
    - handle_car_available: Process car becoming available for new assignment
    - handle_car_relocate: Process system-initiated car redistribution
    
    Helper methods provide common functionality:
    - Distance/time calculations
    - Nearest car/station finding
    - Car assignment logic
    - Single car relocation logic
    """
    
    @staticmethod
    def compute_travel_time(distance: float, speed: float, map_unit_km: float = 0.1) -> float:
        """
        Compute travel time in minutes from grid distance.
        
        Converts grid-based distance to kilometers, then calculates travel time
        based on speed and converts to minutes.
        
        Args:
            distance: Distance in grid units
            speed: Speed in km/h
            map_unit_km: Conversion factor from grid units to kilometers (default: 0.1)
            
        Returns:
            Travel time in minutes
        """
        distance_km = distance * map_unit_km
        return (distance_km / speed) * 60.0
    
    @staticmethod
    def compute_distance(loc1: tuple, loc2: tuple) -> float:
        """
        Compute Euclidean distance between two locations on the grid.
        
        Args:
            loc1: First location as (x, y) tuple
            loc2: Second location as (x, y) tuple
            
        Returns:
            Euclidean distance in grid units
        """
        return ((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2) ** 0.5
    
    @staticmethod
    def find_nearest_available_car(location: tuple, cars: list[Car], max_distance: float) -> Optional[tuple[Car, float]]:
        """
        Find the nearest available car within maximum pickup distance.
        
        Searches all cars for available ones within range and returns the closest.
        
        Args:
            location: Target location (x, y) where car is needed
            cars: List of all cars in system
            max_distance: Maximum acceptable distance in grid units
            
        Returns:
            Tuple of (car, distance) if suitable car found, None otherwise
        """
        available_cars = []
        for car in cars:
            if car.is_available():
                dist = EventHandler.compute_distance(location, car.location)
                if dist <= max_distance:
                    available_cars.append((car, dist))
        
        if available_cars:
            return min(available_cars, key=lambda x: x[1])
        return None
    
    @staticmethod
    def find_nearest_station(location: tuple, stations: list[ChargingStation], 
                           exclude_full: bool = True) -> Optional[tuple[ChargingStation, float]]:
        """
        Find the nearest charging station to a location.
        
        Args:
            location: Target location (x, y) to find station near
            stations: List of all charging stations
            exclude_full: If True, only consider stations with available spots
            
        Returns:
            Tuple of (station, distance) if suitable station found, None otherwise
        """
        candidates = []
        for station in stations:
            if exclude_full and station.is_full():
                continue
            dist = EventHandler.compute_distance(location, station.location)
            candidates.append((station, dist))
        
        if candidates:
            return min(candidates, key=lambda x: x[1])
        return None
    
    @staticmethod
    def _assign_car_to_user(car: Car, user: User, stations: list[ChargingStation], 
                           car_sharing_system: CarSharingSystem, future_event_set: FutureEventSet,
                           current_time: float, map_unit_km: float) -> bool:
        """
        Assign a car to a user and schedule the trip completion event.
        
        This helper performs the complete car assignment workflow:
        1. Remove car from charging/station if applicable
        2. Assign car to user and user to car
        3. Update availability and usage counters
        4. Record waiting time
        5. Calculate and record trip distance
        6. Schedule CAR_PARKING event for trip completion
        
        Updates car_sharing_system statistics including:
        - last_available_car_count (via update_available_car_count)
        - last_used_car_count (via update_used_car_count)
        - last_used_station_spots (via release_station_spot if car was at station)
        - total_trip_distance
        - total_waiting_time
        
        Args:
            car: Car to assign to user
            user: User requesting the car
            stations: List of all stations (to remove car from station if parked)
            car_sharing_system: System object for statistics updates
            future_event_set: FES to schedule trip completion event
            current_time: Current simulation time (minutes)
            map_unit_km: Conversion factor from grid units to km
        
        Returns:
            True if assignment successful
        """
        # Remove car from charging/station if necessary
        if car.charging:
            car_sharing_system.remove_charging_car(car)
        if car.station_id is not None:
            station = stations[car.station_id]
            station.remove_car(car.car_id)
            car.unpark_from_station()
            car_sharing_system.release_station_spot()
        
        # Assign car to user
        car.assign_user(user.user_id)
        user.assign_car(car.car_id, current_time)
        car_sharing_system.update_available_car_count()
        car_sharing_system.update_used_car_count()
        car_sharing_system.add_waiting_time(user)
        
        # Schedule trip completion
        trip_distance = EventHandler.compute_distance(user.location, user.destination)
        car.drive(trip_distance, user.destination)
        car_sharing_system.add_trip_distance(trip_distance)
        
        travel_time = EventHandler.compute_travel_time(trip_distance, car.speed, map_unit_km)
        car_parking_event = Event(EventType.CAR_PARKING, current_time + travel_time, 
                                 user_id=user.user_id, car_id=car.car_id)
        future_event_set.schedule(car_parking_event)
        return True
    
    @staticmethod
    def _schedule_next_user(user: User, car_sharing_system: CarSharingSystem, 
                           future_event_set: FutureEventSet, current_time: float,
                           arrival_rate: float, min_trip_distance: float,
                           demand_origin_map: np.ndarray, demand_dest_map: np.ndarray):
        """
        Generate and schedule the next user request event.
        
        Samples inter-arrival time from exponential distribution based on arrival_rate.
        Samples origin and destination from probability maps, ensuring minimum trip distance.
        Creates new user and schedules USER_REQUEST event.
        
        Args:
            user: Current user (used to get next user_id and parameters)
            car_sharing_system: System to register new active user
            future_event_set: FES to schedule next request event
            current_time: Current simulation time (minutes)
            arrival_rate: Current arrival rate (requests per minute)
            min_trip_distance: Minimum required trip distance in grid units
            demand_origin_map: Probability map for sampling trip origins
            demand_dest_map: Probability map for sampling trip destinations
        """
        inter_arrival_time = np.random.exponential(1 / arrival_rate)
        next_request_time = current_time + inter_arrival_time
        next_user_id = user.user_id + 1
        
        # Sample origin and destination ensuring min trip distance
        next_user_location = MappingUtilities.get_random_location_from_prob_map(demand_origin_map)
        travel_distance = 0.0
        while travel_distance < min_trip_distance:
            next_user_destination = MappingUtilities.get_random_location_from_prob_map(demand_dest_map)
            travel_distance = EventHandler.compute_distance(next_user_location, next_user_destination)
        
        next_user = User(next_user_id, next_user_location, next_user_destination, 
                        user.max_waiting_time, user.max_pickup_distance, next_request_time)
        car_sharing_system.add_active_user(next_user)
        user_request_event = Event(EventType.USER_REQUEST, next_request_time, user_id=next_user_id)
        future_event_set.schedule(user_request_event)
    
    @staticmethod
    def handle_user_request(event: Event, cars: list[Car], car_sharing_system: CarSharingSystem, 
                            stations: list[ChargingStation], future_event_set: FutureEventSet, current_time: float,
                            arrival_rate: float, min_trip_distance: float, demand_origin_map: np.ndarray, 
                            demand_dest_map: np.ndarray, map_unit_km: float = 0.1) -> bool:
        """
        Handle USER_REQUEST event when a user requests a car.
        
        Processing flow:
        1. Retrieve user from active users
        2. Increment total_user_requests counter
        3. Search for nearest available car within max_pickup_distance
        4. If car found: assign car to user immediately
        5. If no car: add user to waiting queue and schedule USER_ABANDON event
        6. Schedule next USER_REQUEST event
        
        Statistics Updated:
        - total_user_requests (incremented)
        - If car assigned: all assignment-related statistics
        - If queued: waiting_users dictionary updated
        
        Args:
            event: USER_REQUEST event being processed
            cars: List of all cars in system
            car_sharing_system: System object for state and statistics
            stations: List of all stations
            future_event_set: FES for scheduling new events
            current_time: Current simulation time (minutes)
            arrival_rate: Current arrival rate for next user
            min_trip_distance: Minimum trip distance for next user
            demand_origin_map: Origin probability map for next user
            demand_dest_map: Destination probability map for next user
            map_unit_km: Grid to km conversion factor
            
        Returns:
            True if user was immediately assigned a car, False if queued
        """
        user = car_sharing_system.get_active_user(event.user_id)
        if user is None:
            return False
        
        user.request_car()
        car_sharing_system.user_requested_car()
        
        # Try to find and assign nearest available car
        result = EventHandler.find_nearest_available_car(user.location, cars, user.max_pickup_distance)
        
        if result is not None:
            car, _ = result
            assigned = EventHandler._assign_car_to_user(car, user, stations, car_sharing_system, 
                                                       future_event_set, current_time, map_unit_km)
        else:
            # No car available - add to waiting queue
            user.put_in_queue()
            car_sharing_system.add_client_to_waiting(user)
            abandon_event = Event(EventType.USER_ABANDON, current_time + user.max_waiting_time, 
                                user_id=user.user_id)
            future_event_set.schedule(abandon_event)
            assigned = False
        
        # Schedule next user request
        EventHandler._schedule_next_user(user, car_sharing_system, future_event_set, current_time,
                                        arrival_rate, min_trip_distance, demand_origin_map, demand_dest_map)
        
        return assigned
    
    @staticmethod
    def handle_user_abandon(event: Event, car_sharing_system: CarSharingSystem) -> bool:
        """
        Handle USER_ABANDON event when a user gives up waiting.
        
        Occurs when user has been waiting for max_waiting_time without car assignment.
        
        Processing flow:
        1. Retrieve user from active users
        2. Check if user is still waiting
        3. Mark user as abandoned
        4. Remove from waiting queue and active users
        5. Increment abandonment counter

        Statistics Updated:
        - total_abandoned_requests (incremented)
        - abandoned_users (user added)
        - waiting_users (user removed)
        - active_users (user removed)
        
        Args:
            event: USER_ABANDON event being processed
            car_sharing_system: System object for state and statistics
            
        Returns:
            True if user was waiting and successfully abandoned, False otherwise
        """
        user: User = car_sharing_system.get_active_user(event.user_id)
        if user is None:
            return False
        if user.is_waiting():
            user.abandon_request()
            car_sharing_system.user_abandoned(user=user)
            car_sharing_system.remove_client_from_waiting(user)
            car_sharing_system.remove_active_user(user)
            return True
        return False
    
    @staticmethod
    def _park_car_at_station(car: Car, station: ChargingStation, car_sharing_system: CarSharingSystem, relocate: bool = False):
        """
        Park a car at a charging station and start charging.
        
        Helper method to handle parking logic whether from regular
        trip completion or relocation.
        
        Processing:
        1. If not relocate: reserve spot at station (relocate already reserved)
        2. Associate car with station_id
        3. Increment occupied station spots counter
        4. Add car to charging cars registry
        5. Enable charging flag
        6. Update car location to station location
        
        Statistics Updated:
        - last_used_station_spots (via occupy_station_spot)
        - charging_cars dictionary (car added)
        
        Args:
            car: Car to park at station
            station: Station where car will park
            car_sharing_system: System for statistics updates
            relocate: If True, spot was already reserved during relocation scheduling
        """
        if not relocate:
            station.park_car(car.car_id)
        car.park_in_station(station.station_id)
        car_sharing_system.occupy_station_spot()
        car_sharing_system.add_charging_car(car)
        car.charging = True
        car.location = station.location
    
    @staticmethod
    def handle_car_parking(event: Event, cars: list[Car], car_sharing_system: CarSharingSystem,
                           stations: list[ChargingStation], future_event_set: FutureEventSet, current_time: float) -> bool:
        """
        Handle CAR_PARKING event when car completes trip or relocation.
        
        Two distinct processing paths:
        
        A) Regular Trip Completion (car.is_relocating() == False):
           1. Release user from car
           2. Mark user as served and remove from active users
           3. Update used car count
           4. Try to park at nearest station within max_destination_charging_distance
           5. If parked successfully: start charging and schedule CAR_AVAILABLE when charged
           6. If car available with sufficient autonomy: schedule immediate CAR_AVAILABLE
           7. If car unavailable (low autonomy, no nearby station): mark as dispersed
           
        B) Relocation Completion (car.is_relocating() == True):
           1. Stop relocation mode
           2. Park at destination station (spot already reserved)
           3. Start charging
           4. Schedule CAR_AVAILABLE when charged to minimum
        
        Statistics Updated:
        - last_used_car_count (via update_used_car_count)
        - last_available_car_count (via update_available_car_count if dispersed)
        - last_used_station_spots (if parked at station)
        - charging_cars (if charging)
        - dispersed_cars (if stranded)
        - active_users (user removed if trip completion)
        
        Args:
            event: CAR_PARKING event being processed
            cars: List of all cars
            car_sharing_system: System for state and statistics
            stations: List of all stations
            future_event_set: FES for scheduling CAR_AVAILABLE event
            current_time: Current simulation time (minutes)
            
        Returns:
            True if parking processed successfully
        """
        car: Car = cars[event.car_id]
        
        if not car.is_relocating():
            # Regular trip completion
            car.release_user()
            user: User = car_sharing_system.get_active_user(event.user_id)
            if user is not None:
                user.resign_car()
                car_sharing_system.remove_active_user(user)
            car_sharing_system.update_used_car_count()
            
            # Try to park at nearest station
            result = EventHandler.find_nearest_station(car.location, stations, exclude_full=True)
            if result is not None:
                nearest_station, min_distance = result
                if min_distance <= car.max_destination_charging_distance:
                    EventHandler._park_car_at_station(car, nearest_station, car_sharing_system)
            
            # Schedule car available event
            if car.is_available():
                car_available_event = Event(EventType.CAR_AVAILABLE, current_time, car_id=car.car_id)
                future_event_set.schedule(car_available_event)
            elif car.charging:
                time_to_min_autonomy = max(0.0, (car.min_autonomy - car.autonomy) / car.charging_rate)
                car_available_event = Event(EventType.CAR_AVAILABLE, current_time + time_to_min_autonomy, 
                                           car_id=car.car_id)
                future_event_set.schedule(car_available_event)
            else:
                # Car is dispersed (not at station, low autonomy, not available)
                car_sharing_system.add_dispersed_car(car)
                # Update available car count since car status may have changed
                car_sharing_system.update_available_car_count()
            return True
        else:
            # Relocation parking
            station = stations[car.get_relocating_station_id()]
            car.stop_relocating()
            EventHandler._park_car_at_station(car, station, car_sharing_system, relocate=True)
            
            time_to_min_autonomy = max(0.0, (car.min_autonomy - car.autonomy) / car.charging_rate)
            car_available_event = Event(EventType.CAR_AVAILABLE, current_time + time_to_min_autonomy, 
                                       car_id=car.car_id)
            future_event_set.schedule(car_available_event)
            return True
    
    @staticmethod
    def _find_nearest_waiting_user(car_location: tuple, waiting_clients: list[User]) -> Optional[tuple[User, float]]:
        """
        Find the nearest waiting user to a car location.
        
        Searches through all waiting users and identifies the one closest to the given
        car location based on Euclidean distance.
        
        Args:
            car_location: (x, y) coordinates of the car
            waiting_clients: List of users currently waiting for car assignment
            
        Returns:
            Tuple of (user, distance) if waiting users exist, None otherwise
        """
        if not waiting_clients:
            return None
        
        distance_to_users = []
        for user in waiting_clients:
            dist = EventHandler.compute_distance(car_location, user.location)
            distance_to_users.append((user, dist))
        
        return min(distance_to_users, key=lambda x: x[1])
    
    @staticmethod
    def handle_car_available(event: Event, cars: list[Car], stations: list[ChargingStation], 
                            car_sharing_system: CarSharingSystem, future_event_set: FutureEventSet, 
                            current_time: float, map_unit_km: float = 0.1) -> bool:
        """
        Handle CAR_AVAILABLE event when a car becomes available for assignment.
        
        Triggered when a car finishes charging to minimum autonomy threshold or becomes
        available through other means. Attempts to match the newly available car with
        the nearest waiting user.
        
        Processing flow:
        1. Verify car is actually available
        2. Update available car count
        3. Search for nearest waiting user
        4. Check if user is within car's service range (max_pickup_distance)
        5. If match found: assign car to user and schedule trip
        6. Remove user from waiting queue
        7. Cancel pending USER_ABANDON event for assigned user
        
        Statistics Updated:
        - last_available_car_count (via update_available_car_count)
        - All assignment-related statistics (if user matched)
        - waiting_users (user removed if matched)
        
        Args:
            event: CAR_AVAILABLE event being processed
            cars: List of all cars in system
            stations: List of all charging stations
            car_sharing_system: System for state and statistics
            future_event_set: FES for scheduling trip completion and cancelling abandon events
            current_time: Current simulation time (minutes)
            map_unit_km: Grid to km conversion factor
            
        Returns:
            True if car was assigned to a waiting user, False otherwise
        """
        car: Car = cars[event.car_id]
        if not car.is_available():
            return False
        
        car_sharing_system.update_available_car_count()
        waiting_clients = car_sharing_system.get_waiting_clients()
        
        result = EventHandler._find_nearest_waiting_user(car.location, waiting_clients)
        if result is None:
            return False
        
        user, min_distance = result
        if min_distance > user.max_pickup_distance:
            return False
        
        # Assign car to waiting user
        assigned = EventHandler._assign_car_to_user(car, user, stations, car_sharing_system, 
                                                   future_event_set, current_time, map_unit_km)
        if not assigned:
            return False
        
        car_sharing_system.remove_client_from_waiting(user)
        
        # Cancel USER_ABANDON event if scheduled
        for evt in future_event_set.events.queue:
            if evt.event_type == EventType.USER_ABANDON and evt.user_id == user.user_id:
                evt.cancel()
        return True
    
    @staticmethod
    def handle_car_relocate(event: Event, cars: list[Car], stations: list[ChargingStation], car_sharing_system: CarSharingSystem,
                             future_event_set: FutureEventSet, total_cars_to_relocate: int, target_prob_map: np.ndarray,
                             map_unit_km: float = 0.1) -> bool:
        """
        Handle CAR_RELOCATE event to redistribute cars across the system.
        
        Implements a two-phase relocation strategy to improve system efficiency:
        
        Phase 1: Priority Relocation of Dispersed Cars
        - Relocate all dispersed cars (stranded after trips with low autonomy)
        - Target low-utilization charging stations to bring cars back into service
        
        Phase 2: Demand-Based Relocation
        - If relocation quota not reached, relocate additional available cars
        - Prioritize cars farthest from high-demand areas (based on target_prob_map)
        - Move cars from low-demand to high-demand regions
        - Target low-utilization stations to balance station load
        
        For each relocated car:
        1. Mark as unavailable and remove from current station
        2. Identify target station (low-utilization)
        3. Mark car as relocating and drive to target
        4. Reserve spot at target station
        5. Schedule CAR_PARKING event for relocation completion
        
        Statistics Updated:
        - last_available_car_count (cars marked unavailable)
        - last_used_station_spots (removed from source, reserved at target)
        - dispersed_cars (dispersed cars removed)
        
        Args:
            event: CAR_RELOCATE event being processed
            cars: List of all cars in system
            stations: List of all charging stations
            car_sharing_system: System for state and statistics
            future_event_set: FES for scheduling CAR_PARKING events
            total_cars_to_relocate: Target number of cars to relocate (quota)
            target_prob_map: Probability map representing demand distribution
            map_unit_km: Grid to km conversion factor
            
        Returns:
            True if at least one car was relocated, False otherwise
        """
        relocated_cars = 0
        
        # Step 1: Relocate all dispersed cars first
        dispersed_cars = list(car_sharing_system.get_dispersed_cars())
        for car in dispersed_cars:
            target_station = EventHandler._find_low_utilization_station(stations)
            if target_station is None:
                continue
                
            relocated = EventHandler._relocate_single_car(
                car, stations, car_sharing_system, future_event_set, event.time,
                target_prob_map, remove_from_dispersed=True, target_station=target_station,
                map_unit_km=map_unit_km
            )
            relocated_cars += relocated
        
        # Step 2: Relocate cars most distant from demand centers
        remaining = total_cars_to_relocate - relocated_cars
        if remaining > 0:
            available_cars_with_distance = []
            for car in cars:
                if car.is_available():
                    distance = EventHandler._compute_distance_from_demand(car.location, target_prob_map)
                    available_cars_with_distance.append((car, distance))
            
            available_cars_with_distance.sort(key=lambda x: x[1], reverse=True)
            
            for car, _ in available_cars_with_distance[:remaining]:
                target_station = EventHandler._find_low_utilization_station(stations)
                if target_station is None:
                    break
                
                relocated = EventHandler._relocate_single_car(
                    car, stations, car_sharing_system, future_event_set, event.time,
                    target_prob_map, remove_from_dispersed=False, target_station=target_station,
                    map_unit_km=map_unit_km
                )
                relocated_cars += relocated
        
        return relocated_cars > 0

    @staticmethod
    def _find_low_utilization_station(stations: list[ChargingStation]) -> Optional[ChargingStation]:
        """
        Find a charging station with low utilization for relocation target.
        
        Implements a tiered selection strategy to balance station load:
        1. Prefer stations with below 80% of average occupancy rate and not full
        2. Fallback to any non-full station if no low-utilization stations available
        3. Return None if all stations are full
        
        This helps prevent relocating cars to already crowded stations and
        distributes cars more evenly across the charging infrastructure.
        
        Args:
            stations: List of all charging stations in the system
            
        Returns:
            ChargingStation object with low utilization if available, None if all full
        """
        if not stations:
            return None
        
        # Calculate average occupancy rate
        avg_occupancy_rate = sum(s.occupied_spots / s.capacity for s in stations) / len(stations)
        
        # Find stations with below-average utilization and not full
        low_util_stations = [
            s for s in stations 
            if (s.occupied_spots / s.capacity) < avg_occupancy_rate * 0.8 and not s.is_full()
        ]
        
        if not low_util_stations:
            # Fallback: any non-full station
            low_util_stations = [s for s in stations if not s.is_full()]
        
        if not low_util_stations:
            return None
        
        # Return random low-utilization station
        return np.random.choice(low_util_stations)

    @staticmethod
    def _compute_distance_from_demand(location: tuple, prob_map: np.ndarray) -> float:
        """
        Compute weighted average distance from a location to high-demand areas.
        
        Uses Monte Carlo sampling to estimate how far a car location is from typical
        demand centers. Samples random locations weighted by the probability map
        (higher probability = higher demand) and computes average distance.
        
        This metric helps identify cars that are poorly positioned relative to
        anticipated demand, making them good candidates for relocation.
        
        Args:
            location: (x, y) coordinates of car to evaluate
            prob_map: Probability distribution representing demand intensity
            
        Returns:
            Average distance to sampled demand centers in grid units
        """
        # Sample high-demand locations from the probability map
        num_samples = 20
        total_distance = 0.0
        
        for _ in range(num_samples):
            demand_center = MappingUtilities.get_random_location_from_prob_map(prob_map)
            distance = EventHandler.compute_distance(location, demand_center)
            total_distance += distance
        
        # Return average distance to sampled demand centers
        return total_distance / num_samples

    @staticmethod
    def _relocate_single_car(car: Car, stations: list[ChargingStation], car_sharing_system: CarSharingSystem,
                            future_event_set: FutureEventSet, current_time: float, target_prob_map: np.ndarray,
                            remove_from_dispersed: bool = False, target_station: Optional[ChargingStation] = None,
                            map_unit_km: float = 0.1) -> int:
        """
        Relocate a single car to a target charging station.
        
        Performs the complete relocation workflow for one car.
        
        Upon arrival (CAR_PARKING event):
        - Car will park at target station
        - Start charging
        - Eventually become available again (CAR_AVAILABLE event)
        
        Args:
            car: Car to relocate
            stations: List of all charging stations
            car_sharing_system: System for state updates
            future_event_set: FES for scheduling parking event
            current_time: Current simulation time (minutes)
            target_prob_map: Demand probability map (currently unused but available for future enhancements)
            remove_from_dispersed: If True, remove car from dispersed cars registry
            target_station: Destination charging station (must not be None)
            map_unit_km: Grid to km conversion factor
            
        Returns:
            1 if relocation was performed, 0 otherwise (for counting)
        """
        # Mark car as unavailable and remove from current location
        if car.available:
            car.available = False
            car_sharing_system.update_available_car_count()
            
        if car.charging:
            car_sharing_system.remove_charging_car(car)
            
        if car.station_id is not None:
            station = stations[car.station_id]
            station.remove_car(car.car_id)
            car.unpark_from_station()
            car_sharing_system.release_station_spot()
        
        if remove_from_dispersed:
            car_sharing_system.remove_dispersed_car(car)
        
        # Mark as relocating and schedule travel
        car.relocate(target_station.station_id)
        distance_to_station = EventHandler.compute_distance(car.location, target_station.location)
        car.drive(distance_to_station, target_station.location)
        
        # Reserve spot at target station
        target_station.park_car(car.car_id)
        
        # Schedule parking event
        travel_time = EventHandler.compute_travel_time(distance_to_station, car.speed, map_unit_km)
        car_parking_event = Event(EventType.CAR_PARKING, current_time + travel_time, car_id=car.car_id)
        future_event_set.schedule(car_parking_event)
        
        return 1


# ------------------------------------------------------------------------------
# MAPPING UTILITIES
# ------------------------------------------------------------------------------
class MappingUtilities:
    """
    A utility class for creating and manipulating probability-based spatial maps.
    This class provides static methods for generating synthetic population density maps
    using Gaussian mixture models, sampling locations from probability distributions,
    and visualizing probability maps.
    """

    @staticmethod
    def create_population_density(width: int = 100, height: int = 100, n_centers: int = 5) -> np.ndarray:
        """
        Create a synthetic population density map using Gaussian mixtures.

        This method generates a 2D population density map by combining multiple Gaussian 
        distributions (mixture model) centered at random locations. The result is normalized 
        to represent a probability distribution over the spatial domain.

        Args:
            width (int, optional): Width of the density map in grid cells. Defaults to 100.
            height (int, optional): Height of the density map in grid cells. Defaults to 100.
            n_centers (int, optional): Number of Gaussian centers to generate in the mixture. 
                Defaults to 5.

        Returns:
            np.ndarray: A 2D numpy array of shape (height, width) containing normalized 
                population density values that sum to 1.0. Each element represents the 
                probability density at that grid location.
        """
        x, y = np.meshgrid(np.linspace(0, width, width), 
                        np.linspace(0, height, height))
        pos: np.ndarray = np.dstack((x, y))
        density: np.ndarray = np.zeros((height, width))
        for _ in range(n_centers):
            center = [np.random.uniform(-width/2, 1.5*width), 
                    np.random.uniform(-height/2, 1.5*height)]
            cov = [[1000, 0],
                [0, 1000]]
            rv = multivariate_normal(center, cov)
            density += rv.pdf(pos)
        total = np.sum(density)
        density /= total
        return density 
    
    @staticmethod
    def get_random_location_from_prob_map(prob_map: np.ndarray) -> tuple[int, int]:
        """
        Sample a random location from a probability map using weighted random selection.
        This method treats the probability map as a discrete probability distribution and samples
        a single (x, y) coordinate according to the probabilities. Locations with higher probability
        values are more likely to be selected.
            prob_map (np.ndarray): A 2D numpy array representing a probability distribution.
                Values should be non-negative and sum to 1.0 (or will be treated as relative weights).
            tuple[int, int]: A tuple (x, y) representing the sampled grid coordinates, where:
                - x is the column index (horizontal position)
                - y is the row index (vertical position)
        """
        flat_prob = prob_map.flatten()
        flat_index = np.random.choice(len(flat_prob), p=flat_prob)
        y, x = np.unravel_index(flat_index, prob_map.shape)
        return (x, y)
    
    @staticmethod
    def plot_probability_map(prob_map: np.ndarray, title: str):
        """
        Visualize a probability map as a heatmap.
        This method creates a matplotlib heatmap visualization of the given probability map.
            prob_map (np.ndarray): A 2D numpy array containing probability density values to visualize.
            title (str): The title to display on the plot.
        """
        plt.imshow(prob_map, cmap='hot', interpolation='nearest', origin='lower')
        plt.colorbar(label='Probability Density')
        plt.title(title)
        plt.show()


# ------------------------------------------------------------------------------
# SIMULATION ENGINE
# ------------------------------------------------------------------------------
class SimulationEngine:
    """
    Central controller and orchestrator for the car sharing discrete-event simulation.
    
    This class serves as the main simulation engine, managing the entire lifecycle of
    the car sharing system simulation from initialization through execution to results
    reporting. It coordinates the interaction between cars, users, charging stations,
    and implements time-varying demand patterns with optional traffic congestion effects.
    
    The simulation supports both stationary (constant arrival rate) and non-stationary
    (time-varying arrival rate) modes, with configurable peak hours following realistic
    daily patterns. It includes advanced features like transient detection, confidence
    interval computation per arrival rate phase, and periodic car relocation strategies.
    
    Attributes:
        -- Simulation Parameters --
        num_cars (int): Total number of cars in the fleet
        num_stations (int): Total number of charging stations
        max_station_capacity (int): Maximum parking spots per station
        simulation_time (float): Total simulation duration in minutes
        max_arrival_rate (float): Maximum user arrival rate in requests/minute
        max_autonomy (int): Maximum car battery range in grid units
        charging_rate (float): Battery charging speed in grid units/minute
        min_autonomy (int): Minimum battery level required for availability
        max_destination_charging_distance (float): Max distance car travels to charge after trip
        base_speed (float): Base vehicle speed in km/h (before congestion)
        speed (float): Current vehicle speed in km/h (adjusted for congestion)
        min_trip_distance (float): Minimum acceptable trip distance in grid units
        max_pickup_distance (float): Maximum distance user accepts for pickup
        max_waiting_time (float): Maximum time user waits before abandoning request
        relocation_interval (float): Time between periodic relocation events in minutes
        cars_per_relocation (int): Number of cars relocated per relocation event
        
        -- Simulation Mode Flags --
        stationary (bool): If True, use constant arrival rate; if False, time-varying rates
        enable_congestion (bool): If True, adjust speed based on traffic congestion
        congestion_factor (float): Maximum speed reduction factor (0.0 to 1.0)
        map_unit_km (float): Conversion factor from grid units to kilometers
        
        -- Statistical Analysis Tools --
        transient_detector (Optional[TransientDetector]): Detector for transient phase end
        transient_end_time (Optional[float]): Detected transient phase end time in minutes
        confidence_interval_checker (bool): If True, compute confidence intervals
        confidence_level (float): Confidence level for CI computation (default 0.95)
        max_interval_width (float): Target maximum relative CI width
        min_sample_count (int): Minimum number of batches required for CI
        batch_size (float): Duration of each batch for CI computation in minutes
        
        -- Confidence Intervals Per Arrival Rate (Non-Stationary Mode) --
        ci_checkers_per_rate (dict): ConfidenceInterval checker for each arrival rate
        batch_data_per_rate (dict): Batch accumulation data per arrival rate
        final_ci_per_rate (dict): Final confidence intervals per arrival rate
        
        -- System Components --
        fes (FutureEventSet): Priority queue of scheduled future events
        cars (list[Car]): All vehicles in the system
        stations (list[ChargingStation]): All charging stations
        car_sharing_system (CarSharingSystem): Central system state and statistics manager
        total_capacity (int): Total parking capacity across all stations
        
        -- Spatial Distribution Maps --
        home_prob_map (np.ndarray): Probability distribution for home locations
        work_prob_map (np.ndarray): Probability distribution for work locations
        demand_prob_map (np.ndarray): Combined probability distribution (average of home/work)
        
        -- Time-Series Tracking --
        avg_num_total_users_over_windows (list): Average total users per detection window
    """
    
    def __init__(self, seed: int, map_seed: int, num_cars: int, num_stations: int, 
                 max_station_capacity: int, simulation_time: float, max_arrival_rate: float, 
                 max_autonomy: int, charging_rate: float, min_autonomy: int, 
                 max_destination_charging_distance: float, max_pickup_distance: float, 
                 max_waiting_time: float, speed: float = 30.0, min_trip_distance: float = 30.0, 
                 relocation_interval: float = 1440.0, cars_per_relocation: int = 100,
                 transient_detector: Optional[TransientDetector] = None, 
                 confidence_interval_checker: bool = False, map_unit_km: float = 0.1, 
                 stationary: bool = False, enable_congestion: bool = True, 
                 congestion_factor: float = 0.5, confidence_level: float = 0.95, 
                 max_interval_width: float = 0.05, min_sample_count: int = 10,
                 batch_size: float = 30.0):
        """
        Initialize the simulation engine with all system parameters and initial state.
        
        Sets up the complete simulation environment including:
        - Generating spatial probability maps for user origins and destinations
        - Creating and distributing charging stations based on demand
        - Allocating vehicles proportionally to station capacities
        - Initializing the future event set with first user request
        - Configuring statistical analysis tools (transient detection, CI computation)
        
        Args:
            seed: Random seed for simulation events (user arrivals, destinations)
            map_seed: Random seed for spatial map generation (deterministic maps)
            num_cars: Total number of vehicles in the fleet
            num_stations: Total number of charging stations to create
            max_station_capacity: Maximum parking spots allowed per station
            simulation_time: Total simulation duration in minutes
            max_arrival_rate: Peak user arrival rate in requests/minute
            max_autonomy: Maximum battery range in grid units
            charging_rate: Battery recharge speed in grid units/minute
            min_autonomy: Minimum battery level for car availability
            max_destination_charging_distance: Max distance between charging station and destination
            max_pickup_distance: Max distance user accepts to walk to car
            max_waiting_time: Max wait time before user abandons request (minutes)
            speed: Base vehicle speed in km/h (default: 30.0)
            min_trip_distance: Minimum trip distance to enforce (default: 30.0)
            relocation_interval: Time between relocations in minutes (default: 1440.0)
            cars_per_relocation: Cars relocated per event (default: 100)
            transient_detector: Optional transient phase detector
            confidence_interval_checker: Enable CI computation if True (default: False)
            map_unit_km: Grid to km conversion factor (default: 0.1)
            stationary: Use constant arrival rate if True (default: False)
            enable_congestion: Enable traffic congestion effects if True (default: True)
            congestion_factor: Max speed reduction from congestion (default: 0.5)
            confidence_level: CI confidence level (default: 0.95)
            max_interval_width: Target max relative CI width (default: 0.05)
            min_sample_count: Min batches for CI (default: 10)
            batch_size: Batch duration for CI in minutes (default: 30.0)
        """
        # Set seed for map generation
        np.random.seed(map_seed)
        self.num_cars = num_cars
        self.num_stations = num_stations
        self.max_station_capacity = max_station_capacity
        self.simulation_time = simulation_time
        self.max_arrival_rate = max_arrival_rate
        self.max_autonomy = max_autonomy
        self.charging_rate = charging_rate
        self.min_autonomy = min_autonomy
        self.max_destination_charging_distance = max_destination_charging_distance
        self.base_speed = speed 
        self.speed = speed
        self.min_trip_distance = min_trip_distance
        self.max_pickup_distance = max_pickup_distance
        self.max_waiting_time = max_waiting_time
        self.relocation_interval = relocation_interval
        self.cars_per_relocation = cars_per_relocation
        self.stationary = stationary
        self.enable_congestion = enable_congestion
        self.congestion_factor = congestion_factor
        self.map_unit_km = map_unit_km
        self.transient_detector: Optional[TransientDetector] = transient_detector
        self.confidence_interval_checker: bool = confidence_interval_checker
        self.confidence_level: float = confidence_level
        self.max_interval_width: float = max_interval_width
        self.min_sample_count: int = min_sample_count
        self.batch_size: float = batch_size
        self.fes: FutureEventSet = FutureEventSet()
        if transient_detector is None and confidence_interval_checker:
            self.transient_end_time = 24 * 60.0  # Default transient period: 1 day
        else:
            self.transient_end_time = None  # To be determined by transient detector
        self.avg_num_total_users_over_windows: list[tuple[float, float]] = []
        
        # Confidence intervals per arrival rate phase (for non-stationary systems)
        # Dictionary of ConfidenceInterval checkers, one for each arrival rate multiplier
        self.ci_checkers_per_rate: dict[float, ConfidenceInterval] = {}  # rate_multiplier -> ConfidenceInterval
        self.batch_data_per_rate: dict[float, dict] = {}  # rate_multiplier -> {sum, count, next_batch_time}
        self.final_ci_per_rate: dict[float, tuple[float, float]] = {}  # rate_multiplier -> (lower, upper)

        # Create two separate probability maps: home and work distributions
        # Use map_seed to deterministically generate distinct maps
        np.random.seed(map_seed)
        self.home_prob_map = MappingUtilities.create_population_density()
        np.random.seed(map_seed + 1)
        self.work_prob_map = MappingUtilities.create_population_density()
        # Unified fallback map (average)
        self.demand_prob_map = 0.5 * (self.home_prob_map + self.work_prob_map)

        # Initialize stations
        self.stations: list[ChargingStation] = []
        for i in range(self.num_stations):
            location = MappingUtilities.get_random_location_from_prob_map(self.demand_prob_map)
            capacity = int((self.demand_prob_map[location[1], location[0]] - min(self.demand_prob_map.flatten())) / \
                       (max(self.demand_prob_map.flatten()) - min(self.demand_prob_map.flatten())) * \
                       (self.max_station_capacity - 1) + 1)
            station = ChargingStation(station_id=i, location=location, capacity=capacity)
            self.stations.append(station)
        
        self.total_capacity: int = sum(station.capacity for station in self.stations)
        # If total capacity is less than number of cars, incrementally add capacity to high-demand stations
        while self.total_capacity < self.num_cars:
            location = MappingUtilities.get_random_location_from_prob_map(self.demand_prob_map)
            nearest_station = None
            min_distance = float('inf')
            for station in self.stations:
                distance_to_station = EventHandler.compute_distance(location, station.location)
                if distance_to_station < min_distance:
                    min_distance = distance_to_station
                    nearest_station = station
            if nearest_station is not None and nearest_station.capacity < self.max_station_capacity:
                nearest_station.capacity += 1
                self.total_capacity += 1

        print(f"Total parking station capacity: {self.total_capacity}, Number of cars allocated: {self.num_cars}")

        # Allocate cars proportionally to station capacity for balanced initial distribution
        self.cars: list[Car] = []
        cars_allocated = 0
        for station in self.stations:
            cars_for_this_station = int((station.capacity / self.total_capacity) * self.num_cars)
            for _ in range(min(cars_for_this_station, station.capacity)):
                if cars_allocated >= self.num_cars:
                    break
                car = Car(car_id=cars_allocated, location=station.location, max_autonomy=self.max_autonomy,
                         charging_rate=self.charging_rate, min_autonomy=self.min_autonomy,
                         max_destination_charging_distance=self.max_destination_charging_distance,
                         speed=self.speed, station_id=station.station_id)
                station.park_car(car.car_id)
                self.cars.append(car)
                cars_allocated += 1
        
        # Allocate remaining cars if any
        while cars_allocated < self.num_cars:
            for station in self.stations:
                if cars_allocated >= self.num_cars:
                    break
                if not station.is_full():
                    car = Car(car_id=cars_allocated, location=station.location, max_autonomy=self.max_autonomy,
                             charging_rate=self.charging_rate, min_autonomy=self.min_autonomy,
                             max_destination_charging_distance=self.max_destination_charging_distance,
                             speed=self.speed, station_id=station.station_id)
                    station.park_car(car.car_id)
                    self.cars.append(car)
                    cars_allocated += 1

        # Change seed for simulation events
        np.random.seed(seed)
        self.car_sharing_system = CarSharingSystem(cars=self.cars, stations=self.stations)

    def event_loop(self):
        """
        Execute the main discrete-event simulation loop.
        
        This is the core simulation method that processes events chronologically from
        the future event set until either the queue is empty or simulation time expires.
        
        The loop performs the following at each iteration:
        1. Retrieves the next event from the priority queue (ordered by time)
        2. Advances simulation clock to event time
        3. Charges all cars at stations for the elapsed time interval
        4. Updates time-weighted statistics (availability, utilization, etc.)
        5. Adjusts vehicle speed based on current traffic congestion
        6. Samples system state for statistical analysis
        7. Tracks batch data for confidence interval computation (per arrival rate)
        8. Updates transient detection algorithm (if enabled)
        9. Tracks peak arrival rate windows for specialized statistics
        10. Dispatches event to appropriate handler based on event type
        
        Event types handled:
        - USER_REQUEST: New user requests a car
        - USER_ABANDON: User abandons request after timeout
        - CAR_PARKING: Car completes trip and parks at station
        - CAR_AVAILABLE: Car finishes charging and becomes available
        - CAR_RELOCATE: Periodic system relocation event
        
        The simulation supports both stationary mode (constant arrival rate) and
        non-stationary mode (time-varying arrival rates with distinct peak, day, and
        night periods). In non-stationary mode, confidence intervals are computed
        separately for each arrival rate phase using batch averages.
        
        At simulation end, any open peak window is properly closed and statistics
        are finalized.
        """
        first_user_location = MappingUtilities.get_random_location_from_prob_map(self.home_prob_map)
        travel_distance = 0.0
        while travel_distance < self.min_trip_distance:
            first_user_destination = MappingUtilities.get_random_location_from_prob_map(self.home_prob_map)
            travel_distance = EventHandler.compute_distance(first_user_location, first_user_destination)
        first_user = User(user_id=1, location=first_user_location, destination=first_user_destination,
                          max_waiting_time=self.max_waiting_time, max_pickup_distance=self.max_pickup_distance, request_time=0.0)
        self.car_sharing_system.add_active_user(first_user)
        
        current_time: float = 0.0
        first_event = Event(EventType.USER_REQUEST, current_time, user_id=1)
        self.fes.schedule(first_event)
        
        # Schedule first relocation event
        next_relocation_time = self.relocation_interval
        relocation_event = Event(EventType.CAR_RELOCATE, next_relocation_time)
        self.fes.schedule(relocation_event)

        # Main event loop
        while not self.fes.is_empty():
            event: Event = self.fes.get_next_event()
            if event is None or event.time > self.simulation_time:
                break
            # compute arrival rate for this time
            arrival_rate = self.get_arrival_rate(event.time)
            current_time = event.time
            
            # Charge cars during the time interval
            self.car_sharing_system.charge_cars(current_time - self.car_sharing_system.last_event_time)
            
            # Update time-weighted statistics
            self.car_sharing_system.update_time_weighted_statistics(current_time)
            
            # Update speed based on traffic congestion
            if self.enable_congestion:
                self.speed = self.car_sharing_system.compute_congestion_speed(
                    self.base_speed, self.enable_congestion, self.congestion_factor
                )
                self.car_sharing_system.update_speed(self.speed)
            
            # Sample statistics
            self.car_sharing_system.sample_statistics(current_time)
            
            # Track batch data per arrival rate for CI computation (non-stationary mode)
            if not self.stationary and self.confidence_interval_checker and event.time > self.transient_end_time:
                current_arrival_rate = self.get_arrival_rate(event.time)
                # Normalize to multiplier (relative to max_arrival_rate) and round to avoid floating point issues
                rate_multiplier = round(current_arrival_rate / self.max_arrival_rate, 1)
                
                # Initialize CI checker and batch data for this rate if not exists
                if rate_multiplier not in self.ci_checkers_per_rate:
                    self.ci_checkers_per_rate[rate_multiplier] = ConfidenceInterval(
                        self.min_sample_count,
                        self.max_interval_width,
                        self.confidence_level
                    )
                    self.batch_data_per_rate[rate_multiplier] = {
                        'sum': 0,
                        'count': 0,
                        'next_batch_time': event.time + self.batch_size,
                    }
                
                batch_data = self.batch_data_per_rate[rate_multiplier]
                ci_checker = self.ci_checkers_per_rate[rate_multiplier]
                
                # Accumulate total users for current batch
                num_total_users = self.car_sharing_system.sampled_num_total_users[-1][1]
                batch_data['sum'] += num_total_users
                batch_data['count'] += 1
                
                # Check if batch is complete
                if event.time >= batch_data['next_batch_time']:
                    if batch_data['count'] > 0:
                        batch_avg_total_users = batch_data['sum'] / batch_data['count']
                        ci_checker.add_data_point(batch_avg_total_users)

                        
                        # Check if CI is achieved for this rate
                        if ci_checker.has_enough_data() and rate_multiplier not in self.final_ci_per_rate:
                            final, ci = ci_checker.compute_interval()
                            if final:
                                self.final_ci_per_rate[rate_multiplier] = ci
                                print(f"CI achieved for rate {rate_multiplier}x at time {event.time:.0f}: {ci}")
                    
                    # Reset batch for next interval
                    batch_data['sum'] = 0
                    batch_data['count'] = 0
                    batch_data['next_batch_time'] = event.time + self.batch_size

            # Update transient detector (now using number of total users)
            if self.transient_detector and self.transient_end_time is None:
                self.transient_detector.add_value(event.time, self.car_sharing_system.sampled_num_total_users[-1][1])
                next_interval_start = self.transient_detector.compute_next_interval_start()
                if event.time >= next_interval_start:
                    new_avg = self.transient_detector.compute_average()
                    self.transient_detector.add_average(event.time, new_avg)
                    self.avg_num_total_users_over_windows.append((event.time, new_avg))
                    if self.transient_detector.is_transient_over():
                        self.transient_end_time = self.transient_detector.get_transient_end_time()
                        print(f"Transient detected at time {self.transient_end_time}")
            
            # Track peak arrival rate windows (windows where arrival rate equals max_arrival_rate)
            current_arrival_rate = self.get_arrival_rate(event.time)
            is_at_max_rate = abs(current_arrival_rate - self.max_arrival_rate) < 0.01  # Allow small tolerance
            
            if is_at_max_rate and not self.car_sharing_system.in_peak_window:
                # Entering a peak window
                self.car_sharing_system.start_peak_window(event.time)
            elif not is_at_max_rate and self.car_sharing_system.in_peak_window:
                # Exiting a peak window
                self.car_sharing_system.end_peak_window(event.time)

            # Handle event based on its type
            if event.event_type == EventType.USER_REQUEST:
                origin_map, dest_map = self.get_origin_destination_maps(current_time)
                EventHandler.handle_user_request(event, self.cars, self.car_sharing_system,
                                                self.stations, self.fes, current_time,
                                                arrival_rate, self.min_trip_distance, origin_map, dest_map,
                                                map_unit_km=self.map_unit_km)
            elif event.event_type == EventType.USER_ABANDON:
                EventHandler.handle_user_abandon(event, self.car_sharing_system)
            elif event.event_type == EventType.CAR_PARKING:
                EventHandler.handle_car_parking(event, self.cars, self.car_sharing_system, self.stations, self.fes, current_time)
            elif event.event_type == EventType.CAR_AVAILABLE:
                EventHandler.handle_car_available(event, self.cars, self.stations, self.car_sharing_system, self.fes, current_time,
                                                   map_unit_km=self.map_unit_km)
            elif event.event_type == EventType.CAR_RELOCATE:
                relocated = EventHandler.handle_car_relocate(event, self.cars, self.stations, self.car_sharing_system,
                                                            self.fes, self.cars_per_relocation, self.demand_prob_map,
                                                            map_unit_km=self.map_unit_km)
                if relocated:
                    self.car_sharing_system.relocation_performed(current_time)
                # Schedule next periodic relocation
                next_relocation_time = current_time + self.relocation_interval
                if next_relocation_time < self.simulation_time:
                    relocation_event = Event(EventType.CAR_RELOCATE, next_relocation_time)
                    self.fes.schedule(relocation_event)
        
        # End of simulation: close any open peak window
        if self.car_sharing_system.in_peak_window:
            self.car_sharing_system.end_peak_window(self.simulation_time)

    def get_arrival_rate(self, time_min: float) -> float:
        """
        Determine user arrival rate based on time-of-day schedule.
        
        Implements a realistic daily arrival rate pattern with distinct periods:
        - Morning Peak (06:00-10:00): 1.0x max_arrival_rate (commute to work)
        - Day Hours (10:00-16:00): 0.6x max_arrival_rate (midday activity)
        - Evening Peak (16:00-20:00): 1.0x max_arrival_rate (commute home)
        - Night Hours (20:00-06:00): 0.2x max_arrival_rate (minimal activity)
        
        If simulation is in stationary mode, returns constant max_arrival_rate
        regardless of time.
        
        Args:
            time_min: Current simulation time in minutes
            
        Returns:
            Arrival rate in requests per minute for the current time period
        """
        # Define window multipliers relative to max_arrival_rate
        morning_peak = 1.0
        evening_peak = 1.0
        day_multiplier = 0.6
        night_multiplier = 0.2

        minutes_per_day = 24 * 60
        t = time_min % minutes_per_day

        if self.stationary:
            # Always max rate
            return self.max_arrival_rate

        # Time-varying schedule
        if 6 * 60 <= t < 10 * 60:
            return self.max_arrival_rate * morning_peak
        if 10 * 60 <= t < 16 * 60:
            return self.max_arrival_rate * day_multiplier
        if 16 * 60 <= t < 20 * 60:
            return self.max_arrival_rate * evening_peak
        return self.max_arrival_rate * night_multiplier

    def get_origin_destination_maps(self, time_min: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Select appropriate origin-destination probability maps based on time window.
        
        Returns different spatial distribution maps for different times of day:
        - Morning Peak (06:00-10:00): Morning commute patterns (home → work)
        - Day Hours (10:00-16:00): Midday activity patterns (home → home, local)
        - Evening Peak (16:00-20:00): Evening commute patterns (work → home)
        - Night Hours (20:00-06:00): Nighttime patterns (home → home)
        
        If stationary mode is enabled, returns unified demand_prob_map for both
        origin and destination regardless of time.
        
        Args:
            time_min: Current simulation time in minutes
            
        Returns:
            Tuple of (origin_map, destination_map) as 2D numpy arrays with 
            probability distributions for trip origins and destinations
        """
        if self.stationary:
            return self.demand_prob_map, self.demand_prob_map

        minutes_per_day = 24 * 60
        t = time_min % minutes_per_day
        if 6 * 60 <= t < 10 * 60:
            return self.home_prob_map, self.work_prob_map
        if 10 * 60 <= t < 16 * 60:
            return self.home_prob_map, self.home_prob_map
        if 16 * 60 <= t < 20 * 60:
            return self.work_prob_map, self.home_prob_map
        return self.home_prob_map, self.home_prob_map
                
    def get_statistics(self) -> dict[str, float]:
        """
        Compute and return comprehensive simulation performance statistics.
        
        Calculates key performance indicators (KPIs) from the car sharing system's
        accumulated data over the simulation run. All rates are computed using
        time-weighted integration (area under curves) for accuracy.
        
        Statistics computed:
        - total_user_requests: Total number of user trip requests
        - total_abandoned_requests: Requests abandoned due to timeout
        - percentage_abandoned_requests: Abandonment rate as percentage
        - average_waiting_time: Mean time from request to pickup (minutes)
        - average_trip_distance: Mean trip distance (grid units)
        - vehicle_availability_rate: Fraction of time vehicles are available
        - vehicle_utilization_rate: Fraction of time vehicles are in use
        - charging_station_utilization_rate: Fraction of station capacity used
        - average_queue_length: Mean number of users waiting for cars
        
        Returns:
            Dictionary mapping statistic names to their computed values
        """
        stats = {
            "total_user_requests": self.car_sharing_system.total_user_requests,
            "total_abandoned_requests": self.car_sharing_system.total_abandoned_requests,
            "percentage_abandoned_requests": (self.car_sharing_system.total_abandoned_requests / \
            max(1, self.car_sharing_system.total_user_requests)) * 100,
            "average_waiting_time": self.car_sharing_system.total_waiting_time / \
            max(1, self.car_sharing_system.total_user_requests - self.car_sharing_system.total_abandoned_requests),
            "average_trip_distance": self.car_sharing_system.total_trip_distance / \
            max(1, self.car_sharing_system.total_user_requests - self.car_sharing_system.total_abandoned_requests),
            "vehicle_availability_rate": self.car_sharing_system.area_under_available_cars / (self.num_cars * self.simulation_time),
            "vehicle_utilization_rate": self.car_sharing_system.area_under_used_cars / (self.num_cars * self.simulation_time),
            "charging_station_utilization_rate": self.car_sharing_system.area_under_used_station / \
            (self.total_capacity * self.simulation_time),
            "average_queue_length": self.car_sharing_system.area_under_queue_length / self.simulation_time,
        }
        return stats
    
    def plot_transient_phase(self):
        """
        Visualize transient phase detection results.
        
        Creates a step plot showing the number of total users (queued + in service)
        over simulation time. If transient detection was performed, overlays:
        - Vertical red line indicating detected transient end time
        - Green horizontal lines showing average values per detection window
        """
        samples = self.car_sharing_system.get_num_total_users_over_time()
        times = samples[0]
        queue_lengths = samples[1]

        plt.figure(figsize=(8, 5))
        plt.step(times, queue_lengths, where='post')

        if self.transient_end_time is not None:
            plt.axvline(x=self.transient_end_time, color='r', linestyle='--', label='Transient End Time')

        if self.avg_num_total_users_over_windows:
            windows = sorted(self.avg_num_total_users_over_windows, key=lambda x: x[0])
            window_size = 100.0
            if self.transient_detector is not None:
                window_size = getattr(self.transient_detector, "window_size", None) or window_size
            for i, (t, avg) in enumerate(windows):
                x = [t, t + window_size]
                y = [avg, avg]
                label = 'Avg Num Total Users Over Windows' if i == 0 else None
                plt.plot(x, y, color='green', linewidth=2, label=label)

        plt.title('Number of Total Users Over Time')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Number of Total Users')
        plt.legend()
        plt.grid()
        plt.show()

    def print_statistics(self):
        """
        Display comprehensive simulation results and visualizations.
        
        Outputs three main sections:
        
        1. Overall Statistics: Core performance metrics across entire simulation
           - User request counts (total, abandoned, served)
           - Average waiting time and trip distance  
           - Vehicle availability and utilization rates
           - Charging station utilization
           - Average queue length
        
        2. Peak Window Statistics: Performance during high-demand periods
           - Only displayed if peak windows were tracked
           - Shows statistics aggregated over all peak arrival rate windows
           - Includes duration, rates, and abandonment during peaks
        
        3. Confidence Intervals Per Arrival Rate: Statistical analysis by rate phase
           - Only for non-stationary simulations with CI enabled
           - Separate 95% CI for mean total users at each arrival rate (1.0x, 0.6x, 0.2x)
           - Shows batches collected, mean, bounds, width, achieved status
        
        Additionally calls plot_transient_phase() if transient detection was enabled
        to visualize system stabilization.
        """
        stats = self.get_statistics()
        
        # Print overall statistics in formatted table
        print("\n" + "="*70)
        print("OVERALL SIMULATION STATISTICS")
        print("="*70)
        print(f"Total user requests: {stats['total_user_requests']}")
        print(f"Total abandoned requests: {stats['total_abandoned_requests']}")
        print(f"Percentage abandoned requests: {stats['percentage_abandoned_requests']:.2f}%")
        print(f"Average waiting time: {stats['average_waiting_time']:.2f} minutes")
        print(f"Average trip distance: {stats['average_trip_distance']:.2f} grid units")
        print(f"Vehicle availability rate: {stats['vehicle_availability_rate']:.2f}")
        print(f"Vehicle utilization rate: {stats['vehicle_utilization_rate']:.2f}")
        print(f"Charging station utilization rate: {stats['charging_station_utilization_rate']:.2f}")
        print(f"Average queue length: {stats['average_queue_length']:.2f} users")
        print("="*70)
        
        # Print peak window statistics
        peak_stats = self.car_sharing_system.get_peak_window_statistics()
        if peak_stats['num_peak_windows'] > 0:
            print("\n" + "="*70)
            print("PEAK ARRIVAL RATE WINDOW STATISTICS")
            print("="*70)
            print(f"Number of peak windows: {peak_stats['num_peak_windows']}")
            print(f"Total peak duration: {peak_stats['total_peak_duration']:.2f} minutes ({peak_stats['total_peak_duration']/60:.2f} hours)")
            print(f"Average availability rate: {peak_stats['avg_availability_rate']:.2f}%")
            print(f"Average utilization rate: {peak_stats['avg_utilization_rate']:.2f}%")
            print(f"Average station utilization: {peak_stats['avg_station_utilization']:.2f}%")
            print(f"Average waiting time: {peak_stats['avg_waiting_time']:.2f} minutes")
            print(f"Average queue length: {peak_stats['avg_queue_length']:.2f} users")
            print(f"Average abandonment rate: {peak_stats['avg_abandonment_rate']:.2f}%")
            print(f"Total requests during peaks: {peak_stats['total_peak_requests']}")
            print(f"Total served during peaks: {peak_stats['total_peak_served']}")
            print(f"Total abandoned during peaks: {peak_stats['total_peak_abandoned']}")
            print("="*70)
        
        # Print confidence intervals per arrival rate (for non-stationary systems)
        if not self.stationary and self.ci_checkers_per_rate:
            print("\n" + "="*80)
            print("CONFIDENCE INTERVALS PER ARRIVAL RATE PHASE (95% CI)")
            print("="*80)
            print(f"{'Rate Mult.':<12} {'Batches':<10} {'Mean Users':<12} {'CI Lower':<12} {'CI Upper':<12} {'CI Width':<12} {'Achieved':<10}")
            print("-" * 80)
            
            for rate_mult in sorted(self.ci_checkers_per_rate.keys()):
                ci_checker = self.ci_checkers_per_rate[rate_mult]
                num_batches = ci_checker.get_sample_size()
                
                if num_batches > 0:
                    mean = ci_checker.average
                    
                    # Check if CI has been computed
                    if rate_mult in self.final_ci_per_rate:
                        lower, upper = self.final_ci_per_rate[rate_mult]
                        achieved = "Yes"
                    elif ci_checker.has_enough_data():
                        result = ci_checker.compute_interval()
                        if result:
                            final, (lower, upper) = result
                            achieved = "Yes" if final else "No"
                        else:
                            lower = upper = mean
                            achieved = "N/A"
                    else:
                        lower = upper = mean
                        achieved = "N/A"
                    
                    ci_width = upper - lower
                    print(f"{rate_mult:<12.1f} {num_batches:<10} {mean:<12.2f} {lower:<12.2f} {upper:<12.2f} {ci_width:<12.2f} {achieved:<10}")
                else:
                    print(f"{rate_mult:<12.1f} {num_batches:<10} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<10}")
            
            print("="*80)
            print("Note: Rate multipliers correspond to arrival rate phases:")
            print("  1.0x = Peak hours (morning 6-10am, evening 4-8pm)")
            print("  0.6x = Day hours (10am-4pm)")
            print("  0.2x = Night hours (8pm-6am)")
            print(f"Batch size: {self.batch_size} minutes")
            print("="*80)

        if self.transient_end_time is not None:
            self.plot_transient_phase()


# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # SIMULATION PARAMETERS
    SEED = [0]
    MAP_SEED = [10]
    NUM_CARS = [400]
    NUM_STATIONS = [150]
    MAX_STATION_CAPACITY = [4]
    SIMULATION_TIME = [7 * 24 * 60]
    MAX_ARRIVAL_RATE = [4.0] 
    MAX_AUTONOMY = [1500]
    CHARGING_RATE = [12]
    MIN_AUTONOMY = [200]
    MAX_DESTINATION_CHARGING_DISTANCE = [10]
    MAX_PICKUP_DISTANCE = [10]
    MAX_WAITING_TIME = [15]
    MAX_SPEED = [30.0]
    MIN_TRIP_DISTANCE = [30.0]
    RELOCATION_INTERVAL = [1440.0]
    CARS_PER_RELOCATION = [100]

    # Transient detection parameters
    WINDOW_SIZE = 30
    STRIDE = 2
    NUM_INTERVALS = 40
    THRESHOLD = 0.3
    transient_detector = TransientDetector(WINDOW_SIZE, STRIDE, NUM_INTERVALS, THRESHOLD)
    transient_detector = None  # Disable transient detection

    # Confidence Interval parameters
    CONFIDENCE_LEVEL = 0.95
    MAX_INTERVAL_WIDTH = 0.05
    MIN_SAMPLES_COUNT = 10

    # Boolean Parameters
    STATIONARY = False 
    ENABLE_CONGESTION = True 
    CONFIDENCE_INTERVAL_CHECKER = False

    all_configs = list(product(
        SEED, MAP_SEED, NUM_CARS, NUM_STATIONS, MAX_STATION_CAPACITY,
        SIMULATION_TIME, MAX_ARRIVAL_RATE, MAX_AUTONOMY, CHARGING_RATE,
        MIN_AUTONOMY, MAX_DESTINATION_CHARGING_DISTANCE, MAX_PICKUP_DISTANCE,
        MAX_WAITING_TIME, MAX_SPEED, MIN_TRIP_DISTANCE,
        RELOCATION_INTERVAL, CARS_PER_RELOCATION
    ))
    
    for idx, config in enumerate(all_configs, start=1):
        (
            seed_val, 
            map_seed_val, 
            num_cars_val, 
            num_stations_val, 
            max_station_capacity_val,
            simulation_time_val, 
            max_arrival_rate_val, 
            max_autonomy_val, 
            charging_rate_val,
            min_autonomy_val, 
            max_destination_charging_distance_val, 
            max_pickup_distance_val,
            max_waiting_time_val, 
            average_speed_val, 
            min_trip_distance_val,
            relocation_interval_val, 
            cars_per_relocation_val
        ) = config

        # Print complete configuration before run
        print("\n" + "=" * 80)
        print(f"SIMULATION RUN {idx}/{len(all_configs)} - CONFIGURATION")
        print("=" * 80)
        print("\n--- Random Seeds ---")
        print(f"  Simulation seed:                    {seed_val}")
        print(f"  Map generation seed:                {map_seed_val}")
        
        print("\n--- Fleet & Infrastructure ---")
        print(f"  Number of cars:                     {num_cars_val}")
        print(f"  Number of charging stations:        {num_stations_val}")
        print(f"  Max station capacity:               {max_station_capacity_val} spots")
        
        print("\n--- Simulation Parameters ---")
        print(f"  Simulation time:                    {simulation_time_val:.0f} minutes ({simulation_time_val/60:.1f} hours, {simulation_time_val/1440:.1f} days)")
        print(f"  Stationary mode:                    {STATIONARY}")
        print(f"  Max arrival rate (peak):            {max_arrival_rate_val} requests/minute")
        print(f"  Congestion enabled:                 {ENABLE_CONGESTION}")

        print("\n--- Vehicle Parameters ---")
        print(f"  Max autonomy:                       {max_autonomy_val} grid units")
        print(f"  Min autonomy (availability):        {min_autonomy_val} grid units")
        print(f"  Charging rate:                      {charging_rate_val} grid units/minute")
        print(f"  Base speed:                         {average_speed_val} km/h")
        print(f"  Max destination-charging distance:  {max_destination_charging_distance_val} grid units")
        
        print("\n--- User Parameters ---")
        print(f"  Min trip distance:                  {min_trip_distance_val} grid units")
        print(f"  Max pickup distance:                {max_pickup_distance_val} grid units")
        print(f"  Max waiting time:                   {max_waiting_time_val} minutes")
        
        print("\n--- Relocation Strategy ---")
        print(f"  Relocation interval:                {relocation_interval_val} minutes ({relocation_interval_val/60:.1f} hours)")
        print(f"  Cars per relocation event:          {cars_per_relocation_val}")
        
        print("\n--- Statistical Analysis ---")
        print(f"  Transient detection:                {"Enabled" if transient_detector else "Disabled"}")
        print(f"  Confidence intervals:               {"Enabled" if CONFIDENCE_INTERVAL_CHECKER else "Disabled"}")
        print(f"  Confidence level:                   {CONFIDENCE_LEVEL}")
        print(f"  Max CI width:                       {MAX_INTERVAL_WIDTH}")
        print(f"  Min sample count:                   {MIN_SAMPLES_COUNT}")
        
        print("=" * 80)
        print("Starting simulation...\n")

        plt.close('all')

        start_time = time.time()
        simulation_engine = SimulationEngine(
            seed=int(seed_val),
            map_seed=int(map_seed_val),
            num_cars=int(num_cars_val),
            num_stations=int(num_stations_val),
            max_station_capacity=int(max_station_capacity_val),
            simulation_time=float(simulation_time_val),
            max_arrival_rate=float(max_arrival_rate_val),
            max_autonomy=int(max_autonomy_val),
            charging_rate=float(charging_rate_val),
            min_autonomy=int(min_autonomy_val),
            max_destination_charging_distance=float(max_destination_charging_distance_val),
            max_pickup_distance=float(max_pickup_distance_val),
            max_waiting_time=float(max_waiting_time_val),
            speed=float(average_speed_val),
            min_trip_distance=float(min_trip_distance_val),
            relocation_interval=float(relocation_interval_val),
            cars_per_relocation=int(cars_per_relocation_val),
            transient_detector=transient_detector,
            confidence_interval_checker=CONFIDENCE_INTERVAL_CHECKER,
            stationary=STATIONARY,
            enable_congestion=ENABLE_CONGESTION,
            confidence_level=CONFIDENCE_LEVEL,
            max_interval_width=MAX_INTERVAL_WIDTH,
            min_sample_count=MIN_SAMPLES_COUNT
        )

        simulation_engine.event_loop()
        elapsed = time.time() - start_time
        print(f"\n{'=' * 80}")
        print(f"Run {idx} completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print(f"{'=' * 80}\n")
        simulation_engine.print_statistics()
