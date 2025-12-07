from enum import Enum
from queue import PriorityQueue
from typing import Optional
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import time
from itertools import product
from confidence_interval import ConfidenceInterval
from rv_generation import RVGenerator

# ==================================================
# INTERARRIVAL AND SERVICE DISTRIBUTIONS
# ==================================================

class DistributionType(Enum):
    EXPONENTIAL = "EXPONENTIAL"
    HYPEREXPONENTIAL = "HYPEREXPONENTIAL"
    ERLANG_K = "ERLANG-K"
    PARETO = "PARETO"

# ------------------------------------------------------------------------------
# EVENT DEFINITION
# ------------------------------------------------------------------------------
class EventType(Enum):
    USER_REQUEST = "USER_REQUEST"
    USER_ABANDON = "USER_ABANDON"
    CAR_PARKING = "CAR_PARKING"
    CAR_AVAILABLE = "CAR_AVAILABLE"
    CAR_RELOCATE = "CAR_RELOCATE"

class Event:
    def __init__(self, event_type: EventType, time: float, user_id: Optional[int] = None, 
                 car_id: Optional[int] = None, station_id: Optional[int] = None):
        self.event_type = event_type
        self.time = time
        self.user_id = user_id
        self.car_id = car_id
        self.station_id = station_id
        self.cancelled: bool = False

    def cancel(self):
        self.cancelled = True

    def is_cancelled(self) -> bool:
        return self.cancelled
    
    def __lt__(self, other) -> bool:
        return self.time < other.time
   

# ------------------------------------------------------------------------------
# CAR
# ------------------------------------------------------------------------------
class Car:
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
        self.location = destination
        self.autonomy -= distance
        return True

    def charge(self, time: float):
        """Charge car for given time interval."""
        if self.charging:
            self.autonomy = min(self.autonomy + self.charging_rate * time, self.max_autonomy)
            if self.autonomy >= self.min_autonomy:
                self.available = True
    
    def park_in_station(self, station_id: int):
        """Park car at station."""
        self.station_id = station_id

    def unpark_from_station(self):
        """Unpark car from station."""
        self.station_id = None

    def assign_user(self, user_id: int):
        """Assign car to user."""
        if self.available:
            self.charging = False
            self.assigned_user_id = user_id
            self.available = False
        else:
            print("Car is not available for assignment.")
    
    def release_user(self):
        """Release user after trip completion."""
        if self.assigned_user_id is not None:
            self.assigned_user_id = None
            if self.autonomy < self.min_autonomy:
                self.available = False
            else:
                self.available = True
        else:
            print("No user is assigned to this car.")

    def is_available(self) -> bool:
        """Check if car is available."""
        return self.available

    def relocate(self, station_id: int):
        """Mark car as relocating to station."""
        self.relocating = True
        self.relocating_station_id = station_id

    def is_relocating(self) -> bool:
        """Check if car is relocating."""
        return self.relocating

    def get_relocating_station_id(self) -> int:
        """Get relocation destination station ID."""
        return self.relocating_station_id
    
    def stop_relocating(self):
        """Stop relocation mode."""
        self.relocating = False
        self.relocating_station_id = None


# ------------------------------------------------------------------------------
# USER
# ------------------------------------------------------------------------------
class User:
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
        """Assign car to user."""
        self.assigned_car_id = car_id
        self.waiting = False
        self.service_time = service_time

    def resign_car(self):
        """Release car after trip."""
        self.assigned_car_id = None
        self.served = True

    def request_car(self):
        """Mark user as requesting car."""
        self.waiting = True

    def abandon_request(self):
        """Abandon request after timeout."""
        self.waiting = False
        self.assigned_car_id = None

    def put_in_queue(self):
        """Place user in waiting queue."""
        self.waiting = True

    def is_waiting(self) -> bool:
        """Check if user is waiting."""
        return self.waiting

    def waiting_time(self) -> float:
        """Calculate waiting time between request and service."""
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
    def __init__(self, station_id: int, location: tuple, capacity: int):
        self.station_id = station_id
        self.location = location 
        self.capacity = capacity
        self.occupied_spots: int = 0
        self.parked_cars: list[int] = []
        self.full: bool = False

    def park_car(self, car_id: int) -> bool:
        """Park car if space available."""
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
        """Remove car from station."""
        if car_id in self.parked_cars:
            self.parked_cars.remove(car_id)
            self.occupied_spots -= 1
            self.full = False
            return True
        else:
            print("Car not found at the charging station.")
            return False

    def is_full(self) -> bool:
        """Check if station is full."""
        return self.full
        

# ------------------------------------------------------------------------------
# FUTURE EVENT SET
# ------------------------------------------------------------------------------
class FutureEventSet:
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
    

# -------------------------------------------------------------------------------
# CAR SHARING SYSTEM
# -------------------------------------------------------------------------------
class CarSharingSystem:
    def __init__(self, cars: list[Car], stations: list[ChargingStation]):
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
        
        # Interarrival time tracking
        self.interarrival_times: list[float] = []  # Track all interarrival times
        self.arrival_times: list[float] = []  # Track all arrival times for histogram

        # Initialize time-weighted statistics
        self.area_under_available_cars: float = 0.0
        self.area_under_used_cars: float = 0.0
        self.area_under_used_station: float = 0.0
        self.area_under_queue_length: float = 0.0  # Area under waiting queue length curve
        self.last_event_time: float = 0.0
        self.last_available_car_count: int = sum(1 for c in cars if c.is_available())
        self.last_used_car_count: int = 0
        self.last_used_station_spots: int = len(cars)
        self.last_queue_length: int = 0  # Current waiting queue length

    def update_time_weighted_statistics(self, new_time: float):
        """Update time-weighted statistics using integration."""
        start = self.last_event_time
        time_delta = new_time - start
        if time_delta > 0:
            self.area_under_available_cars += self.last_available_car_count * time_delta
            self.area_under_used_cars += self.last_used_car_count * time_delta
            self.area_under_used_station += self.last_used_station_spots * time_delta
            self.area_under_queue_length += self.last_queue_length * time_delta

        self.last_event_time = new_time

    def sample_statistics(self, current_time: float):
        """Sample current system state for time-series analysis."""
        self.num_samples += 1
        # Count serving users: cars with assigned_user_id that are not relocating
        num_serving_users = sum(1 for car in self.cars 
                               if car.assigned_user_id is not None 
                               and not car.is_relocating())
        self.sampled_num_serving_users.append((current_time, num_serving_users))
        self.sampled_num_total_users.append((current_time, len(self.active_users)))

    def update_speed(self, new_speed: float):
        """Update speed of all cars."""
        for car in self.cars:
            car.speed = new_speed
    
    def compute_congestion_speed(self, base_speed: float, congestion_enabled: bool = True, 
                                congestion_factor: float = 0.5) -> float:
        """Compute speed adjusted for traffic congestion."""
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
        """Accumulate trip distance."""
        self.total_trip_distance += distance

    def add_waiting_time(self, user: User):
        """Accumulate waiting time."""
        self.total_waiting_time += user.waiting_time()

    def add_charging_car(self, car: Car):
        """Register car as charging."""
        self.charging_cars[car.car_id] = car

    def remove_charging_car(self, car: Car):
        """Remove car from charging registry."""
        if car.car_id in self.charging_cars:
            del self.charging_cars[car.car_id]

    def charge_cars(self, interval: float):
        """Charge all cars at stations."""
        for car in self.charging_cars.values():
            car.charge(interval)

    def add_dispersed_car(self, car: Car):
        """Register car as dispersed."""
        self.dispersed_cars[car.car_id] = car

    def remove_dispersed_car(self, car: Car):
        """Remove car from dispersed registry."""
        if car.car_id in self.dispersed_cars:
            del self.dispersed_cars[car.car_id]

    def get_dispersed_cars(self) -> list[Car]:
        """Get list of dispersed cars."""
        return list(self.dispersed_cars.values())

    def add_active_user(self, user: User):
        """Register user as active."""
        self.active_users[user.user_id] = user

    def remove_active_user(self, user: User):
        """Remove user from active users."""
        if user.user_id in self.active_users:
            del self.active_users[user.user_id]

    def get_active_user(self, user_id: int) -> Optional[User]:
        """Get active user by ID."""
        return self.active_users.get(user_id, None)

    def add_client_to_waiting(self, user: User):
        """Add user to waiting queue."""
        self.waiting_users[user.user_id] = user
        self.last_queue_length = len(self.waiting_users)

    def remove_client_from_waiting(self, user: User):
        """Remove user from waiting queue."""
        if user.user_id in self.waiting_users:
            del self.waiting_users[user.user_id]
            self.last_queue_length = len(self.waiting_users)
    
    def get_waiting_clients(self) -> list[User]:
        """Get list of waiting users."""
        return list(self.waiting_users.values())
    
    def update_available_car_count(self):
        """Update count of available cars."""
        self.last_available_car_count = sum(1 for c in self.cars if c.is_available())

    def update_used_car_count(self):
        """Update count of cars assigned to users."""
        self.last_used_car_count = sum(1 for c in self.cars if c.assigned_user_id is not None)

    def user_requested_car(self):
        """Increment user requests counter."""
        self.total_user_requests += 1

    def user_abandoned(self, user: User):
        """Record user abandonment."""
        self.total_abandoned_requests += 1
        self.abandoned_users[user.user_id] = user

    def occupy_station_spot(self):
        """Increment station spot usage counter."""
        self.last_used_station_spots += 1

    def release_station_spot(self):
        """Decrement station spot usage counter."""
        self.last_used_station_spots -= 1

    def relocation_performed(self, time: float):
        """Record relocation event."""
        self.last_relocation_time = time
    
    def get_num_serving_users_over_time(self) -> tuple[list[float], list[int]]:
        """Get time series of serving users."""
        times = [t for t, _ in self.sampled_num_serving_users]
        num_serving = [n for _, n in self.sampled_num_serving_users]
        return times, num_serving
    
    def get_num_total_users_over_time(self) -> tuple[list[float], list[int]]:
        """Get time series of total active users."""
        times = [t for t, _ in self.sampled_num_total_users]
        num_total = [n for _, n in self.sampled_num_total_users]
        return times, num_total
    
    def add_interarrival_time(self, interarrival_time: float):
        """Record interarrival time."""
        self.interarrival_times.append(interarrival_time)
    
    def add_arrival_time(self, arrival_time: float):
        """Record arrival time for histogram."""
        self.arrival_times.append(arrival_time)


# ------------------------------------------------------------------------------
# EVENT HANDLERS
# ------------------------------------------------------------------------------
class EventHandler:
    @staticmethod
    def compute_travel_time(distance: float, speed: float, map_unit_km: float = 0.1) -> float:
        distance_km = distance * map_unit_km
        return (distance_km / speed) * 60.0
    
    @staticmethod
    def compute_distance(loc1: tuple, loc2: tuple) -> float:
        return ((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2) ** 0.5
    
    @staticmethod
    def find_nearest_available_car(location: tuple, cars: list[Car], max_distance: float) -> Optional[tuple[Car, float]]:
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
                           min_trip_distance: float,
                           demand_origin_map: np.ndarray, demand_dest_map: np.ndarray,
                           interarrival_type: DistributionType = DistributionType.EXPONENTIAL,
                           interarrival_params: dict = {"lambda": 1.0}):
        inter_arrival_time = EventHandler._generate_new_time(interarrival_type, interarrival_params)
        next_request_time = current_time + inter_arrival_time
        next_user_id = user.user_id + 1
        
        # Track interarrival time and arrival time
        car_sharing_system.add_interarrival_time(inter_arrival_time)
        car_sharing_system.add_arrival_time(next_request_time)
        
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
                            min_trip_distance: float, demand_origin_map: np.ndarray, 
                            demand_dest_map: np.ndarray, map_unit_km: float = 0.1,
                            interarrival_type: DistributionType = DistributionType.EXPONENTIAL,
                            interarrival_params: dict = {"lambda": 1.0}) -> bool:
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
                                        min_trip_distance, demand_origin_map, demand_dest_map,
                                        interarrival_type, interarrival_params)
        
        return assigned
    
    @staticmethod
    def handle_user_abandon(event: Event, car_sharing_system: CarSharingSystem) -> bool:
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

# ------------------------------------------------------------------------------
# MAPPING UTILITIES
# ------------------------------------------------------------------------------
class MappingUtilities:

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
    def __init__(self, seed: int, map_seed: int, num_cars: int, num_stations: int, 
                 max_station_capacity: int, simulation_time: float,
                 max_autonomy: int, charging_rate: float, min_autonomy: int, 
                 max_destination_charging_distance: float, max_pickup_distance: float, 
                 max_waiting_time: float, speed: float = 30.0, min_trip_distance: float = 30.0, 
                 relocation_interval: float = 1440.0, cars_per_relocation: int = 100,
                 confidence_interval_checker: bool = False, map_unit_km: float = 0.1, 
                 enable_congestion: bool = True, congestion_factor: float = 0.5, 
                 confidence_level: float = 0.95, max_interval_width: float = 0.05, 
                 min_sample_count: int = 10, batch_size: float = 30.0,
                 interarrival_type: DistributionType = DistributionType.EXPONENTIAL,
                 interarrival_params: dict = {"lambda": 1.0},):
        # Set seed for map generation
        np.random.seed(map_seed)
        self.num_cars = num_cars
        self.num_stations = num_stations
        self.max_station_capacity = max_station_capacity
        self.simulation_time = simulation_time
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
        self.enable_congestion = enable_congestion
        self.congestion_factor = congestion_factor
        self.map_unit_km = map_unit_km
        self.confidence_interval_checker: bool = confidence_interval_checker
        self.confidence_level: float = confidence_level
        self.max_interval_width: float = max_interval_width
        self.min_sample_count: int = min_sample_count
        self.batch_size: float = batch_size
        self.interarrival_type: DistributionType = interarrival_type
        self.interarrival_params: dict = interarrival_params
        self.fes: FutureEventSet = FutureEventSet()
        if confidence_interval_checker:
            self.transient_end_time = 24 * 60.0  # Default transient period: 1 day
        else:
            self.transient_end_time = None  # To be determined by transient detector
        
        # Confidence interval tracking for overall simulation
        self.ci_checker: Optional[ConfidenceInterval] = None
        if confidence_interval_checker:
            self.ci_checker = ConfidenceInterval(
                min_sample_count, max_interval_width, confidence_level
            )
        self.batch_data: dict = {'sum': 0, 'count': 0, 'next_batch_time': 0.0}
        self.final_ci: Optional[tuple[float, float]] = None

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
            
            # Track batch data for confidence interval computation
            if self.confidence_interval_checker and self.ci_checker and event.time > self.transient_end_time:
                # Initialize batch tracking on first sample after transient
                if self.batch_data['next_batch_time'] == 0.0:
                    self.batch_data['next_batch_time'] = event.time + self.batch_size
                
                # Accumulate total users for current batch
                num_total_users = self.car_sharing_system.sampled_num_total_users[-1][1]
                self.batch_data['sum'] += num_total_users
                self.batch_data['count'] += 1
                
                # Check if batch is complete
                if event.time >= self.batch_data['next_batch_time']:
                    if self.batch_data['count'] > 0:
                        batch_avg_total_users = self.batch_data['sum'] / self.batch_data['count']
                        self.ci_checker.add_data_point(batch_avg_total_users)
                        
                        # Check if CI is achieved
                        if self.ci_checker.has_enough_data() and self.final_ci is None:
                            final, ci = self.ci_checker.compute_interval()
                            if final:
                                self.final_ci = ci
                                print(f"Confidence interval achieved at time {event.time:.0f}: {ci}")
                    
                    # Reset batch for next interval
                    self.batch_data['sum'] = 0
                    self.batch_data['count'] = 0
                    self.batch_data['next_batch_time'] = event.time + self.batch_size

            # Handle event based on its type
            if event.event_type == EventType.USER_REQUEST:
                origin_map, dest_map = self.get_origin_destination_maps(current_time)
                EventHandler.handle_user_request(event, self.cars, self.car_sharing_system,
                                                self.stations, self.fes, current_time,
                                                self.min_trip_distance, origin_map, dest_map,
                                                map_unit_km=self.map_unit_km,
                                                interarrival_type=self.interarrival_type,
                                                interarrival_params=self.interarrival_params)
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

    def get_origin_destination_maps(self, time_min: float) -> tuple[np.ndarray, np.ndarray]:
        return self.demand_prob_map, self.demand_prob_map
                
    def get_statistics(self) -> dict[str, float]:
        # Compute interarrival statistics
        interarrival_times = self.car_sharing_system.interarrival_times
        if len(interarrival_times) > 0:
            mean_interarrival = np.mean(interarrival_times)
            std_interarrival = np.std(interarrival_times, ddof=1) if len(interarrival_times) > 1 else 0.0
            cv_interarrival = std_interarrival / mean_interarrival if mean_interarrival > 0 else 0.0
        else:
            mean_interarrival = 0.0
            std_interarrival = 0.0
            cv_interarrival = 0.0
        
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
            "mean_interarrival_time": mean_interarrival,
            "std_interarrival_time": std_interarrival,
            "cv_interarrival_time": cv_interarrival,
        }
        return stats
    
    def plot_arrival_histogram(self):
        arrival_times = self.car_sharing_system.arrival_times
        
        if len(arrival_times) == 0:
            print("No arrival data to plot.")
            return
        
        # Create bins for the entire simulation period
        # Use bins of 60 minutes (1 hour) for better visualization
        bin_width = 60.0  # minutes
        num_bins = int(np.ceil(self.simulation_time / bin_width))
        bins = np.linspace(0, self.simulation_time, num_bins + 1)
        
        # Count arrivals in each bin
        counts, bin_edges = np.histogram(arrival_times, bins=bins)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Histogram of arrivals per time bin
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax1.bar(bin_centers, counts, width=bin_width * 0.9, alpha=0.7, edgecolor='black')
        ax1.axhline(y=np.mean(counts), color='r', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(counts):.2f}')
        ax1.axhline(y=np.mean(counts) + np.std(counts), color='orange', linestyle=':', linewidth=1.5,
                   label=f'Mean ± Std: {np.mean(counts):.2f} ± {np.std(counts):.2f}')
        ax1.axhline(y=np.mean(counts) - np.std(counts), color='orange', linestyle=':', linewidth=1.5)
        ax1.set_xlabel(f'Time (minutes)', fontsize=12)
        ax1.set_ylabel('Number of Arrivals', fontsize=12)
        ax1.set_title(f'Arrival Pattern: Number of Arrivals per {bin_width:.0f}-Minute Interval\n' + 
                     f'Distribution: {self.interarrival_type.value}', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Distribution of arrivals per bin (to show variability)
        ax2.hist(counts, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
        ax2.axvline(x=np.mean(counts), color='r', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(counts):.2f}')
        ax2.axvline(x=np.mean(counts) + np.std(counts), color='orange', linestyle=':', linewidth=1.5)
        ax2.axvline(x=np.mean(counts) - np.std(counts), color='orange', linestyle=':', linewidth=1.5,
                   label=f'Std: {np.std(counts):.2f}')
        ax2.set_xlabel('Arrivals per Bin', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'Distribution of Arrival Counts Across Bins', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'arrival_pattern_histogram_{self.interarrival_type.value}.png', dpi=300)
        plt.show()
        
        # Print bin statistics
        print("\n" + "="*70)
        print("ARRIVAL PATTERN STATISTICS")
        print("="*70)
        print(f"Bin width: {bin_width:.0f} minutes")
        print(f"Number of bins: {num_bins}")
        print(f"Mean arrivals per bin: {np.mean(counts):.2f}")
        print(f"Std arrivals per bin: {np.std(counts):.2f}")
        print(f"Min arrivals in a bin: {np.min(counts)}")
        print(f"Max arrivals in a bin: {np.max(counts)}")
        print("="*70)
    
    def plot_transient_phase(self):
        samples = self.car_sharing_system.get_num_total_users_over_time()
        times = samples[0]
        queue_lengths = samples[1]

        plt.figure(figsize=(8, 5))
        plt.step(times, queue_lengths, where='post')

        if self.transient_end_time is not None:
            plt.axvline(x=self.transient_end_time, color='r', linestyle='--', label='Transient End Time')

        plt.title(f'Number of Total Users Over Time - {self.interarrival_type.value}', fontsize=14, fontweight='bold')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Number of Total Users')
        plt.legend()
        plt.grid()
        plt.savefig(f'transient_phase_detection_{self.interarrival_type.value}.png', dpi=300)
        plt.show()

    def print_statistics(self):
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
        print("\n--- Interarrival Time Statistics ---")
        print(f"Mean interarrival time: {stats['mean_interarrival_time']:.4f} minutes")
        print(f"Std interarrival time: {stats['std_interarrival_time']:.4f} minutes")
        print(f"Total interarrival samples: {len(self.car_sharing_system.interarrival_times)}")
        print("="*70)
        
        # Print confidence interval if computed
        if self.confidence_interval_checker and self.ci_checker:
            print("\n" + "="*70)
            print("CONFIDENCE INTERVAL ANALYSIS")
            print("="*70)
            num_batches = self.ci_checker.get_sample_size()
            print(f"Number of batches collected: {num_batches}")
            print(f"Batch size: {self.batch_size} minutes")
            print(f"Transient period: {self.transient_end_time:.0f} minutes")
            
            if num_batches > 0:
                mean = self.ci_checker.average
                print(f"Mean total users in system: {mean:.2f}")
                
                # Check if CI has been computed
                if self.final_ci is not None:
                    lower, upper = self.final_ci
                    ci_width = upper - lower
                    print(f"Confidence interval ({self.confidence_level*100:.0f}%): [{lower:.2f}, {upper:.2f}]")
                    print(f"CI width: {ci_width:.2f}")
                    print(f"Relative width: {(ci_width/mean*100):.2f}%")
                    print(f"Status: ✓ Achieved")
                elif self.ci_checker.has_enough_data():
                    result = self.ci_checker.compute_interval()
                    if result:
                        final, (lower, upper) = result
                        ci_width = upper - lower
                        print(f"Confidence interval ({self.confidence_level*100:.0f}%): [{lower:.2f}, {upper:.2f}]")
                        print(f"CI width: {ci_width:.2f}")
                        print(f"Relative width: {(ci_width/mean*100):.2f}%")
                        print(f"Status: {'✓ Achieved' if final else '⚠ Not yet achieved'}")
                    else:
                        print(f"Status: ⚠ Insufficient data for CI computation")
                else:
                    print(f"Status: ⚠ Not enough samples (need at least {self.min_sample_count})")
            else:
                print(f"Status: ⚠ No batches collected after transient period")
            
            print("="*70)
        
        # Plot arrival histogram to show distribution differences
        self.plot_arrival_histogram()

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

    # Confidence Interval parameters
    CONFIDENCE_LEVEL = 0.95
    MAX_INTERVAL_WIDTH = 0.05
    MIN_SAMPLES_COUNT = 10

    # Boolean Parameters
    ENABLE_CONGESTION = True 
    CONFIDENCE_INTERVAL_CHECKER = True

    # Interarrival exponential distribution
    # interarrival_type = DistributionType.EXPONENTIAL
    # interarrival_params = {"lambda": 4.0}

    # Interarrival hyperexponential distribution
    # interarrival_type = DistributionType.HYPEREXPONENTIAL
    # interarrival_params = {"lambdas": [6.0, 4.0, 2.0], "probabilities": [0.6, 0.2, 0.2]}

    # Interarrival erlang-k distribution
    # interarrival_type = DistributionType.ERLANG_K
    # interarrival_params = {"lambda": 12.0, "k": 3}

    # Interarrival pareto distribution
    interarrival_type = DistributionType.PARETO
    interarrival_params = {"alpha": (5.0/3.0), "scale": 0.1}

    all_configs = list(product(
        SEED, MAP_SEED, NUM_CARS, NUM_STATIONS, MAX_STATION_CAPACITY,
        SIMULATION_TIME, MAX_AUTONOMY, CHARGING_RATE,
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
        print(f"  Confidence intervals:               {"Enabled" if CONFIDENCE_INTERVAL_CHECKER else "Disabled"}")
        print(f"  Confidence level:                   {CONFIDENCE_LEVEL}")
        print(f"  Max CI width:                       {MAX_INTERVAL_WIDTH}")
        print(f"  Min sample count:                   {MIN_SAMPLES_COUNT}")

        print("\n--- Interarrival Distribution ---")
        print(f"  Type:                             {interarrival_type}")
        print(f"  Parameters:                       {interarrival_params}")
        
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
            confidence_interval_checker=CONFIDENCE_INTERVAL_CHECKER,
            enable_congestion=ENABLE_CONGESTION,
            confidence_level=CONFIDENCE_LEVEL,
            max_interval_width=MAX_INTERVAL_WIDTH,
            min_sample_count=MIN_SAMPLES_COUNT,
            interarrival_type=interarrival_type,
            interarrival_params=interarrival_params,
        )

        simulation_engine.event_loop()
        elapsed = time.time() - start_time
        print(f"\n{'=' * 80}")
        print(f"Run {idx} completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print(f"{'=' * 80}\n")
        simulation_engine.print_statistics()
