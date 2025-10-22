"""
Rationale of the Simulation:

The simulation models a car-sharing system based on a fleet of electric vehicles (EVs) distributed across the city at various charging 
stations. Throughout the day, user requests arrive at varying rates, reflecting typical urban mobility patterns: higher demand is observed 
during morning commuting hours (e.g., people traveling to work or school), around lunchtime, and again during the evening commute 
(typically 5-7 p.m.).

The population and key destination hubs (such as workplaces and schools) follow distinct spatial distributions, which together define the 
predominant travel flows between residential and activity areas. Each user can request a vehicle through the mobile application. If no 
nearby vehicle is currently available, the user is placed in a waiting queue. When a suitable car becomes available within a defined 
proximity threshold, the user is notified and proceeds to the assigned vehicle.

Once the trip begins, the car is driven to the specified destination and parked either at the nearest available charging station or 
directly at the destination if no charging station is nearby. Vehicles parked at charging stations automatically restore their battery 
autonomy.

To balance supply and demand dynamically, vehicles can be relocated during the day—potentially multiple times—to better match the spatial 
distribution of requests. Relocations always occur between charging stations.

When the car autonomy is low, then it has to be relocated to a recharging station and, until it reaches a minimum autonomy, 
it has to be marked as unavailable.

Trip origins and destinations, as well as trip lengths, are stochastically generated based on time-dependent spatial distributions. For 
instance, during morning hours, trip origins follow the population density distribution while destinations follow the workplace distribution; 
this pattern is reversed in the evening. User request arrivals are modeled using a Poisson process, while vehicle speed varies throughout 
the day as a function of traffic density (i.e., the number of active cars), thereby simulating congestion effects.

The locations of charging stations are fixed and determined according to both population and workplace/school densities, ensuring that 
high-demand areas for trip start and end points are well covered.

The system also incorporates several operational constraints and behavioral parameters, including:

	- Maximum waiting time: the maximum time a user tolerates before a nearby vehicle becomes available (after which they abandon the request).

	- Maximum pickup distance: the maximum allowable distance between a user's location and the assigned vehicle.

	- Maximum destination-charging distance: the maximum acceptable distance between the trip destination and a charging station 
    (beyond which the car is parked directly at the destination).

	- Vehicle autonomy: vehicles with low battery levels must be relocated to a charging station or marked as unavailable.

	- Minimum trip distance: users will not choose a shared car for very short trips below a given distance threshold.
	
	- Recharging rate: the rate at which the cars are recharged.

    - Minimum autonomy: battery percentage under which the car becomes unavailable.

Finally, several key performance indicators (KPIs) are monitored to evaluate system efficiency and user satisfaction, including:

	- Vehicle availability rate,

	- Vehicle utilization rate,

	- Average user waiting time before a nearby car becomes available,

	- Average walking distance from user location to assigned car,

	- Average trip distance traveled per user,

	- Utilization rate of charging station parking spots,

	- Number of active users (i.e., users requesting cars),

	- Spatial distribution of demand, useful for optimizing relocation strategies.
"""
from enum import Enum
from queue import PriorityQueue
from typing import Optional
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import time
from itertools import product


# ------------------------------------------------------------------------------
# EVENT DEFINITION
# ------------------------------------------------------------------------------
class EventType(Enum):
    """Enumeration of event types in the car sharing system."""
    USER_REQUEST = "USER_REQUEST"
    USER_ABANDON = "USER_ABANDON"
    CAR_PARKING = "CAR_PARKING"
    CAR_AVAILABLE = "CAR_AVAILABLE"
    CAR_RELOCATE = "CAR_RELOCATE"

class Event:
    """Class representing an event in the car sharing system."""
    def __init__(self, event_type: EventType, time: float, user_id: Optional[int] = None, 
                 car_id: Optional[int] = None, station_id: Optional[int] = None):
        self.event_type = event_type
        self.time = time  # in minutes
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
    """Class representing a car in the car sharing system."""
    def __init__(self, car_id: int, location: tuple, max_autonomy: float, charging_rate: float, min_autonomy: float, 
                 max_destination_charging_distance: float, speed: float, station_id: int):
        self.car_id = car_id
        self.location = location  # (x, y) coordinates
        self.autonomy = max_autonomy  # in hm
        self.max_autonomy = max_autonomy  # in hm
        self.charging_rate = charging_rate  # in hm per minute
        self.min_autonomy = min_autonomy  # in hm
        self.max_destination_charging_distance = max_destination_charging_distance
        self.speed = speed
        self.station_id = station_id
        self.available: bool = True
        self.assigned_user_id: Optional[int] = None
        self.charging: bool = True
        self.relocating: bool = False
        self.relocating_station_id: Optional[int] = None

    def drive(self, distance: float, destination: tuple) -> bool:
        """Drive the car to a new location."""
        self.location = destination
        self.autonomy -= distance
        return True

    def charge(self, time: float):
        """Charge the car for a given time interval."""
        if self.charging:
            self.autonomy = min(self.autonomy + self.charging_rate * time, self.max_autonomy)
            if self.autonomy >= self.min_autonomy:
                self.available = True
    
    def park_in_station(self, station_id: int):
        self.station_id = station_id

    def unpark_from_station(self):
        self.station_id = None

    def assign_user(self, user_id: int):
        if self.available:
            self.charging = False
            self.assigned_user_id = user_id
            self.available = False
        else:
            print("Car is not available for assignment.")
    
    def release_user(self):
        if self.assigned_user_id is not None:
            self.assigned_user_id = None
            if self.autonomy < self.min_autonomy:
                self.available = False
            else:
                self.available = True
        else:
            print("No user is assigned to this car.")

    def is_available(self) -> bool:
        return self.available

    def relocate(self, station_id: int):
        """Mark the car as relocating to a given station."""
        self.relocating = True
        self.relocating_station_id = station_id

    def is_relocating(self) -> bool:
        return self.relocating

    def get_relocating_station_id(self) -> int:
        return self.relocating_station_id
    
    def stop_relocating(self):
        self.relocating = False
        self.relocating_station_id = None


# ------------------------------------------------------------------------------
# USER
# ------------------------------------------------------------------------------
class User:
    """Class representing a user in the car sharing system."""
    def __init__(self, user_id: int, location: tuple, destination: tuple, max_waiting_time: float, 
                 max_pickup_distance: float, request_time: float):
        self.user_id = user_id
        self.location = location  # (x, y) coordinates
        self.destination = destination  # (x, y) coordinates
        self.max_waiting_time = max_waiting_time  # in minutes
        self.max_pickup_distance = max_pickup_distance  # in hm
        self.request_time = request_time
        self.waiting: bool = True
        self.assigned_car_id: Optional[int] = None
        self.service_time: Optional[float] = None
        self.served: bool = False

    def assign_car(self, car_id: int, service_time: float):
        """User is assigned a car."""
        self.assigned_car_id = car_id
        self.waiting = False
        self.service_time = service_time

    def resign_car(self):
        self.assigned_car_id = None
        self.served = True

    def request_car(self):
        """User makes a car request."""
        self.waiting = True

    def abandon_request(self):
        """User abandons the car request since too much time has passed."""
        self.waiting = False
        self.assigned_car_id = None

    def put_in_queue(self):
        """User is put in the waiting queue."""
        self.waiting = True

    def is_waiting(self) -> bool:
        return self.waiting

    def is_served(self) -> bool:
        return self.served

    def waiting_time(self) -> float:
        """Calculate the waiting time of the user, between the request and service time."""
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
    """Class representing a charging station in the car sharing system."""
    def __init__(self, station_id: int, location: tuple, capacity: int):
        self.station_id = station_id
        self.location = location  # (x, y) coordinates
        self.capacity = capacity  # number of parking spots
        self.occupied_spots: int = 0
        self.parked_cars: list[int] = []  # list of car_ids
        self.full: bool = False

    def park_car(self, car_id: int) -> bool:
        """Park a car at the charging station, if there is available spot."""
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
        """Remove a car from the charging station."""
        if car_id in self.parked_cars:
            self.parked_cars.remove(car_id)
            self.occupied_spots -= 1
            self.full = False
            return True
        else:
            print("Car not found at the charging station.")
            return False

    def is_full(self) -> bool:
        return self.full
        

# ------------------------------------------------------------------------------
# FUTURE EVENT SET
# ------------------------------------------------------------------------------
class FutureEventSet:
    """Class representing the future event set of the car sharing system."""
    def __init__(self):
        self.events: PriorityQueue[Event] = PriorityQueue()
        self.event_count: int = 0

    def schedule(self, event: Event):
        self.events.put(event)
        self.event_count += 1

    def get_next_event(self) -> Optional[Event]:
        """Retrieve the next non-cancelled event from the future event set."""
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
    

# -------------------------------------------------------------------------------
# CAR SHARING SYSTEM
# -------------------------------------------------------------------------------
class CarSharingSystem:
    """Class representing the car sharing system and its statistics."""
    def __init__(self, cars: list[Car], stations: list[ChargingStation]):
        self.cars = cars
        self.stations = stations

        # Initialize statistics
        self.total_user_requests: int = 0
        self.total_abandoned_requests: int = 0
        self.total_waiting_time: float = 0.0
        self.total_trip_distance: float = 0.0

        # Relocation tracking
        self.total_relocations: int = 0
        self.last_relocation_time: float = 0.0
        self.total_relocated_cars: int = 0

        # Client tracking
        self.waiting_users: dict[int, User] = {}
        self.active_users: dict[int, User] = {}

        # Car tracking
        self.charging_cars: dict[int, Car] = {car.car_id: car for car in cars if car.charging}
        self.dispersed_cars: dict[int, Car] = {}

        # Sample statistics
        self.sampled_waiting_queue_lengths: list[tuple[float, int]] = []  # (time, queue_length)
        self.sampled_num_dispersed_cars: list[tuple[float, int]] = []  # (time, num_dispersed_cars)
        self.sampled_perc_abandoned_requests: list[tuple[float, int]] = []  # (time, total_abandoned_requests)
        self.num_samples: int = 0
        self.total_queue_length: int = 0
        self.sampled_avg_queue_length: list[tuple[float, float]] = []

        # Initialize time-weighted statistics
        self.area_under_available_cars: float = 0.0
        self.area_under_used_cars: float = 0.0
        self.area_under_used_station: float = 0.0
        self.last_event_time: float = 0.0
        self.last_available_car_count: int = sum(1 for c in cars if c.is_available())
        self.last_used_car_count: int = 0
        self.last_used_station_spots: int = len(cars)

    def update_time_weighted_statistics(self, new_time: float):
        """Update time-weighted statistics based on the time elapsed since the last event."""
        time_delta = new_time - self.last_event_time
        self.area_under_available_cars += self.last_available_car_count * time_delta
        self.area_under_used_cars += self.last_used_car_count * time_delta
        self.area_under_used_station += self.last_used_station_spots * time_delta
        self.last_event_time = new_time

    def sample_statistics(self, current_time: float):
        self.num_samples += 1
        self.total_queue_length += len(self.waiting_users)
        self.sampled_waiting_queue_lengths.append((current_time, len(self.waiting_users)))
        self.sampled_num_dispersed_cars.append((current_time, len(self.dispersed_cars)))
        self.sampled_perc_abandoned_requests.append((current_time, 
                                                     self.total_abandoned_requests / max(1, self.total_user_requests)))
        self.sampled_avg_queue_length.append((current_time, self.total_queue_length / self.num_samples))

    def update_speed(self, new_speed: float):
        """Update the speed of all cars in the system."""
        for car in self.cars:
            car.speed = new_speed

    def add_trip_distance(self, distance: float):
        """Update total trip distance."""
        self.total_trip_distance += distance

    def add_waiting_time(self, user: User):
        """Update total waiting time based on a user's waiting time."""
        self.total_waiting_time += user.waiting_time()

    def add_charging_car(self, car: Car):
        """Add a car to the charging cars dictionary."""
        self.charging_cars[car.car_id] = car

    def remove_charging_car(self, car: Car):
        """Remove a car from the charging cars dictionary."""
        if car.car_id in self.charging_cars:
            del self.charging_cars[car.car_id]

    def charge_cars(self, interval: float):
        """Charge all cars that are currently charging for a given time interval."""
        for car in self.charging_cars.values():
            car.charge(interval)

    def add_dispersed_car(self, car: Car):
        """Add a car to the dispersed cars dictionary."""
        self.dispersed_cars[car.car_id] = car

    def remove_dispersed_car(self, car: Car):
        """Remove a car from the dispersed cars dictionary."""
        if car.car_id in self.dispersed_cars:
            del self.dispersed_cars[car.car_id]

    def get_dispersed_cars(self) -> list[Car]:
        return list(self.dispersed_cars.values())

    def get_dispersed_car_count(self) -> int:
        return len(self.dispersed_cars)

    def add_active_user(self, user: User):
        """Add a user to the active users dictionary."""
        self.active_users[user.user_id] = user

    def remove_active_user(self, user: User):
        """Remove a user from the active users dictionary."""
        if user.user_id in self.active_users:
            del self.active_users[user.user_id]

    def get_active_user(self, user_id: int) -> Optional[User]:
        return self.active_users.get(user_id, None)

    def add_client_to_waiting(self, user: User):
        """Add a user to the waiting users dictionary."""
        self.waiting_users[user.user_id] = user

    def remove_client_from_waiting(self, user: User):
        """Remove a user from the waiting users dictionary."""
        if user.user_id in self.waiting_users:
            del self.waiting_users[user.user_id]
    
    def get_waiting_clients(self) -> list[User]:
        return list(self.waiting_users.values())
    
    def get_waiting_client_count(self) -> int:
        return len(self.waiting_users)
    
    def update_available_car_count(self):
        self.last_car_count = sum(1 for c in self.cars if c.is_available())

    def update_used_car_count(self):
        self.last_used_car_count = sum(1 for c in self.cars if c.assigned_user_id is not None)

    def user_requested_car(self):
        """Update total user requests."""
        self.total_user_requests += 1

    def user_abandoned(self):
        """Update total abandoned requests."""
        self.total_abandoned_requests += 1

    def occupy_station_spot(self):
        self.last_used_station_spots += 1

    def release_station_spot(self):
        self.last_used_station_spots -= 1

    def relocation_performed(self, time: float, num_relocated_cars: int = 1):
        """Update relocation statistics."""
        self.total_relocations += 1
        self.last_relocation_time = time
        self.total_relocated_cars += num_relocated_cars

    def get_queue_histogram(self) -> dict[int, int]:
        """Get histogram of waiting queue lengths."""
        histogram = {}
        for _, length in self.sampled_waiting_queue_lengths:
            histogram[length] = histogram.get(length, 0) + 1
        return histogram
    
    def get_dispersed_cars_histogram(self) -> dict[int, int]:
        """Get histogram of number of dispersed cars."""
        histogram = {}
        for _, num in self.sampled_num_dispersed_cars:
            histogram[num] = histogram.get(num, 0) + 1
        return histogram

    def get_avg_waiting_queue_length_over_time(self) -> tuple[list[float], list[float]]:
        """Get 2 lists: times of sampling and average waiting queue lengths over time."""
        times = [t for t, _ in self.sampled_avg_queue_length]
        lengths = [l for _, l in self.sampled_avg_queue_length]
        return times, lengths

    def get_percent_abandoned_requests_over_time(self) -> tuple[list[float], list[float]]:
        """Get 2 lists: times of sampling and percentage of abandoned requests over time."""
        times = [t for t, _ in self.sampled_perc_abandoned_requests]
        percentages = [p for _, p in self.sampled_perc_abandoned_requests]
        return times, percentages


# ------------------------------------------------------------------------------
# EVENT HANDLERS
# ------------------------------------------------------------------------------
class EventHandler:
    """Class containing static methods to handle different event types."""
    @staticmethod
    def handle_user_request(event: Event, cars: list[Car], car_sharing_system: CarSharingSystem, 
                            stations: list[ChargingStation], future_event_set: FutureEventSet, current_time: float,
                            arrival_rate: float, min_trip_distance: float, population_prob_map: np.ndarray, 
                            working_prob_map: np.ndarray, waiting_queue_tolerance: float, remaining_daily_relocations: int, 
                            last_relocation_time: float = 0.0, min_relocation_interval: float = 60.0) -> bool:
        """Handle a user request event."""
        # Retrieve the user making the request
        user: User = car_sharing_system.get_active_user(event.user_id)
        if user is None:
            return False
        user.request_car()
        car_sharing_system.user_requested_car()
        # Try to assign a car
        assigned = False
        # Find the nearest available car to the user
        available_cars: list[tuple[Car, float]] = []
        for c in cars:
            if c.is_available():
                dist = ((c.location[0] - user.location[0]) ** 2 + (c.location[1] - user.location[1]) ** 2) ** 0.5
                available_cars.append((c, dist))
        if available_cars:
            nearest_car, min_distance = min(available_cars, key=lambda x: x[1])
            if min_distance <= user.max_pickup_distance:
                car = nearest_car
                if car.charging:
                    # Remove car from charging cars
                    car_sharing_system.remove_charging_car(car)
                if car.station_id is not None:
                    # Unpark car from station
                    station: ChargingStation = stations[car.station_id]
                    station.remove_car(car.car_id)
                    car.unpark_from_station()
                    car_sharing_system.release_station_spot()
                # Assign car to user and vice versa
                car.assign_user(user.user_id)
                user.assign_car(car.car_id, current_time)
                car_sharing_system.update_available_car_count()
                car_sharing_system.update_used_car_count()
                car_sharing_system.add_waiting_time(user)
                assigned = True
                # Schedule car available event after trip
                trip_distance = ((user.location[0] - user.destination[0]) ** 2 + (user.location[1] - user.destination[1]) ** 2) ** 0.5
                car.drive(trip_distance, user.destination)
                car_sharing_system.add_trip_distance(trip_distance)
                travel_time = (trip_distance * 60) / (10 * car.speed)
                car_parking_event = Event(EventType.CAR_PARKING, current_time + travel_time, user_id=user.user_id, car_id=car.car_id)
                future_event_set.schedule(car_parking_event)
        # No car available within max pickup distance
        if not assigned:
            # Put user in waiting queue
            user.put_in_queue()
            car_sharing_system.add_client_to_waiting(user)
            # Schedule abandon event after max waiting time
            abandon_event = Event(EventType.USER_ABANDON, current_time + user.max_waiting_time, user_id=user.user_id)
            future_event_set.schedule(abandon_event)
            # Check if relocation is needed and possible
            if remaining_daily_relocations > 0 and car_sharing_system.get_waiting_client_count() >= waiting_queue_tolerance \
               and (current_time - last_relocation_time) >= min_relocation_interval:
                # Schedule immediate relocation event
                relocate_event = Event(EventType.CAR_RELOCATE, current_time)
                future_event_set.schedule(relocate_event)

        # Schedule next user request
        inter_arrival_time = np.random.exponential(1 / arrival_rate) # Suppose poisson process with rate = arrival_rate
        next_request_time = current_time + inter_arrival_time
        next_user_id = user.user_id + 1
        # Determine next user location and destination based on time of day
        if 0 <= next_request_time % (24 * 60) <= 12 * 60:
            departure_prob_map = population_prob_map
            destination_prob_map = working_prob_map
        else:
            departure_prob_map = working_prob_map
            destination_prob_map = population_prob_map
        travel_distance = 0.0
        next_user_location = MappingUtilities.get_random_location_from_prob_map(departure_prob_map)
        # Ensure minimum trip distance is met
        while travel_distance < min_trip_distance:
            next_user_destination = MappingUtilities.get_random_location_from_prob_map(destination_prob_map)
            travel_distance = ((next_user_location[0] - next_user_destination[0]) ** 2
                               + (next_user_location[1] - next_user_destination[1]) ** 2) ** 0.5
        next_user = User(next_user_id, next_user_location, next_user_destination, user.max_waiting_time, 
                         user.max_pickup_distance, next_request_time)
        car_sharing_system.add_active_user(next_user)
        user_request_event = Event(EventType.USER_REQUEST, next_request_time, user_id=next_user_id)
        future_event_set.schedule(user_request_event)
        return assigned
    
    @staticmethod
    def handle_user_abandon(event: Event, car_sharing_system: CarSharingSystem) -> bool:
        """Handle a user abandon event."""
        # Retrieve the user abandoning the request
        user: User = car_sharing_system.get_active_user(event.user_id)
        if user is None:
            return False
        if user.is_waiting():
            user.abandon_request()
            car_sharing_system.user_abandoned()
            car_sharing_system.remove_client_from_waiting(user)
            car_sharing_system.remove_active_user(user)
            return True
        return False
    
    @staticmethod
    def handle_car_parking(event: Event, cars: list[Car], car_sharing_system: CarSharingSystem,
                           stations: list[ChargingStation], future_event_set: FutureEventSet, current_time: float, 
                           dispersed_cars_tolerance: float, remaining_daily_relocations: int, 
                           last_relocation_time: float = 0.0, min_relocation_interval: float = 60.0) -> bool:
        """Handle a car parking event."""
        # Retrieve the car being parked
        car: Car = cars[event.car_id]
        if not car.is_relocating():
            car.release_user()
            # Retrieve the user assigned to the car
            user: User = car_sharing_system.get_active_user(event.user_id)
            if user is not None:
                user.resign_car()
                car_sharing_system.remove_active_user(user)
            car_sharing_system.update_used_car_count()
            # Find the closest not full charging station
            nearest_station = None
            min_distance = float('inf')
            for station in stations:
                distance_to_station = ((car.location[0] - station.location[0]) ** 2 
                                       + (car.location[1] - station.location[1]) ** 2) ** 0.5
                if distance_to_station < min_distance and not station.is_full():
                    min_distance = distance_to_station
                    nearest_station = station
            """Park the car at the nearest station if within max_destination_charging_distance,
             otherwise park exactly at destination without charging"""
            if nearest_station is not None and min_distance <= car.max_destination_charging_distance:
                nearest_station.park_car(car.car_id)
                car.park_in_station(nearest_station.station_id)
                car_sharing_system.occupy_station_spot()
                car_sharing_system.add_charging_car(car)
                car.charging = True
                car.location = nearest_station.location
            if car.is_available():
                # Schedule car available event if the autonomy is sufficient
                car_available_event = Event(EventType.CAR_AVAILABLE, current_time, car_id=car.car_id)
                future_event_set.schedule(car_available_event)
            elif car.charging:
                # Schedule car available event when autonomy reaches min_autonomy
                time_to_min_autonomy = (car.min_autonomy - car.autonomy) / car.charging_rate
                if time_to_min_autonomy < 0:
                    time_to_min_autonomy = 0.0
                car_available_event = Event(EventType.CAR_AVAILABLE, current_time + time_to_min_autonomy, car_id=car.car_id)
                future_event_set.schedule(car_available_event)
            else:
                # Car unavailable and not charging, so add to dispersed cars
                car_sharing_system.add_dispersed_car(car)
                # Check if relocation is needed and possible
                if remaining_daily_relocations > 0 and car_sharing_system.get_dispersed_car_count() >= dispersed_cars_tolerance \
                and (current_time - last_relocation_time) >= min_relocation_interval:
                    # Schedule relocation event
                    relocate_event = Event(EventType.CAR_RELOCATE, current_time)
                    future_event_set.schedule(relocate_event)
            return True
        else:
            # Relocation parking
            station: ChargingStation = stations[car.get_relocating_station_id()]
            car.park_in_station(station.station_id)
            car.stop_relocating()
            car_sharing_system.occupy_station_spot()
            car_sharing_system.add_charging_car(car)
            car.charging = True
            car.location = station.location
            # Schedule car available event when autonomy reaches min_autonomy
            time_to_min_autonomy = (car.min_autonomy - car.autonomy) / car.charging_rate
            if time_to_min_autonomy < 0:
                time_to_min_autonomy = 0.0
            car_available_event = Event(EventType.CAR_AVAILABLE, current_time + time_to_min_autonomy, car_id=car.car_id)
            future_event_set.schedule(car_available_event)
            return True
    
    @staticmethod
    def handle_car_available(event: Event, cars: list[Car], stations: list[ChargingStation], car_sharing_system: CarSharingSystem,
                             future_event_set: FutureEventSet, current_time: float) -> bool:
        """Handle a car available event."""
        # Retrieve the car becoming available
        car: Car = cars[event.car_id]
        if car.is_available():
            car_sharing_system.update_available_car_count()
            # Check if there are waiting users
            waiting_clients = car_sharing_system.get_waiting_clients()
            if not waiting_clients:
                return False
            # Find the closest waiting user
            distance_to_users: list[tuple[User, float]] = []
            for user in waiting_clients:
                dist = ((car.location[0] - user.location[0]) ** 2 + (car.location[1] - user.location[1]) ** 2) ** 0.5
                distance_to_users.append((user, dist))
            nearest_user, min_distance = min(distance_to_users, key=lambda x: x[1])
            # Assign car to user if within max pickup distance
            if min_distance <= nearest_user.max_pickup_distance:
                user = nearest_user
                if car.charging:
                    # Remove car from charging cars
                    car_sharing_system.remove_charging_car(car)
                if car.station_id is not None:
                    # Unpark car from station
                    station: ChargingStation = stations[car.station_id]
                    station.remove_car(car.car_id)
                    car.unpark_from_station()
                    car_sharing_system.release_station_spot()
                # Assign car to user and vice versa
                car.assign_user(user.user_id)
                user.assign_car(car.car_id, current_time)
                car_sharing_system.update_available_car_count()
                car_sharing_system.update_used_car_count()
                car_sharing_system.add_waiting_time(user)
                car_sharing_system.remove_client_from_waiting(user)
                # Cancel USER_ABANDON event if scheduled
                for evt in future_event_set.events.queue:
                    if evt.event_type == EventType.USER_ABANDON and evt.user_id == user.user_id:
                        evt.cancel()
                # Schedule car parking event after trip
                trip_distance = ((user.location[0] - user.destination[0]) ** 2 
                                 + (user.location[1] - user.destination[1]) ** 2) ** 0.5
                car.drive(trip_distance, user.destination)
                car_sharing_system.add_trip_distance(trip_distance)
                travel_time = (trip_distance * 60) / (10 * car.speed)
                car_parking_event = Event(EventType.CAR_PARKING, current_time + travel_time, car_id=car.car_id)
                future_event_set.schedule(car_parking_event)
                return True
        return False
    
    @staticmethod
    def handle_car_relocate(event: Event, cars: list[Car], stations: list[ChargingStation], car_sharing_system: CarSharingSystem,
                             future_event_set: FutureEventSet, total_cars_to_relocate: int, target_prob_map) -> bool:
        """Handle a car relocate event."""
        # Helper to relocate a car to the nearest non-full station sampled from the target_prob_map
        def relocate_to_nearest_station(car: Car, remove_from_dispersed: bool = False) -> int:
            nearest_station = None
            min_distance_to_station = float('inf')
            # Sample a target location and find nearest station with available spot
            while True:
                target_location = MappingUtilities.get_random_location_from_prob_map(target_prob_map)
                nearest_station = None
                min_distance_to_station = float('inf')
                for station in stations:
                    distance_to_station = ((target_location[0] - station.location[0]) ** 2 + (target_location[1] - station.location[1]) ** 2) ** 0.5
                    if distance_to_station < min_distance_to_station:
                        min_distance_to_station = distance_to_station
                        nearest_station = station
                if nearest_station is None:
                    return 0
                if not nearest_station.is_full():
                    break
            # Mark car as relocating
            car.relocate(nearest_station.station_id)
            # Drive car to station and update states
            car.drive(min_distance_to_station, nearest_station.location)
            nearest_station.park_car(car.car_id) # Reserve spot at station
            if remove_from_dispersed:
                # Remove car from dispersed cars
                car_sharing_system.remove_dispersed_car(car)
            # Schedule car parking event when autonomy reaches min_autonomy
            travel_time = (min_distance_to_station * 60) / (10 * car.speed)
            scheduled_time = travel_time + event.time
            if scheduled_time < event.time:
                scheduled_time = event.time
            car_parking_event = Event(car_id=car.car_id, event_type=EventType.CAR_PARKING, time=scheduled_time)
            future_event_set.schedule(car_parking_event)
            return 1

        # Relocate first all dispersed cars
        relocated_cars = 0
        for car in car_sharing_system.get_dispersed_cars():
            relocated_cars += relocate_to_nearest_station(car, remove_from_dispersed=True)

        # If more cars need to be relocated, select additional cars from most unlikely areas
        while relocated_cars < total_cars_to_relocate:
            # Sample a random point from the negative target distribution
            negative_target_prob_map = 1.0 - target_prob_map
            total = np.sum(negative_target_prob_map)
            negative_target_prob_map /= total
            negative_target_location = MappingUtilities.get_random_location_from_prob_map(negative_target_prob_map)
            # Find nearest car to target location
            nearest_car = None
            min_distance = float('inf')
            for car in cars:
                distance_to_car = ((negative_target_location[0] - car.location[0]) ** 2 + (negative_target_location[1] - car.location[1]) ** 2) ** 0.5
                if distance_to_car < min_distance and car.is_available():
                    min_distance = distance_to_car
                    nearest_car = car
            if nearest_car is not None:
                # Relocate the car
                car = nearest_car
                car.available = False
                if car.charging:
                    # Remove car from charging cars
                    car_sharing_system.remove_charging_car(car)
                if car.station_id is not None:
                    # Unpark car from station
                    station: ChargingStation = stations[car.station_id]
                    station.remove_car(car.car_id)
                    car.unpark_from_station()
                    car_sharing_system.release_station_spot()
                # Use helper to sample target station and perform relocation
                relocated_cars += relocate_to_nearest_station(car, remove_from_dispersed=False)
        if relocated_cars == total_cars_to_relocate:
            return True
        return False


# ------------------------------------------------------------------------------
# MAPPING UTILITIES
# ------------------------------------------------------------------------------
class MappingUtilities:
    """Class containing static methods for creating and visualizing maps."""
    @staticmethod
    def create_population_density(width: int = 100, height: int = 100, n_centers: int = 5) -> np.ndarray:
        """Create a synthetic population density map using Gaussian mixtures."""
        x, y = np.meshgrid(np.linspace(0, width, width), 
                        np.linspace(0, height, height))
        pos: np.ndarray = np.dstack((x, y))
        # Initialize null density map
        density: np.ndarray = np.zeros((height, width))
        for _ in range(n_centers):
            # Random center location
            center = [np.random.uniform(-width/2, 1.5*width), 
                    np.random.uniform(-height/2, 1.5*height)]
            # Covariance matrix, with no correlation
            cov = [[1000, 0],
                [0, 1000]]
            rv = multivariate_normal(center, cov)
            # Update density map with Gaussian mixture
            density += rv.pdf(pos)
        # Normalize density
        total = np.sum(density)
        density /= total
        return density 
    
    @staticmethod
    def get_random_location_from_prob_map(prob_map: np.ndarray) -> tuple[int, int]:
        """Sample a random location (x, y) from a given probability map."""
        flat_prob = prob_map.flatten()
        flat_index = np.random.choice(len(flat_prob), p=flat_prob)
        y, x = np.unravel_index(flat_index, prob_map.shape)
        return (x, y)
    
    @staticmethod
    def plot_probability_map(prob_map: np.ndarray, title: str):
        """Plot a given probability map using a heatmap."""
        plt.imshow(prob_map, cmap='hot', interpolation='nearest', origin='lower')
        plt.colorbar(label='Probability Density')
        plt.title(title)
        plt.pause(2)

    @staticmethod
    def plot_parking_prob_map_with_parking_stations(prob_map: np.ndarray, stations: list[ChargingStation]):
        """Plot a probability map with charging station locations overlaid."""
        plt.clf()
        plt.imshow(prob_map, cmap='hot', interpolation='nearest', origin='lower')
        plt.colorbar(label='Probability Density')
        for station in stations:
            plt.scatter(station.location[0], station.location[1], c='blue', marker='o', s=100, label='Charging Station' if 'Charging Station' not in plt.gca().get_legend_handles_labels()[1] else "")
            # show capacity label next to each station
            cap_label = f"Cap: {station.capacity}"
            plt.text(station.location[0] + 0.6, station.location[1] + 0.6, cap_label, color='white', fontsize=8, ha='left', va='bottom',
                 bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=1))
        plt.title("Parking Lot Probability Map with Charging Stations")
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.pause(2)

    @staticmethod
    def plot_simulation_state(event: Event, cars: list[Car], user: Optional[User], stations: list[ChargingStation], 
                              waiting_users: list[User], dispersed_cars: list[Car], speed: float = 40.0, 
                              max_pickup_distance: float = 5.0,):
        """DEBUG Method: Plot the current state of the simulation."""
        plt.clf()
        for car in cars:
            if car.charging and car.is_available():
                plt.scatter(car.location[0], car.location[1], c='blue', alpha=0.5, label='Charging Car' if 'Charging Car' not in plt.gca().get_legend_handles_labels()[1] else "")
            elif car.is_available():
                plt.scatter(car.location[0], car.location[1], c='green', alpha=0.5, label='Available Car' if 'Available Car' not in plt.gca().get_legend_handles_labels()[1] else "")
            elif car in dispersed_cars:
                plt.scatter(car.location[0], car.location[1], c='magenta', alpha=0.5, label='Dispersed Car' if 'Dispersed Car' not in plt.gca().get_legend_handles_labels()[1] else "")
            else:
                plt.scatter(car.location[0], car.location[1], c='red', alpha=0.5, label='Unavailable Car' if 'Unavailable Car' not in plt.gca().get_legend_handles_labels()[1] else "")
        # Plot stations with empty boxes
        for station in stations:
            plt.scatter(station.location[0], station.location[1],
                facecolors='none', edgecolors='black', marker='s', s=100,
                label='Charging Station' if 'Charging Station' not in plt.gca().get_legend_handles_labels()[1] else "")
            # show occupancy label (occupied_spots)
            occ_label = f"{station.occupied_spots}"
            plt.text(station.location[0] + 0.6, station.location[1] + 0.6, occ_label,
                 color='black', fontsize=8, ha='left', va='bottom')
        if event.event_type == EventType.USER_REQUEST and user is not None:
            # Plot user location and destination
            plt.scatter(user.location[0], user.location[1], c='orange', marker='x', s=100, label='User Location')
            plt.scatter(user.destination[0], user.destination[1], c='purple', marker='*', s=100, label='User Destination')
            # Plot max pickup distance circle
            pickup_circle = plt.Circle((user.location[0], user.location[1]), user.max_pickup_distance, color='orange', fill=False, linestyle='--', label='Max Pickup Distance')
            plt.gca().add_artist(pickup_circle)
        if event.event_type == EventType.USER_ABANDON:
            # Plot waiting users and their max pickup distance circles with abandoning user in red
            for w_user in waiting_users:
                color = 'red' if w_user.user_id == event.user_id else 'gray'
                plt.scatter(w_user.location[0], w_user.location[1], c=color, marker='x', s=100, label='Abandoning User' if w_user.user_id == event.user_id and 'Abandoning User' not in plt.gca().get_legend_handles_labels()[1] else 'Waiting User' if 'Waiting User' not in plt.gca().get_legend_handles_labels()[1] else "")
                pickup_circle = plt.Circle((w_user.location[0], w_user.location[1]), w_user.max_pickup_distance, color=color, fill=False, linestyle='--', label='Max Pickup Distance' if 'Max Pickup Distance' not in plt.gca().get_legend_handles_labels()[1] else "")
                plt.gca().add_artist(pickup_circle)
        if event.event_type == EventType.CAR_PARKING:
            # Plot the car that is parking
            car: Car = cars[event.car_id]
            plt.scatter(car.location[0], car.location[1], c='purple', marker='o', s=100, label='Parking Car')
            # Plot max destination charging distance circle
            destination_circle = plt.Circle((car.location[0], car.location[1]), car.max_destination_charging_distance, color='purple', fill=False, linestyle='--', label='Max Destination Charging Distance')
            plt.gca().add_artist(destination_circle)
        if event.event_type == EventType.CAR_AVAILABLE:
            # Plot the car that is becoming available
            car = cars[event.car_id]
            plt.scatter(car.location[0], car.location[1], c='cyan', marker='o', s=100, label='Available Car Now')
            # Plot max pickup distance circle
            pickup_circle = plt.Circle((car.location[0], car.location[1]), max_pickup_distance, color='cyan', fill=False, linestyle='--', label='Max Pickup Distance') 
            plt.gca().add_artist(pickup_circle)
            # Plot waiting users
            for w_user in waiting_users:
                plt.scatter(w_user.location[0], w_user.location[1], c='gray', marker='x', s=100, label='Waiting User' if 'Waiting User' not in plt.gca().get_legend_handles_labels()[1] else "")
        if event.event_type == EventType.CAR_RELOCATE:
            # Plot dispersed cars
            for d_car in dispersed_cars:
                plt.scatter(d_car.location[0], d_car.location[1], c='magenta', marker='o', s=100, label='Dispersed Car' if 'Dispersed Car' not in plt.gca().get_legend_handles_labels()[1] else "")

        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.title(f"Day {int(event.time // (24 * 60))}, {int(event.time // 60) % 24:02}:{int(event.time % 60):02}:{int(event.time % 1 * 60):02} - Event: {event.event_type} - Waiting Users: {len(waiting_users)} - Speed: {speed/2:.1f} km/h")
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        # Ensure figure is drawn
        plt.draw()
        plt.pause(1)

        # # Wait for a keyboard press to advance to the next image.
        # # plt.waitforbuttonpress() returns True for key press, False for mouse click.
        # # Loop until a key press is detected.
        # try:
        #     while True:
        #         pressed = plt.waitforbuttonpress(timeout=-1)
        #         if pressed:  # only advance on keyboard press
        #             break
        # except Exception:
        #     # In environments where interactive blocking is not supported, fall back to a short pause
        #     plt.pause(2)


# ------------------------------------------------------------------------------
# SIMULATION ENGINE
# ------------------------------------------------------------------------------
class SimulationEngine:
    """Class representing the core of the simulation for the car-sharing system."""
    def __init__(self, seed: int, map_seed: int, num_cars: int, num_stations: int, max_station_capacity: int, simulation_time: float, 
                 max_arrival_rate: float, max_autonomy: int, charging_rate: float, min_autonomy: int, 
                 max_destination_charging_distance: float, max_pickup_distance: float, max_waiting_time: float, speed: float = 40.0, 
                 min_trip_distance: float = 30.0, max_daily_relocations: int = 1):
        # Set seed for map generation
        np.random.seed(map_seed)
        self.num_cars = min(num_cars, num_stations * max_station_capacity)
        self.num_stations = num_stations
        self.max_station_capacity = max_station_capacity
        self.simulation_time = simulation_time  # in minutes
        self.max_arrival_rate = max_arrival_rate  # in requests per minute
        self.max_autonomy = max_autonomy  # in hm
        self.charging_rate = charging_rate  # in hm per minute
        self.min_autonomy = min_autonomy  # in hm
        self.max_destination_charging_distance = max_destination_charging_distance  # in hm
        self.speed = speed  # average speed in km/h
        self.min_trip_distance = min_trip_distance  # in hm
        self.max_pickup_distance = max_pickup_distance  # in hm
        self.max_waiting_time = max_waiting_time  # in minutes
        self.max_daily_relocations = max_daily_relocations  # max relocations per day
        self.fes: FutureEventSet = FutureEventSet()  # Initialize future event set

        # Define min relocation interval
        self.min_relocation_interval: float = (24 * 60) / (self.max_daily_relocations + 1)  # in minutes

        # Define spatial probability maps
        self.population_prob_map: np.ndarray = MappingUtilities.create_population_density()
        self.workplace_prob_map: np.ndarray = MappingUtilities.create_population_density()
        parking_lot_prob_map = self.population_prob_map + self.workplace_prob_map
        parking_lot_prob_map /= np.sum(parking_lot_prob_map)
        self.parking_lot_prob_map: np.ndarray = parking_lot_prob_map

        # Plot probability maps for verification
        # MappingUtilities.plot_probability_map(self.population_prob_map, "Population Density Probability Map")
        # MappingUtilities.plot_probability_map(self.workplace_prob_map, "Workplace Density Probability Map")
        
        # Initialize stations
        self.stations: list[ChargingStation] = []
        for i in range(self.num_stations):
            # Sample location based on parking lot probability map
            location = MappingUtilities.get_random_location_from_prob_map(self.parking_lot_prob_map)
            # Capacity proportional to expected demand, min-max scaled
            capacity = int((self.parking_lot_prob_map[location[1], location[0]] - min(self.parking_lot_prob_map.flatten())) / \
                       (max(self.parking_lot_prob_map.flatten()) - min(self.parking_lot_prob_map.flatten())) * \
                       (self.max_station_capacity - 1) + 1)
            station = ChargingStation(station_id=i, location=location, capacity=capacity)
            self.stations.append(station)
        # Compute total capacity
        self.total_capacity: int = sum(station.capacity for station in self.stations)
        # Increase capacity if not enough for all cars
        while self.total_capacity < self.num_cars:
            location = MappingUtilities.get_random_location_from_prob_map(self.parking_lot_prob_map)
            # Find nearest station and increase its capacity by 1 if not already at max
            nearest_station = None
            min_distance = float('inf')
            for station in self.stations:
                distance_to_station = ((location[0] - station.location[0]) ** 2 + (location[1] - station.location[1]) ** 2) ** 0.5
                if distance_to_station < min_distance:
                    min_distance = distance_to_station
                    nearest_station = station
            if nearest_station is not None and nearest_station.capacity < self.max_station_capacity:
                nearest_station.capacity += 1
                self.total_capacity += 1

        # Compute relocation tolerances
        self.waiting_queue_tolerance: float = 7 * np.exp(-(self.max_daily_relocations - 1) / 3)
        self.dispersed_cars_tolerance: float = .01 * self.num_cars * np.exp(-(self.max_daily_relocations - 1) / 3)

        # DEBUG: print number of cars
        print(f"Total parking station capacity: {self.total_capacity}, Number of cars allocated: {self.num_cars}")
        print(f"Waiting queue tolerance: {self.waiting_queue_tolerance}, Dispersed cars tolerance: {self.dispersed_cars_tolerance}")

        # MappingUtilities.plot_parking_prob_map_with_parking_stations(self.parking_lot_prob_map, self.stations)

        # Allocate cars to stations
        self.cars: list[Car] = []
        for i in range(self.num_cars):
            # Sample a station with available capacity
            station: ChargingStation = np.random.choice(self.stations)
            while station.is_full():
               station = np.random.choice(self.stations)
            car = Car(car_id=i, location=station.location, max_autonomy=self.max_autonomy,
                      charging_rate=self.charging_rate, min_autonomy=self.min_autonomy,
                      max_destination_charging_distance=self.max_destination_charging_distance,
                      speed=self.speed, station_id=station.station_id)
            station.park_car(car.car_id)
            self.cars.append(car)

        # Change seed for simulation events
        np.random.seed(seed)
        self.car_sharing_system = CarSharingSystem(cars=self.cars, stations=self.stations)

    def event_loop(self):
        """Main event loop for the simulation."""
        # Sample first user location and destination ensuring minimum trip distance
        first_user_location = MappingUtilities.get_random_location_from_prob_map(self.population_prob_map)
        travel_distance = 0.0
        while travel_distance < self.min_trip_distance:
            first_user_destination = MappingUtilities.get_random_location_from_prob_map(self.workplace_prob_map)
            travel_distance = ((first_user_location[0] - first_user_destination[0]) ** 2 + \
                               (first_user_location[1] - first_user_destination[1]) ** 2) ** 0.5
        first_user = User(user_id=1, location=first_user_location, destination=first_user_destination,
                          max_waiting_time=self.max_waiting_time, max_pickup_distance=self.max_pickup_distance, request_time=0.0)
        self.car_sharing_system.add_active_user(first_user)
        # Initialize simulation variables
        current_time: float = 0.0
        day: int = -1
        max_cars_to_relocate: int = self.num_cars // 4
        first_event = Event(EventType.USER_REQUEST, current_time, user_id=1)
        self.fes.schedule(first_event)

        # Main event loop
        while not self.fes.is_empty():
            event: Event = self.fes.get_next_event()
            # Stopping condition
            if event is None or event.time > self.simulation_time:
                break
            current_time = event.time
            # Charge cars during the time interval
            self.car_sharing_system.charge_cars(current_time - self.car_sharing_system.last_event_time)
            # Update speed based on number of active cars
            new_speed = self.speed * np.exp(-self.car_sharing_system.last_used_car_count / (0.1 * self.num_cars))
            self.car_sharing_system.update_speed(new_speed)
            # Update time-weighted statistics
            self.car_sharing_system.update_time_weighted_statistics(current_time)
            # Sample statistics
            self.car_sharing_system.sample_statistics(current_time)

            # Update day count
            if current_time // (24 * 60) > day:
                day += 1
                # Reset daily relocations
                remaining_daily_relocations: int = self.max_daily_relocations

            """
            Update arrival rate based on time of day:
                . 00:00 - 06:00 : 4%
                . 06:00 - 07:00 : 30%
                . 07:00 - 09:00 : 100% (commuting to work)
                . 09:00 - 12:00 : 50%
                . 12:00 - 17:00 : 30%
                . 17:00 - 19:00 : 100% (commuting back home)
                . 19:00 - 21:00 : 50%
                . 21:00 - 00:00 : 20%
            Define also the target probability map for relocations accordingly:
                . 00:00 - 12:00 : relocate cars near home (population density)
                . 12:00 - 24:00 : relocate cars near work (workplace density)
            """
            if 0 <= current_time % (24 * 60) <= 6 * 60:
                arrival_rate = self.max_arrival_rate * 0.04
                total_cars_to_relocate = int(0.04 * max_cars_to_relocate)
                target_prob_map = self.population_prob_map
            elif 6 * 60 < current_time % (24 * 60) <= 7 * 60:
                arrival_rate = self.max_arrival_rate * 0.3
                total_cars_to_relocate = int(0.3 * max_cars_to_relocate)
                target_prob_map = self.population_prob_map
            elif 7 * 60 < current_time % (24 * 60) <= 9 * 60:
                arrival_rate = self.max_arrival_rate * 1.0
                total_cars_to_relocate = int(1.0 * max_cars_to_relocate)
                target_prob_map = self.population_prob_map
            elif 9 * 60 < current_time % (24 * 60) <= 12 * 60:
                arrival_rate = self.max_arrival_rate * 0.5
                total_cars_to_relocate = int(0.5 * max_cars_to_relocate)
                target_prob_map = self.population_prob_map
            elif 12 * 60 < current_time % (24 * 60) <= 17 * 60:
                arrival_rate = self.max_arrival_rate * 0.3
                total_cars_to_relocate = int(0.3 * max_cars_to_relocate)
                target_prob_map = self.workplace_prob_map
            elif 17 * 60 < current_time % (24 * 60) <= 19 * 60:
                arrival_rate = self.max_arrival_rate * 1.0
                total_cars_to_relocate = int(1.0 * max_cars_to_relocate)
                target_prob_map = self.workplace_prob_map
            elif 19 * 60 < current_time % (24 * 60) <= 21 * 60:
                arrival_rate = self.max_arrival_rate * 0.5
                total_cars_to_relocate = int(0.5 * max_cars_to_relocate)
                target_prob_map = self.workplace_prob_map
            else:
                arrival_rate = self.max_arrival_rate * 0.2
                total_cars_to_relocate = int(0.2 * max_cars_to_relocate)
                target_prob_map = self.workplace_prob_map
            
            # Plot simulation state for debugging
            # waiting_clients = self.car_sharing_system.get_waiting_clients()
            # MappingUtilities.plot_simulation_state(event, self.cars, event.user_id and self.car_sharing_system.get_active_user(event.user_id) or None, self.stations, waiting_users=waiting_clients, dispersed_cars=self.car_sharing_system.get_dispersed_cars(), speed=new_speed, max_pickup_distance=self.max_pickup_distance)

            # Handle event based on its type
            if event.event_type == EventType.USER_REQUEST:
                EventHandler.handle_user_request(event, self.cars, self.car_sharing_system,
                                                self.stations, self.fes, current_time,
                                                arrival_rate, self.min_trip_distance, self.population_prob_map,
                                                self.workplace_prob_map, self.waiting_queue_tolerance,
                                                remaining_daily_relocations, self.car_sharing_system.last_relocation_time, 
                                                self.min_relocation_interval)
            elif event.event_type == EventType.USER_ABANDON:
                EventHandler.handle_user_abandon(event, self.car_sharing_system)
            elif event.event_type == EventType.CAR_PARKING:
                EventHandler.handle_car_parking(event, self.cars, self.car_sharing_system, self.stations, self.fes,
                                                current_time, self.dispersed_cars_tolerance,
                                                remaining_daily_relocations, self.car_sharing_system.last_relocation_time, 
                                                self.min_relocation_interval)
            elif event.event_type == EventType.CAR_AVAILABLE:
                EventHandler.handle_car_available(event, self.cars, self.stations, self.car_sharing_system,
                                                 self.fes, current_time)
            elif event.event_type == EventType.CAR_RELOCATE:
                # DEBUG: print relocation event
                # print(f"Relocation event at Day {int(event.time // 1440)}, {int((event.time % 1440) // 60):02}:{int(event.time % 60):02}:{int(event.time % 1 * 60):02} - Cars to relocate: {total_cars_to_relocate}")
                relocated = EventHandler.handle_car_relocate(event, self.cars, self.stations, self.car_sharing_system,
                                                            self.fes, total_cars_to_relocate, target_prob_map)
                # If relocation was performed, update stats and decrement daily remaining relocations
                if relocated:
                    self.car_sharing_system.relocation_performed(current_time, total_cars_to_relocate)
                    remaining_daily_relocations -= 1
                
    def get_statistics(self) -> dict[str, float]:
        """Compute and return key statistics of the simulation."""
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
            "total_relocations": self.car_sharing_system.total_relocations,
            "average_number_of_relocated_cars_per_relocation": self.car_sharing_system.total_relocated_cars / \
            max(1, self.car_sharing_system.total_relocations)
        }
        return stats
    
    def plot_time_series(self, time_series: tuple[list[float], list[float]], title: str, xlabel: str, ylabel: str):
        """Plot a time series given as (times, values)."""
        times, values = time_series
        plt.figure()
        plt.plot(times, values)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # Show a limited number of x-axis ticks to avoid overlap
        max_ticks = 20
        n = len(times)
        step = max(1, n // max_ticks)
        indices = list(range(0, n, step))
        selected_times = [times[i] for i in indices]
        # Format x-ticks as "day-hour:minute"
        x_ticks = [f"{int(t // (24 * 60))}-{int(t // 60 % 24):02}:{int(t % 60):02}" for t in selected_times]
        plt.xticks(ticks=selected_times, labels=x_ticks, rotation=45)
        plt.grid()
        plt.show()

    def print_statistics(self) -> tuple[tuple[list[float], list[float]], tuple[list[float], list[float]]]:
        """Print key statistics and plot relevant graphs."""
        # Get and print statistics
        stats = self.get_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")

        # Print histograms visually
        queue_histogram = self.car_sharing_system.get_queue_histogram()
        dispersed_cars_histogram = self.car_sharing_system.get_dispersed_cars_histogram()
        total_queue_samples = len(self.car_sharing_system.sampled_waiting_queue_lengths)
        visual_queue_histogram = {k: int((v / total_queue_samples) * 50) for k, v in queue_histogram.items()}
        total_dispersed_samples = len(self.car_sharing_system.sampled_num_dispersed_cars)
        visual_dispersed_cars_histogram = {k: int((v / total_dispersed_samples) * 50) for k, v in dispersed_cars_histogram.items()}
        print("\nWaiting Queue Length Histogram:")
        for length in sorted(visual_queue_histogram.keys()):
            print(f"Length {length}: {'*' * visual_queue_histogram[length]}")
        print("\nDispersed Cars Histogram:")
        for num in sorted(visual_dispersed_cars_histogram.keys()):
            print(f"Number {num}: {'*' * visual_dispersed_cars_histogram[num]}")

        # Retrieve time series data and return for further processing if needed
        queue_length_series = self.car_sharing_system.get_avg_waiting_queue_length_over_time()
        perc_abandoned_series = self.car_sharing_system.get_percent_abandoned_requests_over_time()
        return queue_length_series, perc_abandoned_series


# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # SIMULATION PARAMETERS
    SEED = [0, 42, 69]
    MAP_SEED = [0, 42, 69]
    NUM_CARS = [250]
    NUM_STATIONS = [85]
    MAX_STATION_CAPACITY = [6]
    SIMULATION_TIME = [30 * 24 * 60]  # in minutes
    MAX_ARRIVAL_RATE = [1.0]  # in requests per minute
    MAX_AUTONOMY = [1500]  # in hm
    CHARGING_RATE = [8]  # in hm per minute
    MIN_AUTONOMY = [200]  # in hm
    MAX_DESTINATION_CHARGING_DISTANCE = [4]  # in hm
    MAX_PICKUP_DISTANCE = [8]  # in hm
    MAX_WAITING_TIME = [12]  # in minutes
    AVERAGE_SPEED = [60.0]  # in km/h
    MIN_TRIP_DISTANCE = [30.0]  # in hm
    MAX_DAILY_RELOCATIONS = [2]  # max relocations per day

    # Iterate over all combinations of parameter lists and run the simulation for each configuration
    all_configs = list(product(
        SEED,
        MAP_SEED,
        NUM_CARS,
        NUM_STATIONS,
        MAX_STATION_CAPACITY,
        SIMULATION_TIME,
        MAX_ARRIVAL_RATE,
        MAX_AUTONOMY,
        CHARGING_RATE,
        MIN_AUTONOMY,
        MAX_DESTINATION_CHARGING_DISTANCE,
        MAX_PICKUP_DISTANCE,
        MAX_WAITING_TIME,
        AVERAGE_SPEED,
        MIN_TRIP_DISTANCE,
        MAX_DAILY_RELOCATIONS
    ))
    # Initialize lists to store results
    run_results = []
    queue_length_series = []
    perc_abandoned_series = []
    for idx, config in enumerate(all_configs, start=1):
        (seed_val, map_seed_val, num_cars_val, num_stations_val, max_station_capacity_val,
         simulation_time_val, max_arrival_rate_val, max_autonomy_val, charging_rate_val,
         min_autonomy_val, max_destination_charging_distance_val, max_pickup_distance_val,
         max_waiting_time_val, average_speed_val, min_trip_distance_val,
         max_daily_relocations_val) = config

        # Convert parameters to appropriate types
        seed_i = int(seed_val)
        map_seed_i = int(map_seed_val)
        num_stations_i = int(num_stations_val)
        max_station_capacity_i = int(max_station_capacity_val)
        simulation_time_i = float(simulation_time_val)
        max_arrival_rate_i = float(max_arrival_rate_val)
        max_autonomy_i = int(max_autonomy_val)
        charging_rate_i = float(charging_rate_val)
        min_autonomy_i = int(min_autonomy_val)
        max_destination_charging_distance_i = float(max_destination_charging_distance_val)
        max_pickup_distance_i = float(max_pickup_distance_val)
        max_waiting_time_i = float(max_waiting_time_val)
        average_speed_i = float(average_speed_val)
        min_trip_distance_i = float(min_trip_distance_val)
        max_daily_relocations_i = int(max_daily_relocations_val)
        num_cars_i = min(int(num_cars_val), num_stations_i * max_station_capacity_i)
        
        # Print configuration
        print("\n" + "=" * 80)
        print(f"Run {idx}/{len(all_configs)} - Configuration:")
        print(f" seed={seed_i}, map_seed={map_seed_i}, num_cars={num_cars_i}, num_stations={num_stations_i}, max_station_capacity={max_station_capacity_i}")
        print(f" simulation_time={simulation_time_i}, max_arrival_rate={max_arrival_rate_i}, max_autonomy={max_autonomy_i}")
        print(f" charging_rate={charging_rate_i}, min_autonomy={min_autonomy_i}, max_destination_charging_distance={max_destination_charging_distance_i}")
        print(f" max_pickup_distance={max_pickup_distance_i}, max_waiting_time={max_waiting_time_i}, average_speed={average_speed_i}")
        print(f" min_trip_distance={min_trip_distance_i}, max_daily_relocations={max_daily_relocations_i}")
        print("=" * 80)

        # Close any open figures to avoid GUI buildup
        try:
            plt.close('all')
        except Exception:
            pass

        # Launch simulation
        start_time = time.time()
        simulation_engine = SimulationEngine(
            seed=seed_i,
            map_seed=map_seed_i,
            num_cars=num_cars_i,
            num_stations=num_stations_i,
            max_station_capacity=max_station_capacity_i,
            simulation_time=simulation_time_i,
            max_arrival_rate=max_arrival_rate_i,
            max_autonomy=max_autonomy_i,
            charging_rate=charging_rate_i,
            min_autonomy=min_autonomy_i,
            max_destination_charging_distance=max_destination_charging_distance_i,
            max_pickup_distance=max_pickup_distance_i,
            max_waiting_time=max_waiting_time_i,
            speed=average_speed_i,
            min_trip_distance=min_trip_distance_i,
            max_daily_relocations=max_daily_relocations_i
        )
        simulation_engine.event_loop()
        elapsed = time.time() - start_time
        stats = simulation_engine.get_statistics()
        print(f"Run {idx} completed in {elapsed:.2f} s. Stats:")
        queue_length_series_i, perc_abandoned_series_i = simulation_engine.print_statistics()
        queue_length_series.append(queue_length_series_i)
        perc_abandoned_series.append(perc_abandoned_series_i)
        # for k, v in stats.items():
        #     print(f"  {k}: {v}")
        # run_results.append({"config": config, "stats": stats, "time_s": elapsed})

    # Optionally, print a brief summary of all runs
    print("\nSummary of all runs:")
    for i, res in enumerate(run_results, start=1):
        cfg = res.get("config")
        if "stats" in res:
            print(f"Run {i}: config={cfg} time_s={res['time_s']:.2f} total_user_requests={res['stats'].get('total_user_requests')}, total_abandoned={res['stats'].get('total_abandoned_requests')}")
        else:
            print(f"Run {i}: config={cfg} ERROR={res.get('error')}")

    # Plot aggregated time series across all runs (optional)
    def plot_multiple_time_series(time_series: list[tuple[list[float], list[float]]], title: str, xlabel: str, ylabel: str):
        plt.figure()
        for series in time_series:
            times, values = series
            plt.plot(times, values, label=f'Run {time_series.index(series)+1}')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # Show a limited number of x-axis ticks to avoid overlap
        max_ticks = 20
        n = len(times)
        step = max(1, n // max_ticks)
        indices = list(range(0, n, step))
        selected_times = [times[i] for i in indices]
        x_ticks = [f"{int(t // (24 * 60))}-{int(t // 60 % 24):02}:{int(t % 60):02}" for t in selected_times]
        plt.xticks(ticks=selected_times, labels=x_ticks, rotation=45)
        plt.legend()
        plt.grid()
        plt.show()
    plot_multiple_time_series(queue_length_series, "Average Waiting Queue Length Over Time (All Runs)", "Time (minutes)", "Average Queue Length")
    plot_multiple_time_series(perc_abandoned_series, "Percentage of Abandoned Requests Over Time (All Runs)", "Time (minutes)", "Percentage Abandoned (%)")

    # Save detailed results to a text file
    with open("simulation_run_results.txt", "w") as f:
        for i, res in enumerate(run_results, start=1):
            cfg = res.get("config")
            f.write(f"Run {i}:\n")
            f.write(f" Configuration: seed={cfg[0]}, num_cars={cfg[1]}, num_stations={cfg[2]}, max_station_capacity={cfg[3]},\n")
            f.write(f"                simulation_time={cfg[4]}, max_arrival_rate={cfg[5]}, max_autonomy={cfg[6]}, charging_rate={cfg[7]},\n")
            f.write(f"                min_autonomy={cfg[8]}, max_destination_charging_distance={cfg[9]}, max_pickup_distance={cfg[10]},\n")
            f.write(f"                max_waiting_time={cfg[11]}, average_speed={cfg[12]}, min_trip_distance={cfg[13]}, max_daily_relocations={cfg[14]}\n")
            if "stats" in res:
                f.write(f" Time (s): {res['time_s']:.2f}\n")
                f.write(" Statistics:\n")
                for k, v in res['stats'].items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f" ERROR: {res.get('error')}\n")
            f.write("\n")