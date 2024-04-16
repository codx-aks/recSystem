import csv
import random

interests = ["technology","sports","music","dance","entertainment","infotainment","food","outdoor","freeentry","competition","online"]
cities = [
    "Hyderabad", "Chennai", "Bangalore", "Coimbatore", "Mysore", "Kochi", "Thiruvananthapuram",
    "Visakhapatnam", "Vijayawada", "Tiruchirappalli", "Madurai", "Mangalore", "Kozhikode",
    "Warangal", "Nellore", "Tiruppur", "Kannur", "Kollam", "Thrissur", "Guntur", "Belgaum",
    "Nizamabad", "Bellary", "Davangere", "Kurnool", "Kakinada", "Anantapur", "Tumkur", "Thanjavur",
    "Kolar", "Alappuzha", "Shimoga", "Hosur", "Rajahmundry", "Kadapa", "Chittoor", "Karaikudi",
    "Hassan", "Salem", "Gulbarga", "Palakkad", "Pathanamthitta", "Kottayam", "Kasargod", "Karur",
    "Ooty", "Mandya", "Chikkamagaluru"
]

def generate_event(event_id):
  name = f"Event{event_id}"
  description = f"description{event_id}"
  num_interests = random.randint(1, 3)  # Choose a random number of interests (1-4)
  event_type = random.sample(interests, num_interests)  # Sample that many interests
  type = "|".join(event_type)
  people_count = random.randint(10, 500)
  age_recommended = random.randint(1, 50)
  days_left = random.randint(1, 40)
  location = random.choice(cities)
  return [event_id, name, description, type, people_count, age_recommended, days_left, location]

# Open the CSV file for writing
with open("events.csv", "a", newline="") as csvfile:
  writer = csv.writer(csvfile)

  # Write the header row
  writer.writerow(["eventId", "name", "description", "type", "peoplecount", "agerecommended", "daysleft", "location"])

  # Generate and write 300 events
  for event_id in range(301, 25001):
    event_data = generate_event(event_id)
    writer.writerow(event_data)

print("CSV file populated with 300 events!")
