#---other packages
import json
import re
#---------MODEL
from pydantic import BaseModel
from collections import defaultdict
from typing import List,Optional


class Unit(BaseModel):
    id: str
    name: str
    type: str  # e.g. "infantry", "armor", "air support"
    strength: int  # combat effectiveness, 0–100
    position: tuple[float, float] = (0.0, 0.0)  # (x, y) coords
    allegiance: str  # "friendly" or "enemy"
    status: Optional[str] = "active"  # e.g. "active", "retreating", "destroyed"

class TerrainFeature(BaseModel):
    type: str  # e.g. "hill", "forest", "building"
    position: tuple[float, float] = (0.0, 0.0)
    size: float  # area/radius in meters

class Terrain(BaseModel):
    type: str  # e.g. "urban", "desert", "jungle"
    features: List[TerrainFeature]
    dimensions: tuple[float, float] = (10.0, 10.0)  # map size in meters (width, height)

class Objective(BaseModel):
    id: str
    description: str
    controlling_unit_ids: List[str] = []
    completed: bool = False
    location: Optional[tuple[float, float]] = None
    priority: Optional[int] = 10   # Lower number = more critical

class BattleEvent(BaseModel):
    timestamp: object  # e.g. "00:05", "12:03 PM"
    description: str
    involved_units: List[str] = []
    location: tuple[float, float] = (0.0, 0.0)
    event_type: str  # e.g. "move", "fire", "retreat", "reinforce"

class Scenario(BaseModel):
    title: str
    description: str
    terrain: Terrain
    units: List[Unit]
    objectives: List[Objective]
    timeline: List[BattleEvent]

# Define the keyword labels we want to tokenize by
KEYWORDS = ['Title', 'Description', 'Unit', 'Feature', 'Objective', 'Event']

def tokenize_by_keyword(text: str):
    text = text.replace("minus", "-")  # Normalize voice-to-text quirks
    pattern = r'\b(' + '|'.join(KEYWORDS) + r')\b(?:\s+is)?'
    tokens = re.split(pattern, text)

    # re.split gives us a list like: ['', 'Title', ' Operation X.', 'Unit', ' ID equals ...', ...]
    # We need to stitch it back together as {keyword: [chunks]}
    data = defaultdict(list)

    current_key = None
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token in KEYWORDS:
            current_key = token
        elif current_key:
            data[current_key].append(token)

    return data

# Run the tokenizer
#tokenized_data = tokenize_by_keyword(voice_text)

#---PARSE each Object out of the voice to text -----
def parse_unit(chunk)-> Unit:
    #Goood Parse of the Unit Data
    print(chunk)
    unit_id = re.compile(r"ID\s?\w+\s?(\w+)\.")
    unit_name = re.compile(r"Name\s?\w+\s?(.+?)\.")
    unit_type= re.compile(r"Type\s?\w+\s?(.+?)\.")
    unit_ste = re.compile(r"Strength\s?\w+\s?(\d+)\.")
    unit_all = re.compile(r"Allegiance\s?\w+\s?(\w+)\.")
    unit_x = re.compile(r"X\s?\w+\s?(-?\d+(?:\.\d+)?)\.")
    unit_y = re.compile(r'Y\s+\w+\s+(-?\s*\d+(?:\.\d+)?)\.')
    unit_status = re.compile(r"Status\s?\w+\s?(\w+)\.")
    return Unit(
          id=re.findall(unit_id,chunk)[0],
          name=re.findall(unit_name, chunk)[0],
          type=re.findall(unit_type,chunk)[0],
          strength=re.findall(unit_ste,chunk)[0],
          allegiance=re.findall(unit_all,chunk)[0],
          position=(float(re.findall(unit_x,chunk)[0].replace(" ","")), float(re.findall(unit_y,chunk)[0].replace(" ",""))),
          status=re.findall(unit_status,chunk)[0])


def parse_feature(chunk) -> TerrainFeature:
    #example:'Type equals Bunker. X equals 0. Y equals 2. Size equals 10.'
    print(chunk)
    feat_type = re.compile(r'Type\s?\w+\s?(.*?)\.')
    feat_x = re.compile(r'X\s?\w+\s?(-?\d+(?:\.\d+)?)\.')
    feat_y = re.compile(r'Y\s+\w+\s+(-?\s*\d+(?:\.\d+)?)\.')
    feat_size = re.compile(r'Size\s+\w+\s(\d+)\.')
    return TerrainFeature(
        type=re.findall(feat_type,chunk)[0],
        position=(float(re.findall(feat_x,chunk)[0].replace(" ","")), float(re.findall(feat_y,chunk)[0].replace(" ",""))),
        size=float(re.findall(feat_size,chunk)[0])
        )


def parse_objective(chunk) -> Objective:
    #Patterns that we need for objectives
    obj_id = re.compile(r'ID\s?\w+\s?(\w+)\.')
    obj_desc = re.compile(r'Desk\s?\w+\s?(.+?)\.')
    obj_x = re.compile(r'X\s?\w+\s?(-?\d+(?:\.\d+)?)\.')
    obj_y = re.compile(r'Y\s+\w+\s+(-?\s*\d+(?:\.\d+)?)\.')
    obj_prior = re.compile(r'Priority\s?\w+\s?(\d+)\.')

    return  Objective (
        id=re.findall(obj_id,chunk)[0],
        description=re.findall(obj_desc,chunk)[0],
        location=(float(re.findall(obj_x,chunk)[0].replace(" ","")), float(re.findall(obj_y,chunk)[0].replace(" ",""))),
        priority=int(re.findall(obj_prior,chunk)[0])
    )


def parse_event(chunk: str) -> BattleEvent:

    #Patterns to capture battle events
    battle_time = re.compile(r'Time\s?\w+\s?(\d+\.\d+)\.')
    battle_desc = re.compile(r'Desk\s?\w+\s?(.+?)\.')
    battle_units = re.compile(r'Units\s?\w+\s?(.+?)\.')
    battle_type = re.compile(r'Type\s?\w+\s?(.+?)\.')
    battle_x = re.compile(r'X\s?\w+\s?(-?\d+(?:\.\d+)?)\.')
    battle_y = re.compile(r'Y\s+\w+\s+(-?\s*\d+(?:\.\d+)?)\.')
    print(chunk)
    return BattleEvent(
        timestamp=re.findall(battle_time,chunk)[0],
        description=re.findall(battle_desc,chunk)[0],
        involved_units=list(re.findall(battle_units,chunk)[0].split(',')),
        event_type=re.findall(battle_type,chunk)[0].lower(),
        location=(float(re.findall(battle_x,chunk)[0].replace(" ","")), float(re.findall(battle_y,chunk)[0].replace(" ",""))),
    )


# -------------------- MAIN PARSER --------------------
def parse_scenario(text: str) -> Scenario:
    tokens = tokenize_by_keyword(text)
    print(tokens['Unit'])
    title = tokens['Title'][0] if tokens['Title'] else "Untitled Scenario"
    description = tokens['Description'][0] if tokens['Description'] else ""
    units = [parse_unit(chunk) for chunk in tokens['Unit']]
    features = [parse_feature(chunk) for chunk in tokens['Feature']]
    objectives = [parse_objective(chunk) for chunk in tokens['Objective']]
    timeline = [parse_event(chunk) for chunk in tokens['Event']]

    terrain = Terrain(type="unknown", features=features, dimensions=(10, 10))

    return Scenario(
        title=title,
        description=description,
        terrain=terrain,
        units=units,
        objectives=objectives,
        timeline=timeline
    )

def write_file(Scenario):
    # Write to file
    with open("scenario.json", "w") as f:
      json.dump(Scenario.dict(), f, indent=2)